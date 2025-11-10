"""
CGI-Stereo: Context and Geometry Interaction Stereo Matching Network
基于上下文与几何交互的立体匹配网络

该网络实现了一种创新的立体视觉深度估计算法，通过以下核心组件实现高精度视差估计：
1. 特征提取：基于 MobileNetV2 的多尺度特征提取器
2. 特征上采样：通过反卷积实现多尺度特征融合
3. 上下文几何融合：将语义特征与几何代价体进行交互融合
4. 沙漏网络：通过多尺度编解码结构优化代价体
5. 空间金字塔：实现亚像素级视差细化

论文核心思想：通过 Context-Geometry Fusion (CGF) 模块，将图像的语义上下文信息
与立体匹配的几何约束信息有效融合，提升视差估计的准确性和鲁棒性。
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *  # 包含 BasicConv, Conv2x, build_norm_correlation_volume 等自定义模块
import math
import gc
import time
import timm  # PyTorch Image Models (TIMM) 库，用于预训练模型

class SubModule(nn.Module):
    """
    基础模块类，提供权重初始化功能
    """
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        """
        权重初始化方法，使用 He 初始化
        对于卷积层：使用正态分布初始化，std = sqrt(2/n)
        对于批归一化层：权重初始化为1，偏置初始化为0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    """
    特征提取网络，基于 MobileNetV2 的多尺度特征提取器

    使用预训练的 MobileNetV2_100 作为骨干网络，提取4个不同尺度的特征图：
    - x4: 1/4分辨率特征，通道数24，包含中等尺度信息
    - x8: 1/8分辨率特征，通道数32，包含较粗尺度信息
    - x16: 1/16分辨率特征，通道数96，包含粗尺度信息
    - x32: 1/32分辨率特征，通道数160，包含最粗尺度的语义信息

    选择 MobileNetV2 的原因：轻量级、计算效率高、特征表达能力强
    """
    def __init__(self):
        super(Feature, self).__init__()
        pretrained = True
        self.features_only = True
        # 使用 timm 库创建预训练的 MobileNetV2，features_only=True 只返回特征图
        self.model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=self.features_only)
        # self.model = timm.create_model('mobilenetv3_small_100', pretrained=pretrained, features_only=self.features_only)

        # 选择的 MobileNetV2 层索引：[1,2,3,5,6]
        # 对应的输出通道数：[16, 24, 32, 96, 160]
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        # 复制 MobileNetV2 的基础结构
        self.conv_stem = self.model.conv_stem
        self.bn1 = self.model.bn1
        if hasattr(self.model, 'act1'):
            self.act1 = self.model.act1
        # self.act1 = self.model.act1

        # 构建不同深度的模块块
        self.block0 = torch.nn.Sequential(*self.model.blocks[0:layers[0]])  # 输出16通道
        self.block1 = torch.nn.Sequential(*self.model.blocks[layers[0]:layers[1]])  # 输出24通道
        self.block2 = torch.nn.Sequential(*self.model.blocks[layers[1]:layers[2]])  # 输出32通道
        self.block3 = torch.nn.Sequential(*self.model.blocks[layers[2]:layers[3]])  # 输出96通道
        self.block4 = torch.nn.Sequential(*self.model.blocks[layers[3]:layers[4]])  # 输出160通道
        '在立体匹配（Stereo）或深度估计任务中，我们一般只取到 block5 输出（160通道）为最高层特征，'
        # 注：这里的 deconv32_16 在当前类中未使用，可能是设计遗留
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)

    def forward(self, x):
        """
        前向传播，提取多尺度特征

        Args:
            x: 输入图像张量 [B, 3, H, W]

        Returns:
            List: 包含4个尺度特征图的列表 [x4, x8, x16, x32]
        """
        # if self.features_only:
        #     # 如果只需要特征图，直接调用模型
        # features = self.model(x)[:4]
        #     return [features[0], features[1], features[2], features[3]]
        # 第一阶段：初始卷积处理
        if hasattr(self, 'act1'):
            x1 = self.act1(self.bn1(self.conv_stem(x)))
        else:
            x1 = self.bn1(self.conv_stem(x))  # [B, 32, H/2, W/2]

        # 第二阶段：通过不同深度的模块提取多尺度特征
        x2 = self.block0(x1)   # [B, 16, H/2, W/2] - 注意：这里输出是16通道
        x4 = self.block1(x2)  # [B, 24, H/4, W/4] - 1/4分辨率
        x8 = self.block2(x4)  # [B, 32, H/8, W/8]  - 1/8分辨率
        x16 = self.block3(x8) # [B, 96, H/16, W/16] - 1/16分辨率
        x32 = self.block4(x16)# [B, 160, H/32, W/32] - 1/32分辨率

        return [x4, x8, x16, x32]

class FeatUp(SubModule):
    """
    特征上采样模块，实现多尺度特征融合

    通过自顶向下的路径，将深层语义特征与浅层细节特征进行融合：
    1. x32(1/32) -> x16(1/16): 使用反卷积上采样并与x16特征拼接
    2. x16 -> x8(1/8): 继续上采样并与x8特征拼接
    3. x8 -> x4(1/4): 最终上采样并与x4特征拼接

    这种设计充分利用了深层特征的语义信息和浅层特征的细节信息，
    为后续的立体匹配提供了更加丰富的特征表示。
    """
    def __init__(self):
        super(FeatUp, self).__init__()
        # 特征通道配置 [16, 24, 32, 96, 160] 对应不同尺度
        chans = [16, 24, 32, 96, 160]

        # 上采样路径：从深层到浅层的反卷积层
        # Conv2x: 反卷积+拼接操作，deconv=True表示使用转置卷积，concat=True表示与对应层特征拼接
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)  # 160->96
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)  # 96*2->32 (拼接后通道翻倍)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)   # 32*2->24

        # 最终融合层：对1/4分辨率的特征进行进一步处理
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)  # 24*2->48

        self.weight_init()

    def forward(self, featL, featR=None):
        """
        前向传播，实现左右图像特征的上采样融合

        Args:
            featL: 左图像特征列表 [x4, x8, x16, x32]
            featR: 右图像特征列表 [y4, y8, y16, y32]

        Returns:
            Tuple: (融合后的左特征, 融合后的右特征)
                  每个都是4层特征的列表 [x4', x8', x16', x32']
        """
        # 解包多尺度特征
        x4, x8, x16, x32 = featL  # 左图像特征
        y4, y8, y16, y32 = featR  # 右图像特征

        # 第一级上采样：1/32 -> 1/16
        # 将高层次的语义特征上采样并与中等层次特征融合
        x16 = self.deconv32_16(x32, x16)  # 输入: (160, 96) -> 输出: 96*2通道
        y16 = self.deconv32_16(y32, y16)

        # 第二级上采样：1/16 -> 1/8
        # 继续融合更细致的特征信息
        x8 = self.deconv16_8(x16, x8)  # 输入: (96*2, 32) -> 输出: 32*2通道
        y8 = self.deconv16_8(y16, y8)

        # 第三级上采样：1/8 -> 1/4
        # 获得最高分辨率的融合特征
        x4 = self.deconv8_4(x8, x4)    # 输入: (32*2, 24) -> 输出: 24*2通道
        y4 = self.deconv8_4(y8, y4)

        # 对1/4分辨率特征进行最终处理
        x4 = self.conv4(x4)  # 进一步提取1/4分辨率的特征
        y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]


class Context_Geometry_Fusion(SubModule):
    """
    上下文几何融合模块 (CGF) - CGI-Stereo 的核心创新组件

    该模块实现了图像语义上下文信息与立体几何约束信息的有效融合：
    1. 语义特征提取：将2D图像特征转换为适合与3D代价体融合的表示
    2. 注意力机制：通过3D卷积计算注意力权重，增强相关区域的响应
    3. 特征融合：使用门控机制将语义特征注入到几何代价体中
    4. 上下文聚合：通过3D卷积进一步聚合融合后的特征

    这种设计让网络能够利用图像的语义信息来指导立体匹配，
    特别是在纹理缺失、遮挡等困难区域提升匹配质量。
    """
    def __init__(self, cv_chan, im_chan):
        """
        初始化上下文几何融合模块

        Args:
            cv_chan: 代价体通道数 (cost volume channels)
            im_chan: 图像特征通道数 (image feature channels)
        """
        super(Context_Geometry_Fusion, self).__init__()

        # 语义特征提取网络：将2D图像特征投影到代价体空间
        self.semantic = nn.Sequential(
            # 首先减少通道数，降低计算复杂度
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            # 将特征维度映射到代价体通道数
            nn.Conv2d(im_chan//2, cv_chan, 1)
        )

        # 注意力计算网络：生成融合权重
        # 使用3D卷积处理代价体与语义特征的和
        self.att = nn.Sequential(
            # 3D卷积层：kernel_size=(1,5,5) 只在空间维度卷积，保持视差维度
            BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True,
                     kernel_size=(1,5,5), padding=(0,2,2), stride=1, dilation=1),
            # 1x1x1卷积：调整特征维度
            nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # 特征聚合网络：进一步处理融合后的特征
        self.agg = BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True,
                            kernel_size=(1,5,5), padding=(0,2,2), stride=1, dilation=1)

        self.weight_init()

    def forward(self, cv, feat):
        """
        前向传播，实现上下文与几何信息的融合

        Args:
            cv: 3D代价体张量 [B, C, D, H, W]，D为视差维度
            feat: 2D图像特征张量 [B, C, H, W]

        Returns:
            Tensor: 融合后的3D特征张量 [B, C, D, H, W]
        """
        # 步骤1：语义特征处理
        # 将2D特征转换为3D特征 (增加视差维度)
        feat = self.semantic(feat).unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

        # 步骤2：注意力权重计算
        # 将语义特征与代价体相加，通过注意力网络生成融合权重
        att = self.att(feat + cv)  # 广播机制：feat会在视差维度扩展

        # 步骤3：特征融合 (门控机制)
        # 使用sigmoid激活的注意力权重作为门控，控制语义特征的注入程度
        cv = torch.sigmoid(att) * feat + cv  # 残差连接 + 门控融合

        # 步骤4：上下文聚合
        # 通过3D卷积进一步聚合融合后的特征，增强表示能力
        cv = self.agg(cv)

        return cv


class hourglass_fusion(nn.Module):
    """
    沙漏融合网络 - 多尺度3D代价体优化模块

    该模块实现了一个3D编解码器结构，通过沙漏网络优化代价体：
    1. 编码路径：逐步下采样，提取不同尺度的特征
       - Level 1: C -> 2C (1/2分辨率)
       - Level 2: 2C -> 4C (1/4分辨率)
       - Level 3: 4C -> 6C (1/8分辨率)
    2. 上下文融合：在每个尺度上应用CGF模块，注入语义信息
    3. 解码路径：逐步上采样，融合浅层特征
       - Level 3 -> Level 2: 6C -> 4C
       - Level 2 -> Level 1: 4C -> 2C
       - Level 1 -> Output: 2C -> 1 (视差概率分布)

    设计理念：通过多尺度处理，网络能够在不同分辨率上捕捉匹配信息，
    同时结合语义上下文指导，提升代价体的质量。
    """
    def __init__(self, in_channels):
        """
        初始化沙漏融合网络

        Args:
            in_channels: 输入代价体通道数 (通常为8)
        """
        super(hourglass_fusion, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))


        self.CGF_32 = Context_Geometry_Fusion(in_channels*6, 160)
        self.CGF_16 = Context_Geometry_Fusion(in_channels*4, 192)
        self.CGF_8 = Context_Geometry_Fusion(in_channels*2, 64)

    def forward(self, x, imgs):
        """
        前向传播，实现多尺度代价体优化

        Args:
            x: 输入3D代价体 [B, C, D, H, W]
            imgs: 多尺度语义特征列表 [x4, x8, x16, x32]

        Returns:
            Tensor: 优化后的视差概率分布 [B, 1, D, H, W]
        """
        # === 编码阶段 ===
        conv1 = self.conv1(x)   # Level 1: [B, 2C, D/2, H/2, W/2]
        conv2 = self.conv2(conv1) # Level 2: [B, 4C, D/4, H/4, W/4]
        conv3 = self.conv3(conv2) # Level 3: [B, 6C, D/8, H/8, W/8]

        # === 最深层语义融合 ===
        conv3 = self.CGF_32(conv3, imgs[3])  # 注入1/32分辨率的语义特征
        conv3_up = self.conv3_up(conv3)      # 上采样到Level 2分辨率

        # === Level 2融合 ===
        # 跳跃连接：拼接上采样特征与编码特征
        conv2 = torch.cat((conv3_up, conv2), dim=1)  # [B, 8C, D/4, H/4, W/4]
        conv2 = self.agg_0(conv2)                    # 特征聚合降维
        conv2 = self.CGF_16(conv2, imgs[2])          # 注入1/16分辨率的语义特征
        conv2_up = self.conv2_up(conv2)              # 上采样到Level 1分辨率

        # === Level 1融合 ===
        # 跳跃连接：拼接上采样特征与编码特征
        conv1 = torch.cat((conv2_up, conv1), dim=1)  # [B, 4C, D/2, H/2, W/2]
        conv1 = self.agg_1(conv1)                    # 特征聚合降维
        conv1 = self.CGF_8(conv1, imgs[1])           # 注入1/8分辨率的语义特征

        # === 最终输出 ===
        conv = self.conv1_up(conv1)  # 上采样到原始分辨率，输出单通道视差概率分布

        return conv


class CGI_Stereo(nn.Module):
    """
    CGI-Stereo 主网络类

    基于上下文与几何交互的立体匹配网络，实现端到端的视差估计：

    核心组件：
    1. 特征提取：基于 MobileNetV2 的多尺度特征提取
    2. 特征融合：通过反卷积实现多尺度特征融合
    3. 代价体构建：基于归一化相关性的立体匹配
    4. 上下文融合：CGF 模块融合语义与几何信息
    5. 沙漏优化：3D 编解码器优化代价体
    6. 空间金字塔：亚像素级视差细化
    7. 视差回归：Top-K 加权求和的软回归

    网络流程：
    左右图像 -> 多尺度特征提取 -> 特征融合 -> 相关性计算 -> CGF融合 ->
    沙漏优化 -> 视差回归 -> 亚像素细化 -> 高分辨率视差图
    """
    def __init__(self, maxdisp):
        """
        初始化 CGI-Stereo 网络

        Args:
            maxdisp: 最大视差值，通常为192或256
        """
        super(CGI_Stereo, self).__init__()
        self.maxdisp = maxdisp

        # === 核心模块初始化 ===
        self.feature = Feature()      # 多尺度特征提取器
        self.feature_up = FeatUp()    # 特征上采样融合模块
        chans = [16, 24, 32, 96, 160]  # MobileNetV2 各层通道数

        # === 空间金字塔特征提取 (Stem Networks) ===
        # stem_2: 提取1/2分辨率的低级特征
        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),  # [B,3,H,W] -> [B,32,H/2,W/2]
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
        )

        # stem_4: 提取1/4分辨率的中级特征
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1), # [B,32,H/2,W/2] -> [B,48,H/4,W/4]
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
        )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )

        self.conv = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.semantic = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0, stride=1, bias=False))
        self.agg = BasicConv(8, 8, is_3d=True, kernel_size=(1,5,5), padding=(0,2,2), stride=1)
        self.hourglass_fusion = hourglass_fusion(8)
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)

    def forward(self, left, right):
        """
        前向传播，实现完整的立体匹配流程

        Args:
            left: 左图像 [B, 3, H, W]
            right: 右图像 [B, 3, H, W]

        Returns:
            List: 训练时返回 [高分辨率视差, 低分辨率视差]，推理时返回 [高分辨率视差]
        """
        # === 1. 多尺度特征提取 ===
        # 使用 MobileNetV2 提取左右图像的多尺度特征
        features_left = self.feature(left)   # [x4, x8, x16, x32]
        features_right = self.feature(right) # [y4, y8, y16, y32]
        print(features_left[0].shape)
        print(features_right[0].shape)

        # 特征融合：通过反卷积实现多尺度特征融合
        features_left, features_right = self.feature_up(features_left, features_right)

        # === 2. 空间金字塔特征提取 ===
        # 为后续的空间金字塔模块提取额外的低级特征
        stem_2x = self.stem_2(left)   # 1/2分辨率特征 [B, 32, H/2, W/2]
        stem_4x = self.stem_4(stem_2x)  # 1/4分辨率特征 [B, 48, H/4, W/4]
        stem_2y = self.stem_2(right)  # 右图像1/2分辨率特征
        stem_4y = self.stem_4(stem_2y) # 右图像1/4分辨率特征

        # === 3. 特征融合 ===
        # 将stem特征与MobileNetV2特征在1/4分辨率处融合
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)  # 通道数: 48+48=96
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        # === 4. 匹配特征提取 ===
        # 提取用于立体匹配的描述符特征
        match_left = self.desc(self.conv(features_left[0]))   # [B, 48, H/4, W/4]
        match_right = self.desc(self.conv(features_right[0])) # [B, 48, H/4, W/4]

        # === 5. 代价体构建 ===
        # 基于归一化相关性构建3D代价体
        corr_volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp//4)  # [B, 1, D, H/4, W/4]
        corr_volume = self.corr_stem(corr_volume)  # 初步处理代价体 [B, 8, D, H/4, W/4]

        # 提取语义特征并扩展到3D空间
        feat_volume = self.semantic(features_left[0]).unsqueeze(2)  # [B, 8, H/4, W/4] -> [B, 8, 1, H/4, W/4]
        # === 6. 初始代价体构建 ===
        # 将语义特征与相关性代价体逐元素相乘，增强匹配位置的响应
        # 然后通过3D卷积聚合局部上下文信息
        volume = self.agg(feat_volume * corr_volume)  # [B, 8, D, H/4, W/4]

        # === 7. 沙漏网络优化 ===
        # 通过3D沙漏网络进一步优化代价体，融合多尺度语义信息
        cost = self.hourglass_fusion(volume, features_left)  # [B, 1, D, H/4, W/4]

        # === 8. 空间金字塔特征提取 (用于亚像素细化) ===
        # 提取高分辨率空间特征，用于后续的亚像素级视差细化
        xspx = self.spx_4(features_left[0])  # 处理1/4分辨率特征
        xspx = self.spx_2(xspx, stem_2x)     # 融合1/2分辨率特征
        spx_pred = self.spx(xspx)            # 生成9通道的空间偏移预测
        spx_pred = F.softmax(spx_pred, 1)    # 转换为概率分布 [B, 9, H, W]

        # === 9. 视差回归 (1/4分辨率) ===
        # 生成视差候选值 (0 到 maxdisp//4)
        disp_samples = torch.arange(0, self.maxdisp//4, dtype=cost.dtype, device=cost.device)
        disp_samples = disp_samples.view(1, self.maxdisp//4, 1, 1).repeat(
            cost.shape[0], 1, cost.shape[3], cost.shape[4])  # [B, D, H/4, W/4]

        # 基于代价体进行软回归：使用Top-K加权求和 (K=2)
        # 相比传统的argmin，这种方法能产生更平滑的视差图
        pred = regression_soft(cost.squeeze(1), disp_samples, 2)  # [B, H/4, W/4]

        # === 10. 亚像素级上下文上采样 ===
        # 利用空间金字塔预测的偏移权重，实现亚像素级视差细化
        # 将1/4分辨率的视差图上采样到全分辨率，同时保持边缘细节
        pred_up = context_upsample(pred, spx_pred)  # [B, H, W]


        # === 11. 输出处理 ===
        if self.training:
            # 训练模式：返回高分辨率和低分辨率两个尺度的视差图
            # 用于多尺度监督损失计算
            return [pred_up*4, pred.squeeze(1)*4]  # 乘以4恢复到原始视差尺度
        else:
            # 推理模式：只返回高分辨率视差图
            return [pred_up*4]

"""
=== 关键技术解析 ===

1. 为什么需要 spx_pred (空间金字塔)？
直接上采样低分辨率视差图会导致边缘模糊和细节丢失。
spx_pred 通过学习局部偏移权重，实现细节保留的亚像素级上采样，
特别在物体边缘和纹理区域效果显著。

2. regression_topk 的优势：
替代传统的 argmin 操作，通过加权插值得到更平滑、更准确的视差图。
Top-K (K=2) 策略能够有效处理视差模糊和噪声。

3. maxdisp//4 的设计原理：
- 在1/4分辨率构建代价体，减少计算量和内存占用
- 视差搜索范围相应缩小为原来的1/4
- 最终结果乘以4恢复到原始尺度

4. CGI-Stereo 的核心创新：
- Context-Geometry Fusion：将语义信息与几何约束有效融合
- 多尺度处理：在不同分辨率层次上优化匹配质量
- 亚像素细化：显著提升视差估计的精度

=== 网络优势 ===
1. 高精度：亚像素级视差估计，达到亚毫米级精度
2. 强鲁棒性：在弱纹理、遮挡区域表现优异
3. 计算效率：合理的复杂度，适合实时应用
4. 端到端训练：无需复杂的后处理
"""

# ================================================================================
# CGI-Stereo 网络测试代码
# ================================================================================

if __name__ == "__main__":
    import torch
    import time
    import numpy as np

    def print_separator(title):
        """打印分隔符和标题"""
        print("\n" + "="*60)
        print(f" {title}")
        print("="*60)

    def test_basic_functionality():
        """测试基本功能"""
        print_separator("1. CGI-Stereo 网络基本功能测试")

        # 设备选择
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 创建网络
        maxdisp = 192
        model = CGI_Stereo(maxdisp=maxdisp).to(device)
        print(f"✓ 网络创建成功，最大视差: {maxdisp}")

        # 测试输入尺寸
        batch_size = 1
        height, width = 384, 640  # 常用的输入尺寸
        left = torch.randn(batch_size, 3, height, width).to(device)
        right = torch.randn(batch_size, 3, height, width).to(device)

        print(f"输入尺寸: {left.shape}")

        # 设置训练模式
        model.train()
        start_time = time.time()
        with torch.no_grad():
            outputs_train = model(left, right)
        train_time = time.time() - start_time

        print(f"✓ 训练模式测试成功")
        print(f"  - 输出数量: {len(outputs_train)}")
        print(f"  - 高分辨率视差图: {outputs_train[0].shape}")
        print(f"  - 低分辨率视差图: {outputs_train[1].shape}")
        print(f"  - 推理时间: {train_time:.4f}s")

        # 设置推理模式
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            outputs_infer = model(left, right)
        infer_time = time.time() - start_time

        print(f"✓ 推理模式测试成功")
        print(f"  - 输出数量: {len(outputs_infer)}")
        print(f"  - 视差图: {outputs_infer[0].shape}")
        print(f"  - 推理时间: {infer_time:.4f}s")

        return model, left, right

    def test_different_input_sizes():
        """测试不同输入尺寸"""
        print_separator("2. 不同输入尺寸测试")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CGI_Stereo(maxdisp=192).to(device)
        model.eval()

        test_sizes = [
            (256, 512),   # 小尺寸
            (384, 640),   # 中等尺寸 (常用)
            (480, 640),   # KITTI 尺寸
            (512, 960),   # 大尺寸
        ]

        for h, w in test_sizes:
            try:
                left = torch.randn(1, 3, h, w).to(device)
                right = torch.randn(1, 3, h, w).to(device)

                with torch.no_grad():
                    start_time = time.time()
                    outputs = model(left, right)
                    end_time = time.time()

                print(f"✓ 尺寸 {h}x{w}: 输出 {outputs[0].shape}, 用时 {end_time-start_time:.4f}s")

            except Exception as e:
                print(f"✗ 尺寸 {h}x{w}: 失败 - {e}")

    def test_gradient_flow():
        """测试梯度流"""
        print_separator("3. 梯度流测试")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CGI_Stereo(maxdisp=192).to(device)
        model.train()

        # 创建需要梯度的输入（先在CPU创建然后移动到GPU，确保是叶子张量）
        left = torch.randn(1, 3, 384, 640, requires_grad=True)
        right = torch.randn(1, 3, 384, 640, requires_grad=True)
        left = left.to(device)
        right = right.to(device)

        # 前向传播
        outputs = model(left, right)

        # 创建虚拟损失
        loss = sum(torch.mean(output) for output in outputs)

        # 反向传播
        loss.backward()

        # 检查输入梯度
        left_grad_norm = left.grad.norm().item() if left.grad is not None else 0.0
        right_grad_norm = right.grad.norm().item() if right.grad is not None else 0.0

        # 检查模型参数梯度
        param_grads = []
        total_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_grads.append((name, grad_norm))
                total_grad_norm += grad_norm

        print(f"✓ 梯度流测试成功")
        print(f"  - 输入梯度范数: 左={left_grad_norm:.6f}, 右={right_grad_norm:.6f}")
        print(f"  - 参数梯度数量: {len(param_grads)}/{len(list(model.named_parameters()))}")
        print(f"  - 总梯度范数: {total_grad_norm:.6f}")

        # 显示前5个参数的梯度范数
        for name, grad_norm in param_grads[:5]:
            print(f"    {name}: {grad_norm:.6f}")

        # 检查梯度是否为0
        zero_grad_count = sum(1 for _, grad_norm in param_grads if grad_norm < 1e-8)
        if zero_grad_count > 0:
            print(f"  - ⚠️ 警告: {zero_grad_count} 个参数梯度接近0")
        else:
            print("  - ✓ 所有参数梯度正常")

    def test_memory_usage():
        """测试内存使用"""
        print_separator("4. 内存使用测试")

        if not torch.cuda.is_available():
            print("CUDA 不可用，跳过内存测试")
            return

        device = torch.device('cuda')

        # 清理GPU内存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        peak_initial = torch.cuda.max_memory_allocated() / 1024**3  # GB

        model = CGI_Stereo(maxdisp=192).to(device)
        model_memory = torch.cuda.memory_allocated() / 1024**3  # GB

        # 测试推理
        model.eval()
        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            left = torch.randn(batch_size, 3, 384, 640).to(device)
            right = torch.randn(batch_size, 3, 384, 640).to(device)

            input_memory = torch.cuda.memory_allocated() / 1024**3  # GB

            with torch.no_grad():
                outputs = model(left, right)

            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

            print(f"✓ Batch Size {batch_size}:")
            print(f"  - 模型参数: {model_memory:.3f} GB")
            print(f"  - 输入数据: {input_memory - model_memory:.3f} GB")
            print(f"  - 峰值内存: {peak_memory:.3f} GB")
            print(f"  - 总内存增长: {peak_memory - initial_memory:.3f} GB")

            del left, right, outputs

    def test_output_properties():
        """测试输出属性"""
        print_separator("5. 输出属性测试")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CGI_Stereo(maxdisp=192).to(device)
        model.eval()

        # 创建测试输入
        left = torch.randn(1, 3, 384, 640).to(device)
        right = torch.randn(1, 3, 384, 640).to(device)

        with torch.no_grad():
            outputs = model(left, right)
            pred_disp = outputs[0]

        print(f"输出统计信息:")
        print(f"  - 形状: {pred_disp.shape}")
        print(f"  - 数据类型: {pred_disp.dtype}")
        print(f"  - 设备: {pred_disp.device}")
        print(f"  - 最小值: {pred_disp.min().item():.3f}")
        print(f"  - 最大值: {pred_disp.max().item():.3f}")
        print(f"  - 平均值: {pred_disp.mean().item():.3f}")
        print(f"  - 标准差: {pred_disp.std().item():.3f}")

        # 检查视差范围是否合理
        max_possible_disp = 192  # 设置的最大视差
        actual_max = pred_disp.max().item()

        if 0 <= actual_max <= max_possible_disp * 1.1:  # 允许10%的误差
            print(f"✓ 视差范围合理: 0 <= {actual_max:.3f} <= {max_possible_disp}")
        else:
            print(f"⚠ 视差范围可能异常: {actual_max:.3f}")

        # 检查是否存在无效值
        if torch.isfinite(pred_disp).all():
            print("✓ 输出值都是有限数")
        else:
            print("✗ 输出中存在无限值或NaN")



    def run_performance_benchmark():
        """运行性能基准测试"""
        print_separator("7. 性能基准测试")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CGI_Stereo(maxdisp=192).to(device)
        model.eval()

        # 预热
        left = torch.randn(1, 3, 384, 640).to(device)
        right = torch.randn(1, 3, 384, 640).to(device)

        for _ in range(5):
            with torch.no_grad():
                _ = model(left, right)

        # 性能测试
        num_runs = 20
        times = []

        print(f"运行 {num_runs} 次推理测试...")

        with torch.no_grad():
            for i in range(num_runs):
                start_time = time.time()
                outputs = model(left, right)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)

                if (i + 1) % 5 == 0:
                    print(f"  完成 {i+1}/{num_runs}")

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time

        print(f"✓ 性能测试结果:")
        print(f"  - 平均推理时间: {avg_time:.4f} ± {std_time:.4f}s")
        print(f"  - 最快时间: {min_time:.4f}s")
        print(f"  - 最慢时间: {max_time:.4f}s")
        print(f"  - 理论FPS: {fps:.2f}")

        # 计算FLOPs (如果安装了fvcore)
        try:
            from fvcore.nn import FlopCountAnalysis
            model_flops = FlopCountAnalysis(model, (left, right))
            total_flops = model_flops.total()
            print(f"  - 总计算量: {total_flops/1e9:.2f} GFLOPs")
        except ImportError:
            print("  - 未安装 fvcore，跳过 FLOPs 计算")

    def main():
        """主测试函数"""
        print("🚀 CGI-Stereo 网络测试开始")
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")

        try:
            # 基本功能测试
            model, left, right = test_basic_functionality()

            # 不同输入尺寸测试
            test_different_input_sizes()

            # 梯度流测试
            test_gradient_flow()

            # 内存使用测试
            test_memory_usage()

            # 输出属性测试
            test_output_properties()

            # 性能基准测试
            run_performance_benchmark()

            print_separator("✅ 所有测试完成!")
            print("CGI-Stereo 网络运行正常，可以开始训练或推理。")

        except Exception as e:
            print(f"\n❌ 测试过程中出现错误:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            import traceback
            traceback.print_exc()

    # 运行所有测试
    main()