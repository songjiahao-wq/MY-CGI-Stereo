#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGI-Stereo 网络ONNX导出脚本
将训练好的CGI-Stereo模型导出为ONNX格式，便于部署和推理优化
"""

import torch
import numpy as np
import os
import warnings
from pathlib import Path

# 导入CGI-Stereo网络
from models.CGI_Stereo import CGI_Stereo

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

class CGI_Stereo_Exporter:
    """CGI-Stereo ONNX导出器"""

    def __init__(self, model_path=None, maxdisp=192):
        """
        初始化导出器

        Args:
            model_path: 预训练模型路径，如果为None则创建未训练的模型
            maxdisp: 最大视差值
        """
        self.maxdisp = maxdisp
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"使用设备: {self.device}")

        # 创建模型
        self.model = CGI_Stereo(maxdisp=maxdisp).to(self.device)

        # 加载预训练权重
        if model_path and os.path.exists(model_path):
            print(f"加载预训练权重: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            # 处理不同的权重格式
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            print("✓ 权重加载成功")
        else:
            print("使用未训练的模型（随机权重）")

        # 设置为评估模式
        self.model.eval()

    def create_sample_input(self, batch_size=1, height=480, width=640):
        """
        创建示例输入

        Args:
            batch_size: 批次大小
            height: 图像高度
            width: 图像宽度

        Returns:
            tuple: (left_image, right_image)
        """
        left = torch.randn(batch_size, 3, height, width).to(self.device)
        right = torch.randn(batch_size, 3, height, width).to(self.device)
        return left, right

    def export_to_onnx(self, output_path, input_shape=(1, 3, 480, 640),
                      dynamic_axes=None, opset_version=16):
        """
        导出模型为ONNX格式

        Args:
            output_path: 输出ONNX文件路径
            input_shape: 输入张量形状 (batch, channel, height, width)
            dynamic_axes: 动态轴配置，用于支持可变输入尺寸
            opset_version: ONNX操作集版本
        """
        print(f"\n{'='*60}")
        print(f" 导出ONNX模型: {output_path}")
        print(f"{'='*60}")

        # 创建示例输入
        batch_size, channels, height, width = input_shape
        left, right = self.create_sample_input(batch_size, height, width)

        print(f"输入形状: {left.shape}")

        # 配置动态轴
        if dynamic_axes is None:
            # 默认支持动态批次、高度和宽度
            dynamic_axes = {
                'left': {0: 'batch_size', 2: 'height', 3: 'width'},
                'right': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 1: 'height', 2: 'width'}
            }

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            # 导出模型
            with torch.no_grad():
                torch.onnx.export(
                    self.model,
                    (left, right),
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['left', 'right'],
                    output_names=['disparity'],
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )

            print(f"ONNX导出成功: {output_path}")

            # 验证导出的ONNX模型
            if self.verify_onnx_model(output_path, left, right):
                print("ONNX模型验证通过")
            else:
                print("ONNX模型验证失败")

            return True

        except Exception as e:
            print(f"ONNX导出失败: {e}")
            return False

    def verify_onnx_model(self, onnx_path, left, right):
        """
        验证ONNX模型的正确性

        Args:
            onnx_path: ONNX文件路径
            left: 左图像输入
            right: 右图像输入

        Returns:
            bool: 验证是否通过
        """
        try:
            import onnx
            import onnxruntime as ort

            # 检查ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX模型格式检查通过")

            # 创建推理会话
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            ort_session = ort.InferenceSession(onnx_path, providers=providers)

            # 获取输入输出信息
            input_info = ort_session.get_inputs()
            output_info = ort_session.get_outputs()

            print(f"输入信息: {[info.name + ': ' + str(info.shape) for info in input_info]}")
            print(f"输出信息: {[info.name + ': ' + str(info.shape) for info in output_info]}")

            # PyTorch推理
            with torch.no_grad():
                pytorch_output = self.model(left, right)
                pytorch_result = pytorch_output[0].cpu().numpy()

            # ONNX推理
            ort_inputs = {
                'left': left.cpu().numpy(),
                'right': right.cpu().numpy()
            }
            onnx_output = ort_session.run(None, ort_inputs)
            onnx_result = onnx_output[0]

            # 比较结果
            max_diff = np.max(np.abs(pytorch_result - onnx_result))
            mean_diff = np.mean(np.abs(pytorch_result - onnx_result))

            print(f"最大差异: {max_diff:.6f}")
            print(f"平均差异: {mean_diff:.6f}")

            # 如果差异很小，认为验证通过
            tolerance = 1e-4
            if max_diff < tolerance:
                print("✓ PyTorch和ONNX输出一致")
                return True
            else:
                print(f"✗ 输出差异过大 (阈值: {tolerance})")
                return False

        except ImportError:
            print("⚠ 未安装onnx或onnxruntime，跳过验证")
            return True
        except Exception as e:
            print(f"✗ ONNX验证失败: {e}")
            return False

    def export_multiple_formats(self, output_dir="onnx_models", shapes=None):
        """
        导出多种尺寸和格式的ONNX模型

        Args:
            output_dir: 输出目录
            shapes: 输入形状列表，如果为None则使用默认配置
        """
        if shapes is None:
            shapes = [
                (1, 3, 256, 512),   # 小尺寸
                (1, 3, 480, 640),   # 中等尺寸 (默认，KITTI尺寸)
                (1, 3, 384, 640),   # 标准尺寸
                (1, 3, 512, 960),   # 大尺寸
            ]

        print(f"\n{'='*60}")
        print(f" 批量导出ONNX模型到: {output_dir}")
        print(f"{'='*60}")

        success_count = 0

        for i, shape in enumerate(shapes):
            batch_size, channels, height, width = shape
            filename = f"cgi_stereo_{height}x{width}_disp{self.maxdisp}.onnx"
            output_path = os.path.join(output_dir, filename)

            print(f"\n[{i+1}/{len(shapes)}] 导出 {height}x{width}")

            if self.export_to_onnx(output_path, shape):
                success_count += 1

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n批量导出完成: {success_count}/{len(shapes)} 个模型导出成功")
        return success_count == len(shapes)

def main():
    """主函数"""
    print("CGI-Stereo ONNX导出工具")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    # 配置参数
    maxdisp = 192
    model_path = None  # 如果有预训练模型，可以设置路径

    # 创建导出器
    exporter = CGI_Stereo_Exporter(model_path=model_path, maxdisp=maxdisp)

    # 导出选项
    print("\n请选择导出模式:")
    print("1. 导出默认尺寸 (480x640)")
    print("2. 批量导出多种尺寸")
    print("3. 自定义尺寸导出")

    try:
        choice = input("请输入选择 (1-3): ").strip()
    except:
        choice = "1"  # 默认选择

    if choice == "1":
        # 默认导出
        output_path = f"onnx_models/cgi_stereo_480x640_disp{maxdisp}.onnx"
        success = exporter.export_to_onnx(output_path)

    elif choice == "2":
        # 批量导出
        success = exporter.export_multiple_formats()

    elif choice == "3":
        # 自定义导出
        try:
            height = int(input("请输入图像高度: "))
            width = int(input("请输入图像宽度: "))
            batch_size = int(input("请输入批次大小 (默认1): ") or "1")

            output_path = f"onnx_models/cgi_stereo_{height}x{width}_disp{maxdisp}.onnx"
            input_shape = (batch_size, 3, height, width)

            success = exporter.export_to_onnx(output_path, input_shape)

        except ValueError:
            print("输入无效，使用默认配置")
            output_path = f"onnx_models/cgi_stereo_480x640_disp{maxdisp}.onnx"
            success = exporter.export_to_onnx(output_path)
    else:
        print("无效选择，使用默认配置")
        output_path = f"onnx_models/cgi_stereo_480x640_disp{maxdisp}.onnx"
        success = exporter.export_to_onnx(output_path)

    # 总结
    print(f"\n{'='*60}")
    if success:
        print("✅ ONNX导出成功!")
        print("\n使用说明:")
        print("1. ONNX模型可用于各种推理引擎 (ONNX Runtime, TensorRT等)")
        print("2. 推理时需要提供左右图像对作为输入")
        print("3. 输出为单通道视差图，值域约为 [0, maxdisp]")
        print("4. 建议使用GPU进行推理以获得更好性能")
    else:
        print("❌ ONNX导出失败!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()