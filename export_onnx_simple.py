#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGI-Stereo 网络ONNX导出脚本 (简化版)
将训练好的CGI-Stereo模型导出为ONNX格式
"""

import torch
import numpy as np
import os
import warnings

# 导入CGI-Stereo网络
from models.CGI_Stereo import CGI_Stereo

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

def export_cgi_stereo_to_onnx(maxdisp=192, output_path="cgi_stereo.onnx",
                            input_shape=(1, 3, 480, 640)):
    """
    导出CGI-Stereo模型为ONNX格式

    Args:
        maxdisp: 最大视差值
        output_path: 输出ONNX文件路径
        input_shape: 输入张量形状 (batch, channel, height, width)
    """
    print("CGI-Stereo ONNX Export Tool")
    print("=" * 50)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模型
    print("Creating CGI-Stereo model...")
    model = CGI_Stereo(maxdisp=maxdisp).to(device)
    model.eval()

    # 创建示例输入
    batch_size, channels, height, width = input_shape
    left = torch.randn(batch_size, channels, height, width).to(device)
    right = torch.randn(batch_size, channels, height, width).to(device)

    print(f"Input shape: {left.shape}")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    try:
        # 导出模型
        print("Exporting to ONNX...")

        # 简化的导出，不使用动态轴
        torch.onnx.export(
            model,
            (left, right),
            output_path,
            export_params=True,
            opset_version=11,  # 使用较稳定的版本
            do_constant_folding=True,
            input_names=['left', 'right'],
            output_names=['disparity'],
            dynamic_axes=None,  # 暂时不使用动态轴
            verbose=False
        )

        print(f"ONNX export successful: {output_path}")

        # 验证模型大小
        file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
        print(f"Model file size: {file_size:.2f} MB")

        # 简单的输出验证
        print("Verifying output...")
        with torch.no_grad():
            pytorch_output = model(left, right)
            print(f"PyTorch output shape: {pytorch_output[0].shape}")
            print(f"Output range: [{pytorch_output[0].min().item():.2f}, {pytorch_output[0].max().item():.2f}]")

        return True

    except Exception as e:
        print(f"ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def export_multiple_sizes(maxdisp=192, output_dir="onnx_models"):
    """
    导出多种尺寸的ONNX模型

    Args:
        maxdisp: 最大视差值
        output_dir: 输出目录
    """
    print(f"Exporting multiple sizes to: {output_dir}")

    # 不同的输入尺寸
    sizes = [
        (256, 512),   # 小尺寸
        (384, 640),   # 中等尺寸
        (480, 640),   # KITTI尺寸
        (512, 960),   # 大尺寸
    ]

    success_count = 0

    for height, width in sizes:
        filename = f"cgi_stereo_{height}x{width}_disp{maxdisp}.onnx"
        output_path = os.path.join(output_dir, filename)
        input_shape = (1, 3, height, width)

        print(f"\nExporting {height}x{width}...")
        if export_cgi_stereo_to_onnx(maxdisp, output_path, input_shape):
            success_count += 1

        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nExport complete: {success_count}/{len(sizes)} models exported successfully")
    return success_count

def main():
    """主函数"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # 配置
    maxdisp = 192

    print("\nExport options:")
    print("1. Export default size (480x640)")
    print("2. Export multiple sizes")
    print("3. Custom size")

    try:
        choice = input("Select option (1-3): ").strip()
    except:
        choice = "1"  # 默认选择

    if choice == "1":
        # 默认导出
        output_path = f"cgi_stereo_480x640_disp{maxdisp}.onnx"
        success = export_cgi_stereo_to_onnx(maxdisp, output_path)

    elif choice == "2":
        # 批量导出
        success = export_multiple_sizes(maxdisp)

    elif choice == "3":
        # 自定义尺寸
        try:
            height = int(input("Enter image height: "))
            width = int(input("Enter image width: "))

            output_path = f"cgi_stereo_{height}x{width}_disp{maxdisp}.onnx"
            input_shape = (1, 3, height, width)

            success = export_cgi_stereo_to_onnx(maxdisp, output_path, input_shape)

        except ValueError:
            print("Invalid input, using default size")
            output_path = f"cgi_stereo_480x640_disp{maxdisp}.onnx"
            success = export_cgi_stereo_to_onnx(maxdisp, output_path)
    else:
        print("Invalid choice, using default")
        output_path = f"cgi_stereo_480x640_disp{maxdisp}.onnx"
        success = export_cgi_stereo_to_onnx(maxdisp, output_path)

    # 总结
    print("\n" + "=" * 50)
    if success:
        print("Export successful!")
        print("\nUsage instructions:")
        print("1. Use ONNX Runtime for inference")
        print("2. Input: left and right images (batch, 3, height, width)")
        print("3. Output: disparity map (batch, height, width)")
        print("4. Range: approximately [0, maxdisp]")
    else:
        print("Export failed!")
    print("=" * 50)

if __name__ == "__main__":
    main()