# CGI-Stereo 模型导出指南

本文档介绍如何将 CGI-Stereo 网络导出为部署友好的格式。

## 📁 导出文件说明

### 已生成的文件

1. **TorchScript 模型**
   - `cgi_stereo_384x640_disp192_trace.pt` (17.2 MB) - 原始尺寸
   - `cgi_stereo_480x640_disp192_trace.pt` (预计 ~20 MB) - KITTI 尺寸
   - 支持动态输入尺寸
   - 生产就绪，无需 Python 依赖

2. **导出脚本**
   - `export_torchscript.py` - TorchScript 导出工具
   - `export_torchscript_480x640.py` - TorchScript 导出工具（480x640默认）
   - `export_onnx_simple.py` - ONNX 导出工具（需要 onnx 依赖）
   - `export_onnx.py` - 完整 ONNX 导出工具

3. **测试脚本**
   - `test_torchscript_model.py` - TorchScript 模型测试工具
   - `test_cgi_stereo.py` - 网络功能测试工具

## 🚀 快速开始

### 使用 TorchScript 模型

```python
import torch

# 加载模型
model = torch.jit.load('cgi_stereo_384x640_disp192_trace.pt')
model.eval()

# 准备输入 (任意尺寸，但建议接近 384x640)
left = torch.randn(1, 3, 384, 640)  # 左图像
right = torch.randn(1, 3, 384, 640)  # 右图像

# 推理
with torch.no_grad():
    disparity = model(left, right)[0]  # 输出: [1, 384, 640]

print(f"视差范围: [{disparity.min():.2f}, {disparity.max():.2f}]")
```

### 导出不同尺寸的模型

```bash
# 导出默认尺寸
python export_torchscript.py
# 选择 1

# 批量导出多种尺寸
python export_torchscript.py
# 选择 2

# 自定义尺寸导出
python export_torchscript.py
# 选择 3，然后输入高度和宽度
```

## 📊 模型信息

### CGI-Stereo 网络特性

- **输入**: 左右图像对 [B, 3, H, W]
- **输出**: 视差图 [B, H, W]
- **最大视差**: 192 (可配置)
- **基础架构**: MobileNetV2 + CGF + 沙漏网络
- **推理速度**: ~2.5 FPS (CPU, 384x640)

### 支持的输入尺寸

- ✅ 256×512 - 小尺寸，快速推理
- ✅ 480×640 - 中等尺寸 (默认，KITTI 数据集尺寸)
- ✅ 384×640 - 标准尺寸，平衡性能和精度
- ✅ 512×960 - 大尺寸，高精度
- ✅ 自定义尺寸 - 动态支持

## 🔧 部署建议

### 生产环境部署

1. **GPU 推理**
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = torch.jit.load('model.pt', map_location=device)
   model.to(device)
   model.eval()
   ```

2. **批处理优化**
   ```python
   # 批量处理多张图像
   batch_size = 4
   left_batch = torch.randn(batch_size, 3, 384, 640).to(device)
   right_batch = torch.randn(batch_size, 3, 384, 640).to(device)

   with torch.no_grad():
       disparities = model(left_batch, right_batch)
   ```

3. **内存管理**
   ```python
   # 定期清理缓存
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   ```

### 性能优化

- **量化**: 考虑使用半精度 (FP16) 推理
- **TensorRT**: NVIDIA GPU 可考虑 TensorRT 加速
- **多线程**: 使用 `torch.jit.freeze()` 优化 TorchScript 模型

## 🧪 测试和验证

### 运行完整测试

```bash
# 测试网络基本功能
python test_cgi_stereo.py

# 测试 TorchScript 模型
python test_torchscript_model.py
```

### 输出验证

- 视差值范围: [0, 192]
- 输出形状: [batch, height, width]
- 数据类型: float32
- 数值稳定性: 无 NaN 或 Inf

## 📋 文件清单

```
MY-CGI-Stereo/
├── models/
│   ├── CGI_Stereo.py              # 网络定义（已添加详细注释）
│   └── submodule.py               # 子模块定义
├── export_torchscript.py          # TorchScript 导出工具
├── export_onnx_simple.py          # ONNX 导出工具
├── export_onnx.py                 # 完整 ONNX 导出工具
├── test_torchscript_model.py      # TorchScript 模型测试
├── test_cgi_stereo.py             # 网络功能测试
├── cgi_stereo_384x640_disp192_trace.pt  # TorchScript 模型文件
└── README_EXPORT.md               # 本文档
```

## 🚨 注意事项

1. **兼容性**: TorchScript 模型与 PyTorch 2.x 兼容
2. **精度**: TorchScript trace 可能与原始模型有细微差异
3. **内存**: 大尺寸输入需要更多 GPU/CPU 内存
4. **依赖**: 部署时只需要 PyTorch，无需其他依赖

## 🆘 常见问题

### Q: TorchScript 和 ONNX 哪个更好？
A: TorchScript 在 PyTorch 生态中更稳定，ONNX 跨平台兼容性更好。根据目标平台选择。

### Q: 如何处理不同输入尺寸？
A: TorchScript 模型支持动态尺寸，但建议输入尺寸接近训练时的尺寸以获得最佳效果。

### Q: 推理速度太慢怎么办？
A: 考虑使用 GPU、减小输入尺寸、使用半精度推理或模型量化。

### Q: 如何加载预训练权重？
A: 修改 `export_torchscript.py` 中的 `model_path` 参数指向预训练模型文件。

## 📞 技术支持

如遇到问题，请检查：
1. PyTorch 版本兼容性
2. 输入数据格式和范围
3. GPU/CPU 内存是否充足
4. 模型文件是否完整

---

**最后更新**: 2025-11-06
**版本**: 1.0