# Depth Anything 3 - Quick Start Guide

## 环境说明
- **环境名称**: `da3`
- **Python 版本**: 3.10.19
- **PyTorch 版本**: 2.6.0 + CUDA 12.4
- **项目路径**: `C:\Users\Lloyd\Depth-Anything-3`

---

## 1. 激活环境

每次使用前，先激活 Conda 环境：

```bash
conda activate da3
cd C:\Users\Lloyd\Depth-Anything-3
```

---

## 2. 快速运行示例

### 方式 A: 运行完整演示脚本（推荐）

```bash
python run_notebook_demo.py
```

**输出结果**:
- `demo_output/step1_original_images.png` - 原始图片预览
- `demo_output/step2_depth_comparison.png` - 深度对比图（推荐查看）
- `demo_output/depth_000.png`, `depth_010.png` - 单独的深度图

---

### 方式 B: 命令行工具（高级用法）

#### 处理单张图片
```bash
da3 image assets/examples/SOH/000.png --output-dir output_cli
```

#### 批量处理文件夹
```bash
da3 auto assets/examples/SOH --output-dir output_cli
```

#### 处理视频
```bash
da3 video assets/examples/robot_unitree.mp4 --fps 15 --export-dir output_video
```

---

### 方式 C: 使用 Jupyter Notebook

```bash
jupyter notebook demo_depth_anything3.ipynb
```

在浏览器中逐个运行 Cell，可以实时查看结果。

---

## 3. 处理自己的图片

### 方法 1: 使用 Python 脚本

创建一个新的 Python 文件 `my_demo.py`:

```python
import torch
from depth_anything_3.api import DepthAnything3
import matplotlib.pyplot as plt
from PIL import Image

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
model = model.to(device)

# 处理你的图片（修改路径）
image_path = "你的图片路径.jpg"
prediction = model.inference([image_path])

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(Image.open(image_path))
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(prediction.depth[0], cmap='inferno')
axes[1].set_title("Depth Map")
axes[1].axis('off')

plt.tight_layout()
plt.savefig('my_result.png')
plt.show()
```

运行：
```bash
python my_demo.py
```

---

### 方法 2: 使用命令行

```bash
# 单张图片
da3 image "你的图片路径.jpg" --output-dir my_output

# 整个文件夹
da3 auto "你的图片文件夹" --output-dir my_output
```

---

## 4. 切换模型

项目支持多个模型，效果和速度不同：

| 模型名称 | 参数量 | 速度 | 精度 | 显存需求 |
|---------|--------|------|------|---------|
| DA3-SMALL | 0.08B | ⚡⚡⚡ | ⭐⭐⭐ | 低 |
| DA3-BASE | 0.12B | ⚡⚡ | ⭐⭐⭐⭐ | 中 |
| DA3-LARGE | 0.35B | ⚡ | ⭐⭐⭐⭐⭐ | 高 |
| DA3-GIANT | 1.15B | 🐢 | ⭐⭐⭐⭐⭐⭐ | 很高 |
| DA3NESTED-GIANT-LARGE | 1.40B | 🐢 | ⭐⭐⭐⭐⭐⭐ | 很高 |

### 修改模型

在脚本中修改这一行：
```python
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")  # 改为你想要的模型
```

或在命令行中指定：
```bash
da3 auto assets/examples/SOH --model-dir depth-anything/DA3-LARGE --output-dir output
```

---

## 5. 常见问题

### Q1: 没有检测到 CUDA，只用 CPU？
**A**: 检查你的 CUDA 是否安装正确：
```bash
conda activate da3
python -c "import torch; print(torch.cuda.is_available())"
```
如果输出 `False`，说明 PyTorch 没有找到 GPU。检查：
1. NVIDIA 驱动是否安装
2. CUDA Toolkit 版本是否匹配

### Q2: 如何查看模型列表？
**A**: 访问 Hugging Face: https://huggingface.co/depth-anything

### Q3: 运行速度很慢？
**A**: 
- 确保使用 GPU (`torch.cuda.is_available()` 返回 `True`)
- 使用更小的模型 (DA3-SMALL)
- 降低输入图片分辨率

### Q4: 内存不足 (Out of Memory)
**A**:
- 使用更小的模型
- 减少批处理大小
- 降低输入分辨率

---

## 6. 项目文件说明

```
Depth-Anything-3/
├── run_notebook_demo.py      # 完整演示脚本（推荐使用）
├── run_demo.py                # 简单演示脚本
├── demo_depth_anything3.ipynb # Jupyter Notebook
├── demo_output/               # 输出结果文件夹
│   ├── step1_original_images.png
│   ├── step2_depth_comparison.png
│   ├── depth_000.png
│   └── depth_010.png
├── assets/examples/           # 示例数据
│   ├── SOH/                   # 示例图片
│   └── robot_unitree.mp4      # 示例视频
└── src/depth_anything_3/      # 源代码
```

---

## 7. 卸载环境

如果需要重新安装或删除环境：

```bash
conda deactivate
conda env remove -n da3
```

---

## 8. 更新项目

如果项目有更新：

```bash
conda activate da3
cd C:\Users\Lloyd\Depth-Anything-3
git pull
pip install -e . --no-deps  # 重新安装项目
```

---

## 联系方式

- 项目地址: https://github.com/ByteDance-Seed/Depth-Anything-3
- 问题反馈: https://github.com/ByteDance-Seed/Depth-Anything-3/issues

---

**Happy Coding! 🚀**


