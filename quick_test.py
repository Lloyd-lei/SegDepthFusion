"""快速测试 - 验证模型加载和基本功能"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("快速测试开始...")
print("=" * 60)

# 测试 1: PyTorch
print("\n[1/4] 测试 PyTorch...")
import torch
print(f"✓ PyTorch版本: {torch.__version__}")
print(f"✓ CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# 测试 2: SAM3
print("\n[2/4] 测试 SAM3...")
try:
    from sam3_model import SAM3Model
    sam3 = SAM3Model(device="cuda", confidence_threshold=0.5)
    print("✓ SAM3 加载成功")
except Exception as e:
    print(f"✗ SAM3 加载失败: {e}")
    sys.exit(1)

# 测试 3: DA3
print("\n[3/4] 测试 DA3...")
try:
    from da3_model import DA3Model
    da3 = DA3Model(
        model_name="depth-anything/DA3NESTED-GIANT-LARGE",
        device="cuda"
    )
    print("✓ DA3 加载成功")
except Exception as e:
    print(f"✗ DA3 加载失败: {e}")
    sam3.cleanup()
    sys.exit(1)

# 测试 4: 处理单张图片
print("\n[4/4] 测试图片处理...")
try:
    from seg_depth_pipeline import SegDepthPipeline
    
    # 加载测试图片
    test_image = "orange_photos/Image_2025-11-24_232045_923.jpg"
    image = Image.open(test_image).convert("RGB")
    image_np = np.array(image)
    print(f"✓ 图片加载: {image_np.shape}")
    
    # 初始化pipeline
    pipeline = SegDepthPipeline(
        sam3_model=sam3,
        da3_model=da3,
        target_prompt="orange",
        prompts=["orange"]
    )
    
    # 处理图片
    print("  处理中...")
    result = pipeline.process_frame(image_np)
    print(f"✓ 处理完成: {result['status']}")
    
    if result['status'] == 'success':
        print(f"  - 橘子面积: {result['mask_info']['area_ratio']:.4f}")
        print(f"  - 平均深度: {result['depth_info']['mean_depth']:.4f}")
        print(f"  - 奖励值: {result['reward']:.4f}")
    
except Exception as e:
    print(f"✗ 处理失败: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\n清理资源...")
    sam3.cleanup()
    da3.cleanup()

print("\n=" * 60)
print("✓ 所有测试完成！")
print("=" * 60)

