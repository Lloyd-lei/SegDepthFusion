"""
Depth Anything 3 - 深度估计演示脚本
这个脚本包含了 Notebook 中的所有代码
"""

import glob
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from depth_anything_3.api import DepthAnything3

print("="*60)
print("Depth Anything 3 - Demo")
print("="*60)

# 1. Initialize
print("\n[1/6] Initializing...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 2. Load model
print("\n[2/6] Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model_name = "depth-anything/DA3-SMALL"
print(f"Loading: {model_name} ...")

model = DepthAnything3.from_pretrained(model_name)
model = model.to(device=device)
print("Model loaded successfully!")

# 3. Prepare input images
print("\n[3/6] Preparing input images...")
example_path = "assets/examples/SOH"

# Support multiple image formats
image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(example_path, ext)))
image_paths = sorted(image_paths)

print(f"Found {len(image_paths)} images:")
for p in image_paths:
    print(f"  - {os.path.basename(p)}")

# Display original images
print("\nGenerating original images preview...")
fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
if len(image_paths) == 1:
    axes = [axes]

for i, img_path in enumerate(image_paths):
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(f"Original: {os.path.basename(img_path)}", fontsize=12)
    axes[i].axis('off')

plt.tight_layout()
os.makedirs('demo_output', exist_ok=True)
plt.savefig('demo_output/step1_original_images.png', dpi=150, bbox_inches='tight')
print("Saved: demo_output/step1_original_images.png")
plt.close()

# 4. Run inference
print("\n[4/6] Running depth estimation inference...")
prediction = model.inference(image_paths)

print("Inference completed!")
print(f"  - Depth shape: {prediction.depth.shape}")
print(f"  - Confidence shape: {prediction.conf.shape}")

# 5. Visualize depth maps
print("\n[5/6] Visualizing depth maps...")
num_images = len(image_paths)
fig, axes = plt.subplots(2, num_images, figsize=(15, 10))
if num_images == 1:
    axes = axes.reshape(2, 1)

for i, (img_path, depth) in enumerate(zip(image_paths, prediction.depth)):
    # 显示原始图片
    img = Image.open(img_path)
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Original: {os.path.basename(img_path)}", fontsize=12, fontweight='bold')
    axes[0, i].axis('off')
    
    # 显示深度图
    im = axes[1, i].imshow(depth, cmap='inferno')
    axes[1, i].set_title(f"Depth Map", fontsize=12, fontweight='bold')
    axes[1, i].axis('off')
    plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('demo_output/step2_depth_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: demo_output/step2_depth_comparison.png")
plt.close()

# 6. Save results
print("\n[6/6] Saving results to files...")
output_dir = "demo_output"
os.makedirs(output_dir, exist_ok=True)

for i, (img_path, depth) in enumerate(zip(image_paths, prediction.depth)):
    filename = os.path.basename(img_path)
    
    # Normalize depth to 0-255
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 1e-6:
        depth_norm = (depth - depth_min) / (depth_max - depth_min) * 255.0
    else:
        depth_norm = np.zeros_like(depth)
    
    depth_norm = depth_norm.astype(np.uint8)
    
    # Apply colormap
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    
    # Save file
    save_path = os.path.join(output_dir, f"depth_{filename}")
    cv2.imwrite(save_path, cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR))
    print(f"  - Saved: {save_path}")

print(f"\n{'='*60}")
print("All results saved to: demo_output/")
print("  - step1_original_images.png (Original images)")
print("  - step2_depth_comparison.png (Depth comparison)")
print("  - depth_000.png, depth_010.png (Individual depth maps)")
print(f"{'='*60}")
print("\nDemo completed! Check the demo_output folder for results.")

