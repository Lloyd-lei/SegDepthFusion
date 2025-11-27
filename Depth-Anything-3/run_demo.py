import glob
import os
import torch
import cv2
import numpy as np
from depth_anything_3.api import DepthAnything3

def run_demo():
    # 1. 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("[INFO] Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA not detected, using CPU (slow)")

    # 2. 加载模型
    # 为了快速演示，这里使用 DA3-SMALL 模型。
    # 如果需要更高精度，可以改为 "depth-anything/DA3-LARGE" 或 "depth-anything/DA3NESTED-GIANT-LARGE"
    model_name = "depth-anything/DA3-SMALL" 
    print(f"[INFO] Loading model: {model_name} ...")
    
    try:
        model = DepthAnything3.from_pretrained(model_name)
        model = model.to(device=device)
        print("[INFO] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return

    # 3. 准备输入图片
    example_path = os.path.join("assets", "examples", "SOH")
    if not os.path.exists(example_path):
        print(f"[ERROR] Example path not found: {example_path}")
        return
    
    image_paths = sorted(glob.glob(os.path.join(example_path, "*.png")))
    if not image_paths:
        print("[ERROR] No images found")
        return
    
    print(f"[INFO] Found {len(image_paths)} images: {[os.path.basename(p) for p in image_paths]}")

    # 4. 运行推理
    print("[INFO] Starting inference...")
    try:
        # inference 方法接受图片路径列表
        prediction = model.inference(image_paths)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return

    # 5. 保存结果
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Saving results to: {output_dir}")

    # prediction.depth 是 [N, H, W] 的 numpy 数组
    depths = prediction.depth

    for i, (depth, img_path) in enumerate(zip(depths, image_paths)):
        filename = os.path.basename(img_path)
        print(f"   Processing: {filename}")
        
        # 归一化深度图以便可视化 (0-255)
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 1e-6:
            depth_norm = (depth - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth_norm = np.zeros_like(depth)
            
        depth_norm = depth_norm.astype(np.uint8)
        
        # 应用伪彩色 (Inferno colormap)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        
        save_path = os.path.join(output_dir, f"depth_{filename}")
        cv2.imwrite(save_path, depth_color)
        print(f"   Saved: {save_path}")

    print("\n[INFO] Demo completed!")

if __name__ == "__main__":
    run_demo()

