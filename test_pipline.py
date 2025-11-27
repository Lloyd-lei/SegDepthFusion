"""
Quick Test Script for Seg-Depth Pipeline

This script tests the pipeline with orange images
without requiring screen capture.

Usage:
    python test_pipline.py --image path/to/orange.jpg
    python test_pipline.py --folder path/to/orange_photos
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from sam3_model import SAM3Model
from da3_model import DA3Model
from seg_depth_pipeline import SegDepthPipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_single_image(image_path: str, pipeline, output_dir: Path):
    """Test pipeline with a single image."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Process image
    result = pipeline.process_frame(image_np)
    
    # Create output filename
    image_name = Path(image_path).stem
    output_path = output_dir / f"{image_name}_result.png"
    
    # Print results
    if result['status'] == 'success':
        mask_info = result['mask_info']
        depth_info = result['depth_info']
        
        print(f"\n[OK] {image_name}:")
        print(f"  Area: {mask_info['area_ratio']:.4f} | Depth: {depth_info['mean_depth']:.4f} | Reward: {result['reward']:.4f}")
        
        # Visualize
        visualize_result(image_np, result, str(output_path))
    else:
        print(f"\n[FAIL] {image_name}: {result.get('error', 'Failed')}")
    
    return result


def test_pipeline(image_path: str = None, folder_path: str = None):
    """Test pipeline with single image or folder of images."""
    print("=" * 70)
    print("Seg-Depth Pipeline Test - Orange Detection")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Get list of images
    if folder_path:
        folder = Path(folder_path)
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
        image_files.sort()
        print(f"\nFound {len(image_files)} images in {folder}")
    elif image_path:
        image_files = [Path(image_path)]
        print(f"\nProcessing single image: {image_path}")
    else:
        raise ValueError("Must provide either --image or --folder")
    
    if len(image_files) == 0:
        print("No images found!")
        return
    
    # Initialize models
    print("\n" + "=" * 70)
    print("Initializing Models...")
    print("=" * 70)
    
    print("Loading SAM3 model...")
    sam3 = SAM3Model(device="cuda", confidence_threshold=0.5)
    
    print("Loading DA3 model...")
    da3 = DA3Model(
        model_name="depth-anything/DA3NESTED-GIANT-LARGE",
        device="cuda"
    )
    
    # Initialize pipeline
    print("Setting up pipeline...")
    pipeline = SegDepthPipeline(
        sam3_model=sam3,
        da3_model=da3,
        target_prompt="orange",
        prompts=["orange", "fruit", "citrus"]
    )
    
    # Process images
    print("\n" + "=" * 70)
    print(f"Processing {len(image_files)} images...")
    print("=" * 70)
    
    results = []
    for image_file in tqdm(image_files, desc="Processing"):
        try:
            result = test_single_image(str(image_file), pipeline, output_dir)
            results.append((image_file.name, result))
        except Exception as e:
            print(f"\n[ERROR] Error processing {image_file.name}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    successful = sum(1 for _, r in results if r['status'] == 'success')
    print(f"Processed: {len(results)} images")
    print(f"Successful: {successful} images")
    print(f"Failed: {len(results) - successful} images")
    
    if successful > 0:
        avg_reward = np.mean([r['reward'] for _, r in results if r['status'] == 'success'])
        print(f"Average Reward: {avg_reward:.4f}")
    
    # Cleanup
    print("\nCleaning up...")
    sam3.cleanup()
    da3.cleanup()
    
    print("\n[DONE] Test complete! Results saved to:", output_dir.absolute())


def visualize_result(image: np.ndarray, result: dict, output_path: str = 'test_result.png'):
    """Create matplotlib visualization - 只显示橘子区域的深度."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 确保 mask 是 2D 的
    mask = result.get('mask')
    if mask is not None and mask.ndim > 2:
        mask = mask.squeeze()
    
    # 左上: 原始图片
    ax1 = axes[0, 0]
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 右上: 分割结果 (绿色覆盖)
    ax2 = axes[0, 1]
    ax2.imshow(image)
    if mask is not None and mask.sum() > 0:
        # Create colored overlay
        overlay = np.zeros_like(image)
        overlay[mask > 0] = [0, 255, 0]
        ax2.imshow(overlay, alpha=0.4)
        
        # Draw bbox
        if result['mask_info']:
            bbox = result['mask_info']['bbox']
            x1, y1, x2, y2 = bbox
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            ax2.add_patch(rect)
    ax2.set_title("Segmentation (Orange)", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 左下: 完整深度图
    ax3 = axes[1, 0]
    if result['depth'] is not None:
        depth = result['depth']['depth']
        im = ax3.imshow(depth, cmap='turbo')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title(f"Full Depth Map", fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 右下: 只显示橘子区域的深度 (其他区域为黑色/透明)
    ax4 = axes[1, 1]
    if result['depth'] is not None and mask is not None and mask.sum() > 0:
        depth = result['depth']['depth']
        
        # Resize mask to match depth dimensions
        import cv2
        if mask.shape != depth.shape:
            mask_resized = cv2.resize(mask.astype(np.uint8), 
                                     (depth.shape[1], depth.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask_resized = mask
        
        # 创建只有橘子部分的深度图
        masked_depth = np.zeros_like(depth)
        masked_depth[mask_resized > 0] = depth[mask_resized > 0]
        
        # 非橘子区域设置为 NaN（显示为黑色）
        masked_depth_display = masked_depth.copy().astype(float)
        masked_depth_display[mask_resized == 0] = np.nan
        
        im = ax4.imshow(masked_depth_display, cmap='turbo', vmin=depth.min(), vmax=depth.max())
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        title = f"Orange Depth Only\n"
        if result['depth_info']:
            title += f"Mean: {result['depth_info']['mean_depth']:.3f} | Reward: {result['reward']:.3f}"
        ax4.set_title(title, fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test Seg-Depth Pipeline with Orange Images")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single test image (e.g., orange.jpg)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="orange_photos",
        help="Path to folder with orange images (default: orange_photos)"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        test_pipeline(image_path=args.image, folder_path=args.folder)
    except Exception as e:
        logging.error(f"Test failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())