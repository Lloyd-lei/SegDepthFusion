"""
Deployment Script for Seg-Depth Pipeline

This script helps organize files into the correct directory structure.

Usage:
    python deploy.py
"""

import shutil
from pathlib import Path


def create_structure():
    """Create directory structure and move files."""
    
    print("Creating Seg-Depth Pipeline Directory Structure...")
    print("=" * 60)
    
    # Define structure
    structure = {
        "src": {
            "models": ["sam3_model.py", "da3_model.py"],
            "capture": ["screen_capture.py"],
            "pipeline": ["seg_depth_pipeline.py"],
            "visualization": ["visualizer.py"]
        }
    }
    
    # Create directories
    base = Path(".")
    
    for dir_name, subdirs in structure.items():
        dir_path = base / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created: {dir_path}")
        
        # Create __init__.py
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Package initialization."""\n')
        
        if isinstance(subdirs, dict):
            for subdir_name, files in subdirs.items():
                subdir_path = dir_path / subdir_name
                subdir_path.mkdir(exist_ok=True)
                print(f"  ✓ Created: {subdir_path}")
                
                # Create __init__.py
                init_file = subdir_path / "__init__.py"
                if not init_file.exists():
                    init_file.write_text('"""Package initialization."""\n')
                
                # Note files to move (don't actually move, just inform user)
                for file_name in files:
                    print(f"    → Move {file_name} to {subdir_path}/")
        elif isinstance(subdirs, list):
            for file_name in subdirs:
                print(f"  → Move {file_name} to {dir_path}/")
    
    print("\n" + "=" * 60)
    print("Directory structure created!")
    print("\nManual steps required:")
    print("1. Copy the Python files to their respective directories as shown above")
    print("2. Keep main.py, config.yaml, and requirements.txt in the root")
    print("3. Ensure sam3/ and da3/ directories are at the root level")
    print("\nAlternatively, use the FLAT structure (simpler):")
    print("  Just keep all .py files in the root directory alongside sam3/ and da3/")
    print("=" * 60)


def create_flat_structure():
    """
    Alternative: Create a flat structure (recommended for simplicity).
    All files in root directory.
    """
    print("\n" + "=" * 60)
    print("RECOMMENDED: Flat Structure")
    print("=" * 60)
    print("\nFor simplicity, you can keep all files in the root:")
    print("""
seg_depth/
├── sam3/                  # SAM3 repository
├── da3/                   # Depth-Anything-3 repository
├── sam3_model.py         # All modules in root
├── da3_model.py
├── screen_capture.py
├── seg_depth_pipeline.py
├── visualizer.py
├── main.py
├── test_pipeline.py
├── config.yaml
└── requirements.txt
    """)
    print("\nThis structure works perfectly and requires NO file moving!")
    print("=" * 60)


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║         Seg-Depth Pipeline Deployment Helper           ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    print("Choose deployment structure:\n")
    print("1. Flat Structure (RECOMMENDED - simpler)")
    print("2. Organized Structure (better for large projects)")
    print("3. Just show me what to do")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        create_flat_structure()
        print("\n✓ You're all set! Just make sure files are in the root directory.")
    elif choice == "2":
        create_structure()
    else:
        print("\nQuick Setup Instructions:")
        print("=" * 60)
        print("\n1. Navigate to your seg_depth directory:")
        print("   cd /path/to/seg_depth")
        print("\n2. Ensure you have these subdirectories:")
        print("   - sam3/    (SAM3 repository)")
        print("   - da3/     (Depth-Anything-3 repository)")
        print("\n3. Place all provided .py files in the root directory")
        print("\n4. Install dependencies:")
        print("   conda activate seg_depth")
        print("   pip install -r requirements.txt")
        print("\n5. Run the pipeline:")
        print("   python main.py --config config.yaml")
        print("=" * 60)


if __name__ == "__main__":
    main()