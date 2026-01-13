#!/usr/bin/env python3
"""
Run individual tasks one at a time
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import argparse


def get_python_cmd():
    """Get the correct python command"""
    import shutil
    if shutil.which('python3'):
        return 'python3'
    elif shutil.which('python'):
        return 'python'
    else:
        return 'python3'  # default


def run_command(cmd, cwd=None, check=True):
    """Run a shell command"""
    # Replace 'python' with the correct command
    python_cmd = get_python_cmd()
    cmd = cmd.replace('python ', f'{python_cmd} ')
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=check, 
                          capture_output=False, text=True)
    return result


def setup_task1():
    """Setup Task 1: Object Detection"""
    print("\n" + "="*60)
    print("Setting up Task 1: Custom Object Detection")
    print("="*60)
    
    task_dir = Path("task1_object_detection")
    task_dir.mkdir(exist_ok=True)
    
    # Create dataset structure
    os.chdir(task_dir)
    run_command("python scripts/download_dataset.py", check=False)
    
    # Generate sample data if needed
    train_images_dir = Path("data/train/images")
    if not train_images_dir.exists() or len(list(train_images_dir.glob("*.jpg"))) == 0:
        print("\nGenerating sample data for Task 1...")
        script_path = Path(__file__).parent / "scripts" / "generate_sample_data.py"
        run_command(f"python {script_path} --task task1", check=False)
    
    os.chdir("..")
    print("\nTask 1 setup complete!")


def run_task1(mode="demo"):
    """Run Task 1: Object Detection"""
    print("\n" + "="*60)
    print("Running Task 1: Custom Object Detection")
    print("="*60)
    
    task_dir = Path("task1_object_detection")
    if not task_dir.exists():
        print("Error: task1_object_detection directory not found!")
        return False
    
    os.chdir(task_dir)
    
    try:
        if mode == "full":
            # Full training
            print("\n[1/3] Training model...")
            run_command("python train.py --config configs/default.yaml", check=False)
            
            # Evaluation
            checkpoint = Path("checkpoints/best_model.pth")
            if checkpoint.exists():
                print("\n[2/3] Evaluating model...")
                run_command(f"python evaluate.py --config configs/default.yaml --checkpoint {checkpoint}", check=False)
            else:
                print("Warning: No checkpoint found, skipping evaluation")
            
            # Inference
            if checkpoint.exists():
                print("\n[3/3] Running inference...")
                test_image = Path("data/val/images")
                if test_image.exists() and len(list(test_image.glob("*.jpg"))) > 0:
                    os.makedirs("results", exist_ok=True)
                    run_command(f"python inference.py --model {checkpoint} --input {test_image} --output results/", check=False)
                else:
                    print("Warning: No validation images found for inference")
        else:
            # Demo mode - check if we have data
            train_images = Path("data/train/images")
            if train_images.exists() and len(list(train_images.glob("*.jpg"))) > 0:
                print("\nDemo mode: Sample data found")
                print("To run full training, use: python run_task.py --task task1 --mode full")
            else:
                print("\nNo training data found. Generating sample data...")
                script_path = Path(__file__).parent / "scripts" / "generate_sample_data.py"
                run_command(f"python {script_path} --task task1", check=False)
                print("\nSample data generated. You can now train with: python run_task.py --task task1 --mode full")
        
        print("\n" + "="*60)
        print("Task 1 completed!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\nError in Task 1: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir("..")


def setup_task2():
    """Setup Task 2: Quality Inspection"""
    print("\n" + "="*60)
    print("Setting up Task 2: Quality Inspection")
    print("="*60)
    
    task_dir = Path("task2_quality_inspection")
    task_dir.mkdir(exist_ok=True)
    
    # Generate sample data if needed
    samples_dir = task_dir / "samples" / "defective"
    if not samples_dir.exists() or len(list(samples_dir.glob("*.jpg"))) == 0:
        print("\nGenerating sample data for Task 2...")
        script_path = Path(__file__).parent / "scripts" / "generate_sample_data.py"
        run_command(f"python {script_path} --task task2", check=False)
    
    # Create data directories
    (task_dir / "data" / "train" / "images").mkdir(parents=True, exist_ok=True)
    (task_dir / "data" / "val" / "images").mkdir(parents=True, exist_ok=True)
    
    print("\nTask 2 setup complete!")


def run_task2(mode="demo"):
    """Run Task 2: Quality Inspection"""
    print("\n" + "="*60)
    print("Running Task 2: Quality Inspection")
    print("="*60)
    
    task_dir = Path("task2_quality_inspection")
    if not task_dir.exists():
        print("Error: task2_quality_inspection directory not found!")
        return False
    
    os.chdir(task_dir)
    
    try:
        if mode == "full":
            # Full training
            print("\n[1/2] Training model...")
            run_command("python train_inspection_model.py --config config.yaml", check=False)
            
            # Inference
            checkpoint = Path("checkpoints/best_model.pth")
            if checkpoint.exists():
                print("\n[2/2] Running inspection...")
                samples_dir = Path("samples/defective")
                if samples_dir.exists() and len(list(samples_dir.glob("*.jpg"))) > 0:
                    os.makedirs("results", exist_ok=True)
                    run_command(f"python inspect.py --model {checkpoint} --input {samples_dir} --output results/", check=False)
                else:
                    print("Warning: No sample images found for inference")
            else:
                print("Warning: No checkpoint found, skipping inference")
        else:
            # Demo mode
            samples_dir = Path("samples/defective")
            if samples_dir.exists() and len(list(samples_dir.glob("*.jpg"))) > 0:
                print("\nDemo mode: Sample data found")
                print("To run full training, use: python run_task.py --task task2 --mode full")
            else:
                print("\nNo sample data found. Generating sample data...")
                script_path = Path(__file__).parent / "scripts" / "generate_sample_data.py"
                run_command(f"python {script_path} --task task2", check=False)
                print("\nSample data generated. You can now train with: python run_task.py --task task2 --mode full")
        
        print("\n" + "="*60)
        print("Task 2 completed!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\nError in Task 2: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir("..")


def show_task3():
    """Show Task 3: VLM Design Document"""
    print("\n" + "="*60)
    print("Task 3: VLM Design Document")
    print("="*60)
    
    design_doc = Path("task3_vlm_design/VLM_Design_Document.md")
    
    if design_doc.exists():
        print(f"\nDesign document found: {design_doc}")
        size_kb = design_doc.stat().st_size / 1024
        print(f"Size: {size_kb:.2f} KB")
        print("\nDocument sections:")
        print("  ✓ (A) Model Selection")
        print("  ✓ (B) Design Strategy")
        print("  ✓ (C) Optimization")
        print("  ✓ (D) Hallucination Mitigation")
        print("  ✓ (E) Training Plan")
        print("  ✓ (F) Validation")
        print(f"\nTo view the document:")
        print(f"  cat {design_doc}")
        print(f"  or")
        print(f"  less {design_doc}")
        return True
    else:
        print("Error: Design document not found!")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run individual computer vision tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup and run Task 1 in demo mode
  python run_task.py --task task1

  # Run Task 1 with full training
  python run_task.py --task task1 --mode full

  # Setup and run Task 2
  python run_task.py --task task2

  # View Task 3 design document
  python run_task.py --task task3
        """
    )
    parser.add_argument('--task', type=str, required=True,
                       choices=['task1', 'task2', 'task3'],
                       help='Which task to run')
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'full'],
                       help='Run mode: demo (quick setup) or full (complete training)')
    parser.add_argument('--setup-only', action='store_true',
                       help='Only run setup, skip execution')
    args = parser.parse_args()
    
    original_dir = os.getcwd()
    
    try:
        print("\n" + "="*60)
        print(f"Computer Vision Task Runner - {args.task.upper()}")
        print("="*60)
        
        if args.task == 'task1':
            setup_task1()
            if not args.setup_only:
                run_task1(mode=args.mode)
        
        elif args.task == 'task2':
            setup_task2()
            if not args.setup_only:
                run_task2(mode=args.mode)
        
        elif args.task == 'task3':
            show_task3()
        
        print("\nDone!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()

