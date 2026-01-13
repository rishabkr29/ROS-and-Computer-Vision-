# GitHub Upload Guide - Individual Tasks

This guide helps you upload each task separately to GitHub.

## Option 1: Separate Repositories (Recommended)

Create separate GitHub repositories for each task.

### Task 1: Object Detection Repository

```bash
# Create a new directory for Task 1
cd /home/ros_master
mkdir computer_vision_task1
cd computer_vision_task1

# Copy Task 1 files
cp -r /home/ros_master/computer_vision/task1_object_detection/* .
cp /home/ros_master/computer_vision/requirements.txt .
cp /home/ros_master/computer_vision/README.md task1_README.md

# Initialize git
git init
git add .
git commit -m "Initial commit: Task 1 - Custom Object Detection"

# Create GitHub repository (do this on GitHub website first)
# Then add remote and push:
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-task1.git
git branch -M main
git push -u origin main
```

### Task 2: Quality Inspection Repository

```bash
# Create a new directory for Task 2
cd /home/ros_master
mkdir computer_vision_task2
cd computer_vision_task2

# Copy Task 2 files
cp -r /home/ros_master/computer_vision/task2_quality_inspection/* .
cp /home/ros_master/computer_vision/requirements.txt .
cp /home/ros_master/computer_vision/README.md task2_README.md

# Initialize git
git init
git add .
git commit -m "Initial commit: Task 2 - Quality Inspection System"

# Create GitHub repository (do this on GitHub website first)
# Then add remote and push:
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-task2.git
git branch -M main
git push -u origin main
```

### Task 3: VLM Design Repository

```bash
# Create a new directory for Task 3
cd /home/ros_master
mkdir computer_vision_task3
cd computer_vision_task3

# Copy Task 3 files
cp -r /home/ros_master/computer_vision/task3_vlm_design/* .
cp /home/ros_master/computer_vision/README.md task3_README.md

# Initialize git
git init
git add .
git commit -m "Initial commit: Task 3 - VLM Design Document"

# Create GitHub repository (do this on GitHub website first)
# Then add remote and push:
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-task3.git
git branch -M main
git push -u origin main
```

## Option 2: Single Repository with Separate Branches

Keep everything in one repository but use branches.

```bash
cd /home/ros_master/computer_vision

# Initialize git
git init
git add .
git commit -m "Initial commit: All tasks"

# Create branch for Task 1
git checkout -b task1-object-detection
git add task1_object_detection/ requirements.txt
git commit -m "Task 1: Custom Object Detection"
git push -u origin task1-object-detection

# Create branch for Task 2
git checkout main
git checkout -b task2-quality-inspection
git add task2_quality_inspection/ requirements.txt
git commit -m "Task 2: Quality Inspection System"
git push -u origin task2-quality-inspection

# Create branch for Task 3
git checkout main
git checkout -b task3-vlm-design
git add task3_vlm_design/
git commit -m "Task 3: VLM Design Document"
git push -u origin task3-vlm-design
```

## Option 3: Single Repository with Folders (Simplest)

Upload everything to one repository with clear folder structure.

```bash
cd /home/ros_master/computer_vision

# Initialize git
git init

# Create .gitignore (if not exists)
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Model checkpoints (optional - remove if you want to include)
checkpoints/*.pth
*.pth
*.pt

# Data (optional - remove if you want to include sample data)
data/
*.jpg
*.png
*.jpeg
*.bmp
*.json

# Results
results/
*.txt
*.json

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log
EOF

# Add all files
git add .
git commit -m "Initial commit: Computer Vision Tasks 1, 2, and 3"

# Create GitHub repository (do this on GitHub website first)
# Then add remote and push:
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-assignment.git
git branch -M main
git push -u origin main
```

## Automated Script

I've created a script to help you prepare each task for upload. Run:

```bash
cd /home/ros_master/computer_vision
python3 scripts/prepare_github_upload.py --task task1
python3 scripts/prepare_github_upload.py --task task2
python3 scripts/prepare_github_upload.py --task task3
```

## What to Include/Exclude

### Include:
- ✅ All Python source code (.py files)
- ✅ Configuration files (.yaml, .json configs)
- ✅ README files
- ✅ Requirements.txt
- ✅ Sample data (small files) - optional
- ✅ Sample outputs - optional
- ✅ Documentation

### Exclude (add to .gitignore):
- ❌ Large model checkpoints (.pth files > 100MB)
- ❌ Large datasets (images, videos)
- ❌ Python cache (__pycache__)
- ❌ Virtual environments
- ❌ IDE files
- ❌ Log files

## GitHub Repository Setup

### For Each Task Repository:

1. **Create Repository on GitHub:**
   - Go to GitHub.com
   - Click "New Repository"
   - Name: `computer-vision-task1` (or task2, task3)
   - Description: "Task 1: Custom Object Detection with Faster R-CNN"
   - Choose Public or Private
   - **Don't** initialize with README (we already have one)

2. **Add Repository Description:**
   - Task 1: "Custom Object Detection with Faster R-CNN trained from scratch. Includes training, evaluation, and inference scripts."
   - Task 2: "Automated Quality Inspection System for manufacturing defects. Detects scratches, misalignment, missing components, and discoloration."
   - Task 3: "Custom VLM Design Document for Industrial Quality Inspection. Comprehensive design addressing model selection, optimization, and hallucination mitigation."

3. **Add Topics/Tags:**
   - `computer-vision`
   - `object-detection`
   - `faster-rcnn`
   - `quality-inspection`
   - `deep-learning`
   - `pytorch`

## README Files for Each Task

Each task should have its own README. The existing READMEs in each task folder are good, but you can enhance them.

## Quick Upload Commands

### Task 1 Only:
```bash
cd /home/ros_master/computer_vision
git init
git add task1_object_detection/ requirements.txt README.md
git commit -m "Task 1: Custom Object Detection"
git remote add origin https://github.com/YOUR_USERNAME/task1-repo.git
git push -u origin main
```

### Task 2 Only:
```bash
cd /home/ros_master/computer_vision
git init
git add task2_quality_inspection/ requirements.txt README.md
git commit -m "Task 2: Quality Inspection System"
git remote add origin https://github.com/YOUR_USERNAME/task2-repo.git
git push -u origin main
```

### Task 3 Only:
```bash
cd /home/ros_master/computer_vision
git init
git add task3_vlm_design/ README.md
git commit -m "Task 3: VLM Design Document"
git remote add origin https://github.com/YOUR_USERNAME/task3-repo.git
git push -u origin main
```

## Notes

- Replace `YOUR_USERNAME` with your actual GitHub username
- Create the repositories on GitHub first before pushing
- Use `git status` to check what will be uploaded
- Use `git add -n .` to see what will be added without actually adding


