# Quick GitHub Upload - Step by Step

## Method 1: Use the Preparation Script (Easiest)

```bash
cd /home/ros_master/computer_vision

# Prepare Task 1
python3 scripts/prepare_github_upload.py --task task1

# Prepare Task 2
python3 scripts/prepare_github_upload.py --task task2

# Prepare Task 3
python3 scripts/prepare_github_upload.py --task task3
```

This creates clean directories (`task1_github/`, `task2_github/`, `task3_github/`) ready for upload.

Then for each task:

```bash
# Task 1
cd task1_github
git init
git add .
git commit -m "Initial commit: Task 1 - Custom Object Detection"
git remote add origin https://github.com/YOUR_USERNAME/task1-repo.git
git branch -M main
git push -u origin main

# Task 2
cd ../task2_github
git init
git add .
git commit -m "Initial commit: Task 2 - Quality Inspection"
git remote add origin https://github.com/YOUR_USERNAME/task2-repo.git
git branch -M main
git push -u origin main

# Task 3
cd ../task3_github
git init
git add .
git commit -m "Initial commit: Task 3 - VLM Design"
git remote add origin https://github.com/YOUR_USERNAME/task3-repo.git
git branch -M main
git push -u origin main
```

## Method 2: Manual Upload (Direct)

### Task 1:
```bash
cd /home/ros_master/computer_vision/task1_object_detection
git init
git add .
git commit -m "Task 1: Custom Object Detection"
git remote add origin https://github.com/YOUR_USERNAME/task1-repo.git
git push -u origin main
```

### Task 2:
```bash
cd /home/ros_master/computer_vision/task2_quality_inspection
git init
git add .
git commit -m "Task 2: Quality Inspection System"
git remote add origin https://github.com/YOUR_USERNAME/task2-repo.git
git push -u origin main
```

### Task 3:
```bash
cd /home/ros_master/computer_vision/task3_vlm_design
git init
git add .
git commit -m "Task 3: VLM Design Document"
git remote add origin https://github.com/YOUR_USERNAME/task3-repo.git
git push -u origin main
```

## Before Uploading

1. **Create repositories on GitHub first:**
   - Go to github.com
   - Click "New Repository"
   - Create: `task1-repo`, `task2-repo`, `task3-repo`
   - Don't initialize with README

2. **Replace YOUR_USERNAME** with your actual GitHub username

3. **Check what will be uploaded:**
   ```bash
   git status
   git add -n .  # Dry run to see what will be added
   ```

## Important Notes

- Large files (>100MB) may need Git LFS
- Model checkpoints (.pth files) are excluded by .gitignore
- Sample images are excluded (keep structure only)
- Results are excluded (but sample outputs are included)

## Repository Names Suggestions

- `computer-vision-task1-object-detection`
- `computer-vision-task2-quality-inspection`
- `computer-vision-task3-vlm-design`

Or shorter:
- `cv-task1-detection`
- `cv-task2-inspection`
- `cv-task3-vlm-design`


