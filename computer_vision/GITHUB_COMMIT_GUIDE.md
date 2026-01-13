# GitHub Commit Guide - Step by Step

This guide will help you commit each task individually to GitHub.

## Prerequisites

1. **GitHub Account**: Make sure you have a GitHub account
2. **Git Installed**: Check if git is installed
   ```bash
   git --version
   ```
3. **GitHub CLI or Web**: You can use GitHub web interface or command line

## Step-by-Step: Commit Each Task

### Method 1: Using the Preparation Script (Recommended)

#### Step 1: Prepare Each Task for Upload

```bash
cd /home/ros_master/computer_vision

# Prepare Task 1
python3 scripts/prepare_github_upload.py --task task1

# Prepare Task 2
python3 scripts/prepare_github_upload.py --task task2

# Prepare Task 3
python3 scripts/prepare_github_upload.py --task task3
```

This creates clean directories: `task1_github/`, `task2_github/`, `task3_github/`

#### Step 2: Create GitHub Repositories

**On GitHub Website:**
1. Go to https://github.com/new
2. Create three repositories:
   - **Repository 1**: `computer-vision-task1` (or your preferred name)
     - Description: "Custom Object Detection with Faster R-CNN trained from scratch"
     - Public or Private (your choice)
     - **DO NOT** initialize with README, .gitignore, or license
   
   - **Repository 2**: `computer-vision-task2`
     - Description: "Automated Quality Inspection System for manufacturing defects"
     - Public or Private
     - **DO NOT** initialize with README
   
   - **Repository 3**: `computer-vision-task3`
     - Description: "Custom VLM Design Document for Industrial Quality Inspection"
     - Public or Private
     - **DO NOT** initialize with README

#### Step 3: Commit and Push Task 1

```bash
cd /home/ros_master/computer_vision/task1_github

# Initialize git
git init

# Add all files
git add .

# Check what will be committed
git status

# Commit
git commit -m "Initial commit: Task 1 - Custom Object Detection with Faster R-CNN"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-task1.git

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

#### Step 4: Commit and Push Task 2

```bash
cd /home/ros_master/computer_vision/task2_github

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Task 2 - Automated Quality Inspection System"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-task2.git

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

#### Step 5: Commit and Push Task 3

```bash
cd /home/ros_master/computer_vision/task3_github

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Task 3 - VLM Design Document for Industrial Quality Inspection"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-task3.git

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

### Method 2: Direct Commit from Task Directories

If you prefer to commit directly from the task directories:

#### Task 1:
```bash
cd /home/ros_master/computer_vision/task1_object_detection

# Initialize git
git init

# Create .gitignore if needed
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pth
*.pt
checkpoints/
logs/
data/train/images/*.jpg
data/val/images/*.jpg
results/*.jpg
EOF

# Add files
git add .

# Commit
git commit -m "Task 1: Custom Object Detection with Faster R-CNN"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-task1.git

# Push
git branch -M main
git push -u origin main
```

#### Task 2:
```bash
cd /home/ros_master/computer_vision/task2_quality_inspection

git init
git add .
git commit -m "Task 2: Automated Quality Inspection System"
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-task2.git
git branch -M main
git push -u origin main
```

#### Task 3:
```bash
cd /home/ros_master/computer_vision/task3_vlm_design

git init
git add .
git commit -m "Task 3: VLM Design Document for Industrial Quality Inspection"
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-task3.git
git branch -M main
git push -u origin main
```

## Authentication

### Option 1: Personal Access Token (Recommended)

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Use token as password when pushing:
   ```bash
   git push -u origin main
   # Username: YOUR_USERNAME
   # Password: YOUR_TOKEN
   ```

### Option 2: SSH Key

1. Generate SSH key:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```
2. Add to GitHub: Settings → SSH and GPG keys
3. Use SSH URL:
   ```bash
   git remote add origin git@github.com:YOUR_USERNAME/computer-vision-task1.git
   ```

### Option 3: GitHub CLI

```bash
# Install GitHub CLI
sudo apt install gh  # or brew install gh

# Login
gh auth login

# Then push normally
git push -u origin main
```

## Troubleshooting

### Error: "remote origin already exists"
```bash
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin https://github.com/YOUR_USERNAME/repo-name.git
```

### Error: "Authentication failed"
- Use Personal Access Token instead of password
- Check token has `repo` scope
- Verify username is correct

### Error: "Repository not found"
- Make sure repository exists on GitHub
- Check repository name matches
- Verify you have access to the repository

### Error: "Large files"
If you get errors about large files:
```bash
# Check file sizes
find . -type f -size +50M

# Add to .gitignore
echo "large_file.pth" >> .gitignore
git rm --cached large_file.pth
git commit -m "Remove large file"
```

## Verification

After pushing, verify on GitHub:

1. Go to your repository: `https://github.com/YOUR_USERNAME/computer-vision-task1`
2. Check files are uploaded
3. Verify README.md displays correctly
4. Check results files are included

## Quick Commands Summary

```bash
# Prepare all tasks
python3 scripts/prepare_github_upload.py --task task1
python3 scripts/prepare_github_upload.py --task task2
python3 scripts/prepare_github_upload.py --task task3

# For each task directory:
cd taskX_github
git init
git add .
git commit -m "Initial commit: Task X"
git remote add origin https://github.com/YOUR_USERNAME/repo-name.git
git branch -M main
git push -u origin main
```

## Next Steps After Upload

1. **Add Repository Descriptions** on GitHub
2. **Add Topics/Tags**: computer-vision, object-detection, pytorch, etc.
3. **Add License** if needed
4. **Create Releases** for major versions
5. **Add Screenshots** to README if you have them

## Notes

- Replace `YOUR_USERNAME` with your actual GitHub username
- Repository names can be customized
- Use descriptive commit messages
- Keep .gitignore updated to exclude large files

