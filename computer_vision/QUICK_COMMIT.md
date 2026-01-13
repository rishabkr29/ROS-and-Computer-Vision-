# Quick GitHub Commit - Copy & Paste Commands

## Step 1: Prepare Tasks (One Time)

```bash
cd /home/ros_master/computer_vision

# Prepare all tasks
python3 scripts/prepare_github_upload.py --task task1
python3 scripts/prepare_github_upload.py --task task2
python3 scripts/prepare_github_upload.py --task task3
```

## Step 2: Create GitHub Repositories

**Go to:** https://github.com/new

Create three repositories (replace `YOUR_USERNAME`):
1. `computer-vision-task1`
2. `computer-vision-task2`
3. `computer-vision-task3`

**Important:** Don't initialize with README!

## Step 3: Commit Task 1

```bash
cd /home/ros_master/computer_vision/task1_github
git init
git add .
git commit -m "Initial commit: Task 1 - Custom Object Detection"
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-task1.git
git branch -M main
git push -u origin main
```

## Step 4: Commit Task 2

```bash
cd /home/ros_master/computer_vision/task2_github
git init
git add .
git commit -m "Initial commit: Task 2 - Quality Inspection System"
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-task2.git
git branch -M main
git push -u origin main
```

## Step 5: Commit Task 3

```bash
cd /home/ros_master/computer_vision/task3_github
git init
git add .
git commit -m "Initial commit: Task 3 - VLM Design Document"
git remote add origin https://github.com/YOUR_USERNAME/computer-vision-task3.git
git branch -M main
git push -u origin main
```

## Alternative: Use Helper Script

```bash
cd /home/ros_master/computer_vision
./scripts/github_commit.sh
```

Follow the interactive prompts.

## Authentication

When pushing, you'll be asked for credentials:
- **Username**: Your GitHub username
- **Password**: Use a Personal Access Token (not your password)

**Create Token:**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Copy token and use as password

## Replace YOUR_USERNAME

In all commands above, replace `YOUR_USERNAME` with your actual GitHub username.

Example:
- If your username is `johndoe`, use: `https://github.com/johndoe/computer-vision-task1.git`

## Troubleshooting

**"remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/repo-name.git
```

**"Authentication failed"**
- Use Personal Access Token instead of password
- Make sure token has `repo` scope

**"Repository not found"**
- Create repository on GitHub first
- Check repository name matches exactly

