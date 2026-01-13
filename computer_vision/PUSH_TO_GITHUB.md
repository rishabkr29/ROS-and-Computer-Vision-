# Push to GitHub - Commands

You've already committed locally. Now you need to:

## Step 1: Add Remote Repository

Replace `YOUR_USERNAME` with your GitHub username and `REPO_NAME` with your repository name:

```bash
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

**Example:**
```bash
git remote add origin https://github.com/johndoe/computer-vision-task1.git
```

## Step 2: Rename Branch to Main

```bash
git branch -M main
```

## Step 3: Push to GitHub

```bash
git push -u origin main
```

## Complete Commands (Copy & Paste)

**Replace `YOUR_USERNAME` and `REPO_NAME`:**

```bash
# Add remote (replace with your details)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

## If Remote Already Exists

If you get "remote origin already exists":

```bash
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push
git branch -M main
git push -u origin main
```

## Authentication

When you run `git push`, you'll be asked for:
- **Username**: Your GitHub username
- **Password**: Use a Personal Access Token (not your password)

**Create Token:**
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scope: `repo`
4. Copy token and use as password

