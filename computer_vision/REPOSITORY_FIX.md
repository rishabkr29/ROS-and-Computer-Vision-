# Repository Not Found - Fix Guide

## Issue
"Repository not found" means either:
1. Repository doesn't exist on GitHub
2. Wrong repository name
3. No access/permissions

## Solution

### Step 1: Verify Repository Exists

Go to this URL in your browser:
```
https://github.com/rishabkr29/object_detection_with_model_training
```

**If you see "404 Not Found":**
- Repository doesn't exist → Create it (see Step 2)

**If you see the repository:**
- Repository exists → Check authentication (see Step 3)

### Step 2: Create Repository on GitHub

1. Go to: https://github.com/new
2. Repository name: `object_detection_with_model_training`
3. Description: "Custom Object Detection with Model Training from Scratch"
4. Choose: Public or Private
5. **IMPORTANT:** Do NOT check:
   - ❌ Add a README file
   - ❌ Add .gitignore
   - ❌ Choose a license
6. Click "Create repository"

### Step 3: Fix Remote and Push

After creating the repository, run:

```bash
# Make sure you're in the right directory
cd ~/computer_vision/coustom_detection_model

# Remove and re-add remote (force HTTPS)
git remote remove origin
git remote add origin https://github.com/rishabkr29/object_detection_with_model_training.git

# Verify remote
git remote -v
# Should show: https://github.com/... (not git@github.com)

# Push
git push -u origin main
```

### Step 4: Authentication

When you push, you'll be asked for:
- **Username**: `rishabkr29`
- **Password**: Use a **Personal Access Token** (not your GitHub password)

**Create Token:**
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name: "Git Push Token"
4. Select scope: ✅ **repo** (full control)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again)
7. Use this token as your password when pushing

## Alternative: Use SSH Instead

If you have SSH keys set up:

```bash
# Remove HTTPS remote
git remote remove origin

# Add SSH remote
git remote add origin git@github.com:rishabkr29/object_detection_with_model_training.git

# Push
git push -u origin main
```

## Quick Checklist

- [ ] Repository exists on GitHub (check URL)
- [ ] Remote is set correctly (`git remote -v`)
- [ ] Using correct flag: `-u` (lowercase)
- [ ] Branch is `main` (`git branch`)
- [ ] Have Personal Access Token ready (for HTTPS)
- [ ] Or SSH keys configured (for SSH)

