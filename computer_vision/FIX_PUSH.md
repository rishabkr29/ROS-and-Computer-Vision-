# Fix Push Issues

## Issue 1: Wrong Flag
You used `-U` (capital U) but it should be `-u` (lowercase u)

## Issue 2: Repository Not Found
This means either:
- Repository doesn't exist on GitHub
- Wrong repository name
- No access to the repository

## Solution Steps

### Step 1: Check Current Remote
```bash
git remote -v
```

### Step 2: Remove Existing Remote
```bash
git remote remove origin
```

### Step 3: Verify Repository Exists on GitHub
Go to: https://github.com/rishabkr29/object_detection

Make sure:
- Repository exists
- You have access to it
- Repository name is exactly: `object_detection` (not `obeject_detection`)

### Step 4: Add Correct Remote
```bash
# Use the correct repository name
git remote add origin https://github.com/rishabkr29/object_detection.git
```

### Step 5: Push with Correct Flag
```bash
# Use lowercase -u, not -U
git push -u origin main
```

## Complete Fix Commands

```bash
# 1. Check current remote
git remote -v

# 2. Remove existing remote
git remote remove origin

# 3. Add correct remote (verify repository name first!)
git remote add origin https://github.com/rishabkr29/object_detection.git

# 4. Verify remote added
git remote -v

# 5. Push with lowercase -u
git push -u origin main
```

## If Repository Doesn't Exist

1. Go to: https://github.com/new
2. Repository name: `object_detection`
3. Description: "Custom Object Detection with Faster R-CNN"
4. Choose Public or Private
5. **DO NOT** initialize with README
6. Click "Create repository"
7. Then run the push command again

