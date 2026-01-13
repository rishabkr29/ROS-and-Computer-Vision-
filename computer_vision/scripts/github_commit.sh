#!/bin/bash
# Helper script to commit tasks to GitHub

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}GitHub Commit Helper Script${NC}"
echo "================================"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed${NC}"
    exit 1
fi

# Get GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo -e "${RED}Error: GitHub username is required${NC}"
    exit 1
fi

# Function to commit a task
commit_task() {
    local TASK_NUM=$1
    local TASK_NAME=$2
    local REPO_NAME=$3
    local COMMIT_MSG=$4
    local TASK_DIR="../task${TASK_NUM}_github"
    
    echo -e "\n${YELLOW}Processing Task ${TASK_NUM}: ${TASK_NAME}${NC}"
    echo "----------------------------------------"
    
    if [ ! -d "$TASK_DIR" ]; then
        echo -e "${RED}Error: Directory $TASK_DIR not found${NC}"
        echo "Run: python3 scripts/prepare_github_upload.py --task task${TASK_NUM}"
        return 1
    fi
    
    cd "$TASK_DIR" || exit 1
    
    # Check if already a git repo
    if [ -d ".git" ]; then
        echo "Git repository already initialized"
    else
        echo "Initializing git repository..."
        git init
    fi
    
    # Add all files
    echo "Adding files..."
    git add .
    
    # Check if there are changes
    if git diff --staged --quiet && [ -z "$(git status -s)" ]; then
        echo -e "${YELLOW}No changes to commit${NC}"
        cd - > /dev/null
        return 0
    fi
    
    # Commit
    echo "Committing changes..."
    git commit -m "$COMMIT_MSG" || {
        echo -e "${YELLOW}No changes to commit${NC}"
        cd - > /dev/null
        return 0
    }
    
    # Check if remote exists
    if git remote | grep -q "origin"; then
        echo "Remote 'origin' already exists"
        read -p "Do you want to update it? (y/n): " UPDATE_REMOTE
        if [ "$UPDATE_REMOTE" = "y" ]; then
            git remote set-url origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
        fi
    else
        echo "Adding remote origin..."
        git remote add origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
    fi
    
    # Set branch to main
    git branch -M main 2>/dev/null || true
    
    # Push
    echo -e "${YELLOW}Ready to push to GitHub${NC}"
    echo "Repository: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
    read -p "Do you want to push now? (y/n): " PUSH_NOW
    
    if [ "$PUSH_NOW" = "y" ]; then
        echo "Pushing to GitHub..."
        git push -u origin main || {
            echo -e "${RED}Push failed. You may need to:${NC}"
            echo "1. Create the repository on GitHub first"
            echo "2. Set up authentication (Personal Access Token)"
            echo "3. Check repository name: $REPO_NAME"
        }
    else
        echo -e "${GREEN}Files committed locally. Push manually with:${NC}"
        echo "  cd $TASK_DIR"
        echo "  git push -u origin main"
    fi
    
    cd - > /dev/null
    echo -e "${GREEN}Task ${TASK_NUM} completed!${NC}"
}

# Main menu
echo ""
echo "Select which task to commit:"
echo "1) Task 1 - Object Detection"
echo "2) Task 2 - Quality Inspection"
echo "3) Task 3 - VLM Design"
echo "4) All Tasks"
echo "5) Exit"
read -p "Enter choice (1-5): " CHOICE

case $CHOICE in
    1)
        commit_task 1 "Object Detection" "computer-vision-task1" \
            "Initial commit: Task 1 - Custom Object Detection with Faster R-CNN"
        ;;
    2)
        commit_task 2 "Quality Inspection" "computer-vision-task2" \
            "Initial commit: Task 2 - Automated Quality Inspection System"
        ;;
    3)
        commit_task 3 "VLM Design" "computer-vision-task3" \
            "Initial commit: Task 3 - VLM Design Document for Industrial Quality Inspection"
        ;;
    4)
        commit_task 1 "Object Detection" "computer-vision-task1" \
            "Initial commit: Task 1 - Custom Object Detection with Faster R-CNN"
        commit_task 2 "Quality Inspection" "computer-vision-task2" \
            "Initial commit: Task 2 - Automated Quality Inspection System"
        commit_task 3 "VLM Design" "computer-vision-task3" \
            "Initial commit: Task 3 - VLM Design Document for Industrial Quality Inspection"
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Done!${NC}"

