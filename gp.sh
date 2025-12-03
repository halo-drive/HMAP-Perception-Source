#!/bin/bash

# Git add, commit, push script
# Usage: ./gp.sh [commit message]
# If no message provided, uses a timestamp

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not a git repository${NC}"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${YELLOW}Current branch: ${CURRENT_BRANCH}${NC}"

# Check if there are any changes to commit
if [ -z "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}No changes to commit${NC}"
    exit 0
fi

# Set commit message
if [ -z "$*" ]; then
    COMMIT_MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${YELLOW}No commit message provided, using: ${COMMIT_MSG}${NC}"
else
    COMMIT_MSG="$*"
fi

# Show what will be added
echo -e "${YELLOW}Files to be committed:${NC}"
git status --short

# Add all changes
echo -e "\n${GREEN}Adding all changes...${NC}"
git add .

# Commit
echo -e "${GREEN}Committing: ${COMMIT_MSG}${NC}"
git commit -m "$COMMIT_MSG"

if [ $? -ne 0 ]; then
    echo -e "${RED}Commit failed${NC}"
    exit 1
fi

# Push to main (or current branch)
echo -e "${GREEN}Pushing to ${CURRENT_BRANCH}...${NC}"
git push origin "$CURRENT_BRANCH"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully pushed to ${CURRENT_BRANCH}${NC}"
else
    echo -e "${RED}Push failed. You may need to pull first or resolve conflicts${NC}"
    exit 1
fi
