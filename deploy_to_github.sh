#!/bin/bash

# 🚀 Trading Intelligence System - GitHub Deployment Script

echo "🚀 Deploying Trading Intelligence System to GitHub..."

# Check if git is configured
if ! git config --global user.name > /dev/null 2>&1; then
    echo "❌ Git not configured. Please run:"
    echo "git config --global user.name 'Your Name'"
    echo "git config --global user.email 'your.email@example.com'"
    exit 1
fi

# Check if remote exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "📝 No remote repository configured."
    echo "Please create a new repository on GitHub and run:"
    echo "git remote add origin https://github.com/YOUR_USERNAME/trading-intelligence-system.git"
    echo ""
    echo "Or if you want to create a new repository automatically, please provide:"
    echo "1. Your GitHub username"
    echo "2. Repository name (default: trading-intelligence-system)"
    echo "3. GitHub personal access token"
    echo ""
    read -p "GitHub Username: " github_username
    read -p "Repository Name (default: trading-intelligence-system): " repo_name
    repo_name=${repo_name:-trading-intelligence-system}
    read -p "GitHub Personal Access Token: " github_token
    
    if [ -n "$github_username" ] && [ -n "$github_token" ]; then
        echo "🔗 Creating GitHub repository..."
        curl -u "$github_username:$github_token" https://api.github.com/user/repos \
            -d "{\"name\":\"$repo_name\",\"description\":\"World-Class Multi-Asset Trading Intelligence System\",\"private\":false,\"auto_init\":false}"
        
        echo "🔗 Adding remote origin..."
        git remote add origin "https://github.com/$github_username/$repo_name.git"
    else
        echo "❌ Please provide GitHub credentials to continue."
        exit 1
    fi
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
git push -u origin main

echo "✅ Deployment complete!"
echo ""
echo "🌐 Your repository is now available at:"
git remote get-url origin
echo ""
echo "📊 Dashboard can be run locally with:"
echo "python run_dashboard.py"
echo ""
echo "🚀 To deploy to Streamlit Cloud:"
echo "1. Go to https://share.streamlit.io/"
echo "2. Connect your GitHub repository"
echo "3. Deploy streamlit_complete_dashboard.py"
echo ""
echo "🎯 Repository includes:"
echo "- Complete trading intelligence system"
echo "- 12 fully functional dashboard screens"
echo "- Multi-agent architecture"
echo "- Real-time analytics"
echo "- Advanced ML models"
echo "- Comprehensive documentation"
