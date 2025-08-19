#!/bin/bash

echo "🚀 Creating GitHub Repository for Trading Intelligence System"
echo ""

# Get repository details
read -p "Enter your GitHub username: " github_username
read -p "Enter repository name (default: trading-intelligence-system): " repo_name
repo_name=${repo_name:-trading-intelligence-system}

echo ""
echo "📝 Repository Details:"
echo "Username: $github_username"
echo "Repository: $repo_name"
echo ""

# Check if we have a GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "🔑 Please provide your GitHub Personal Access Token:"
    echo "   (You can create one at: https://github.com/settings/tokens)"
    read -s -p "GitHub Token: " github_token
    echo ""
else
    github_token=$GITHUB_TOKEN
fi

echo "🔗 Creating GitHub repository..."

# Create the repository
response=$(curl -s -u "$github_username:$github_token" \
    https://api.github.com/user/repos \
    -d "{
        \"name\": \"$repo_name\",
        \"description\": \"World-Class Multi-Asset Trading Intelligence System with Real-time Analytics Dashboard\",
        \"private\": false,
        \"auto_init\": false,
        \"has_issues\": true,
        \"has_wiki\": true,
        \"has_downloads\": true
    }")

# Check if repository was created successfully
if echo "$response" | grep -q "already exists"; then
    echo "⚠️  Repository already exists. Using existing repository."
elif echo "$response" | grep -q "Bad credentials"; then
    echo "❌ Invalid GitHub token. Please check your token and try again."
    exit 1
elif echo "$response" | grep -q "Not Found"; then
    echo "❌ GitHub username not found. Please check your username and try again."
    exit 1
else
    echo "✅ Repository created successfully!"
fi

# Add remote origin
echo "🔗 Adding remote origin..."
git remote remove origin 2>/dev/null
git remote add origin "https://github.com/$github_username/$repo_name.git"

# Push to GitHub
echo "📤 Pushing code to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Your Trading Intelligence System is now on GitHub!"
    echo ""
    echo "🌐 Repository URL: https://github.com/$github_username/$repo_name"
    echo ""
    echo "📊 To run the dashboard locally:"
    echo "   cd /Users/davidwestera/trading-intelligence-system"
    echo "   python run_dashboard.py"
    echo ""
    echo "🚀 To deploy to Streamlit Cloud:"
    echo "   1. Go to https://share.streamlit.io/"
    echo "   2. Connect your GitHub repository"
    echo "   3. Deploy streamlit_complete_dashboard.py"
    echo ""
    echo "📚 Documentation:"
    echo "   - README.md - Project overview"
    echo "   - DEPLOYMENT_README.md - Detailed deployment guide"
    echo "   - DEPLOYMENT_SUMMARY.md - Quick summary"
    echo ""
    echo "🎯 Your world-class trading intelligence system is ready!"
else
    echo "❌ Failed to push to GitHub. Please check your credentials and try again."
    exit 1
fi
