#!/usr/bin/env python3
"""
Real API Integration Test
Tests the provided API keys and integrates them into the trading intelligence system
"""

import os
import asyncio
import requests
import praw
import tweepy
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv('env_real_keys.env')

class RealAPITester:
    """Test real API integrations"""
    
    def __init__(self):
        self.results = {}
        
    async def test_twitter_api(self):
        """Test Twitter/X API integration"""
        print("🐦 Testing Twitter/X API...")
        
        try:
            # Test with Bearer Token
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            if bearer_token:
                headers = {
                    'Authorization': f'Bearer {bearer_token}',
                    'User-Agent': 'TradingIntelligenceBot/1.0'
                }
                
                # Test search endpoint
                url = "https://api.twitter.com/2/tweets/search/recent"
                params = {
                    'query': 'AAPL stock',
                    'max_results': 10
                }
                
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    tweets = data.get('data', [])
                    print(f"✅ Twitter API working! Found {len(tweets)} tweets about AAPL")
                    self.results['twitter'] = {
                        'status': 'success',
                        'tweets_found': len(tweets),
                        'sample_tweet': tweets[0] if tweets else None
                    }
                else:
                    print(f"❌ Twitter API error: {response.status_code} - {response.text}")
                    self.results['twitter'] = {
                        'status': 'error',
                        'error': f"{response.status_code}: {response.text}"
                    }
            else:
                print("❌ Twitter Bearer Token not found")
                self.results['twitter'] = {'status': 'error', 'error': 'No bearer token'}
                
        except Exception as e:
            print(f"❌ Twitter API test failed: {e}")
            self.results['twitter'] = {'status': 'error', 'error': str(e)}
    
    async def test_reddit_api(self):
        """Test Reddit API integration"""
        print("📱 Testing Reddit API...")
        
        try:
            client_id = os.getenv('REDDIT_CLIENT_ID')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET')
            user_agent = os.getenv('REDDIT_USER_AGENT')
            
            if client_id and client_secret:
                reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
                
                # Test with wallstreetbets subreddit
                subreddit = reddit.subreddit('wallstreetbets')
                posts = []
                
                for post in subreddit.hot(limit=5):
                    posts.append({
                        'title': post.title,
                        'score': post.score,
                        'created_utc': post.created_utc,
                        'num_comments': post.num_comments
                    })
                
                print(f"✅ Reddit API working! Found {len(posts)} posts from r/wallstreetbets")
                self.results['reddit'] = {
                    'status': 'success',
                    'posts_found': len(posts),
                    'sample_post': posts[0] if posts else None
                }
            else:
                print("❌ Reddit credentials not found")
                self.results['reddit'] = {'status': 'error', 'error': 'No credentials'}
                
        except Exception as e:
            print(f"❌ Reddit API test failed: {e}")
            self.results['reddit'] = {'status': 'error', 'error': str(e)}
    
    async def test_openai_api(self):
        """Test OpenAI API integration"""
        print("🤖 Testing OpenAI API...")
        
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                # Test with a simple completion
                url = "https://api.openai.com/v1/chat/completions"
                data = {
                    'model': 'gpt-3.5-turbo',
                    'messages': [
                        {'role': 'user', 'content': 'What is the current sentiment for AAPL stock?'}
                    ],
                    'max_tokens': 50
                }
                
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    print(f"✅ OpenAI API working! Response: {content[:100]}...")
                    self.results['openai'] = {
                        'status': 'success',
                        'response': content,
                        'model_used': 'gpt-3.5-turbo'
                    }
                else:
                    print(f"❌ OpenAI API error: {response.status_code} - {response.text}")
                    self.results['openai'] = {
                        'status': 'error',
                        'error': f"{response.status_code}: {response.text}"
                    }
            else:
                print("❌ OpenAI API key not found")
                self.results['openai'] = {'status': 'error', 'error': 'No API key'}
                
        except Exception as e:
            print(f"❌ OpenAI API test failed: {e}")
            self.results['openai'] = {'status': 'error', 'error': str(e)}
    
    async def test_github_api(self):
        """Test GitHub API integration"""
        print("🐙 Testing GitHub API...")
        
        try:
            token = os.getenv('GITHUB_TOKEN')
            if token:
                headers = {
                    'Authorization': f'token {token}',
                    'Accept': 'application/vnd.github.v3+json'
                }
                
                # Test user info
                url = "https://api.github.com/user"
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    user_data = response.json()
                    print(f"✅ GitHub API working! User: {user_data.get('login', 'Unknown')}")
                    self.results['github'] = {
                        'status': 'success',
                        'user': user_data.get('login'),
                        'repos_url': user_data.get('repos_url')
                    }
                else:
                    print(f"❌ GitHub API error: {response.status_code} - {response.text}")
                    self.results['github'] = {
                        'status': 'error',
                        'error': f"{response.status_code}: {response.text}"
                    }
            else:
                print("❌ GitHub token not found")
                self.results['github'] = {'status': 'error', 'error': 'No token'}
                
        except Exception as e:
            print(f"❌ GitHub API test failed: {e}")
            self.results['github'] = {'status': 'error', 'error': str(e)}
    
    async def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Real API Integration Tests...")
        print("=" * 50)
        
        await self.test_twitter_api()
        await self.test_reddit_api()
        await self.test_openai_api()
        await self.test_github_api()
        
        print("\n" + "=" * 50)
        print("📊 API Test Results Summary:")
        print("=" * 50)
        
        for api, result in self.results.items():
            status = "✅ SUCCESS" if result['status'] == 'success' else "❌ FAILED"
            print(f"{api.upper()}: {status}")
            if result['status'] == 'error':
                print(f"   Error: {result['error']}")
        
        # Save results
        with open('api_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: api_test_results.json")
        
        return self.results

async def main():
    """Main test function"""
    tester = RealAPITester()
    results = await tester.run_all_tests()
    
    # Summary
    successful_apis = sum(1 for r in results.values() if r['status'] == 'success')
    total_apis = len(results)
    
    print(f"\n🎯 SUMMARY: {successful_apis}/{total_apis} APIs working successfully")
    
    if successful_apis == total_apis:
        print("🎉 All APIs are working! Ready for real data integration.")
    else:
        print("⚠️  Some APIs need attention. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
