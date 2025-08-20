# 🎬 YOUTUBE API QUICK SETUP

## 🚀 **5-MINUTE SETUP GUIDE**

### **STEP 1: GOOGLE CLOUD CONSOLE**
1. Go to: [https://console.cloud.google.com/](https://console.cloud.google.com/)
2. Sign in with your Google account
3. Create new project: "Trading Sentiment Analysis"

### **STEP 2: ENABLE API**
1. Go to: APIs & Services → Library
2. Search: "YouTube Data API v3"
3. Click "Enable"

### **STEP 3: CREATE API KEY**
1. Go to: APIs & Services → Credentials
2. Click "Create Credentials" → "API Key"
3. Copy the API key (format: `AIzaSyC...`)

### **STEP 4: RESTRICT KEY (RECOMMENDED)**
1. Click on your API key
2. Under "API restrictions": Select "YouTube Data API v3"
3. Click "Save"

### **STEP 5: ADD TO ENVIRONMENT**
```bash
# Edit env_real_keys.env and add:
YOUTUBE_API_KEY=AIzaSyC...your_actual_key_here
```

### **STEP 6: TEST**
```bash
python test_youtube_api_setup.py
```

## ✅ **EXPECTED RESULT**
```
✅ YouTube API Key is valid!
✅ Found 5 videos for AAPL
✅ Video details retrieved
✅ Found 5 comments
🎉 YouTube API setup test complete!
```

## 🎯 **WHAT YOU GET**
- **Free tier**: 10,000 requests/day
- **Video search**: Find stock analysis videos
- **Comment analysis**: Crowd sentiment
- **Engagement metrics**: Views, likes, comments
- **Real-time data**: Live streams and reactions

## 📞 **NEED HELP?**
- Check `YOUTUBE_API_SETUP_GUIDE.md` for detailed instructions
- Run `python test_youtube_api_setup.py` to test
- Verify API key format: `AIzaSyC...` (39 characters)

**Status: Ready to implement** 🚀
