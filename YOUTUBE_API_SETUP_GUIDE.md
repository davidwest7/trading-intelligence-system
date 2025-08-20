# YouTube API Setup Guide

## 🚨 **CURRENT ISSUE: 403 Error**

Your YouTube API key is configured but getting a 403 error, which means the YouTube Data API v3 is not enabled for your project.

## 🔧 **STEP-BY-STEP FIX:**

### **Step 1: Go to Google Cloud Console**
1. Visit: [https://console.cloud.google.com/](https://console.cloud.google.com/)
2. Sign in with your Google account
3. Select your project: `ecstatic-gantry-469521-d6`

### **Step 2: Enable YouTube Data API v3**
1. In the left sidebar, click **"APIs & Services"** → **"Library"**
2. Search for **"YouTube Data API v3"**
3. Click on **"YouTube Data API v3"**
4. Click **"ENABLE"** button
5. Wait for the API to be enabled

### **Step 3: Verify API Key Permissions**
1. Go to **"APIs & Services"** → **"Credentials"**
2. Find your API key: `AIzaSyCvpFRdM20BoRSzKx92M0yLvuLeFlK10Os`
3. Click on the API key to edit it
4. Under **"API restrictions"**, make sure:
   - **"Don't restrict key"** is selected, OR
   - **"Restrict key"** is selected with **"YouTube Data API v3"** in the list

### **Step 4: Check Billing (if needed)**
1. Go to **"Billing"** in the left sidebar
2. Make sure billing is enabled for your project
3. YouTube API has a generous free tier (10,000 units/day)

### **Step 5: Test the API**
After enabling the API, run this test:

```bash
python test_youtube_api.py
```

## 📊 **EXPECTED RESULTS:**

After proper setup, you should see:
```
🧪 Testing YouTube API Integration
========================================
API Key Status: ✅ Found
Key Format: ✅ Valid
Monitor Status: ✅ Ready

🔍 Testing YouTube API call...
API Call Status: WORKING
✅ Videos Found: 3
📊 Quota Used: 100/10000

📺 First Video:
   Title: Apple Stock Analysis - Latest Updates...
   Channel: Financial Channel Name
   Published: 2025-08-19
```

## 🔍 **TROUBLESHOOTING:**

### **If still getting 403:**
1. **Wait 5-10 minutes** after enabling the API
2. **Check API restrictions** on your key
3. **Verify project selection** in Google Cloud Console
4. **Check billing status** for the project

### **If getting quota exceeded:**
- The free tier allows 10,000 units/day
- Each search request costs 100 units
- You can make 100 searches per day

### **If getting other errors:**
- Check the error message for specific details
- Verify your API key is correct
- Make sure you're using the right project

## 🎯 **NEXT STEPS AFTER FIX:**

Once the API is working:

1. **Test with multiple symbols**:
   ```python
   symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
   ```

2. **Test comment sentiment analysis**:
   ```python
   video_sentiment = await monitor.get_video_sentiment(video_id)
   ```

3. **Integrate with existing system**:
   ```python
   # Combine with Finnhub + NewsAPI
   comprehensive_analysis = await get_comprehensive_analysis(symbol)
   ```

## 📞 **NEED HELP?**

If you're still having issues:

1. **Check Google Cloud Console** for any error messages
2. **Verify API is enabled** in the API Library
3. **Check API key restrictions** in Credentials
4. **Ensure billing is enabled** for the project

---

**Once you've enabled the YouTube Data API v3, run the test again and let me know the results!**
