# OCR Setup Guide for BikeRisk

BikeRisk now supports **automatic OCR** for image-based PDFs! When a PDF doesn't have embedded text (scanned documents), the system automatically falls back to OCR extraction.

## 🔄 How It Works

The extraction pipeline uses a **3-tier fallback system**:

1. **PDFPlumber** (fast, text-based PDFs) ⚡
2. **PyPDF** (fallback for text-based PDFs) 📄
3. **OCR** (for image-based/scanned PDFs) 📷
   - Google Cloud Vision API (primary)
   - Claude PDF Vision (fallback)

## 🚀 Setup Instructions

### Option 1: Google Cloud Vision (Recommended)

**Why Google Cloud Vision?**
- Best accuracy for technical documents
- Excellent table/multi-column handling
- Free tier: 1,000 pages/month
- After free tier: $1.50 per 1,000 pages

**Setup Steps:**

1. **Create Google Cloud Account**
   - Go to https://console.cloud.google.com
   - Create new project or select existing one

2. **Enable Cloud Vision API**
   - Navigate to "APIs & Services" → "Library"
   - Search for "Cloud Vision API"
   - Click "Enable"

3. **Create Service Account**
   - Go to "IAM & Admin" → "Service Accounts"
   - Click "Create Service Account"
   - Name: `bikerisk-ocr` (or any name)
   - Grant role: "Cloud Vision API User"

4. **Generate API Key**
   - Click on your service account
   - Go to "Keys" tab
   - Click "Add Key" → "Create new key"
   - Choose JSON format
   - Download the JSON file

5. **Add to Railway Environment Variables**
   - Copy the **entire contents** of the JSON file
   - In Railway, add environment variable:
     - Name: `GOOGLE_CLOUD_VISION_CREDENTIALS`
     - Value: Paste the entire JSON content (all of it, including braces)

6. **Redeploy**
   - Railway will automatically redeploy with new env var
   - OCR will now work!

### Option 2: Claude PDF Vision (Automatic Fallback)

If Google Cloud Vision is not configured, the system automatically uses Claude PDF Vision.

**Requirements:**
- Your existing `ANTHROPIC_API_KEY` (already configured!)
- No additional setup needed

**Limitations:**
- Uses Claude API tokens
- May be slower than Google Vision for multi-page documents
- Still very accurate!

## 📊 Testing OCR

To test if OCR is working:

1. **Upload a scanned PDF** (image-based, no text layer)
2. Watch the extraction progress
3. Look for: **"📷 Image-based PDF detected - OCR extraction was used"**
4. Check "Extraction Information" expander to see which method was used:
   - "Google Cloud Vision OCR" ✅
   - "Claude PDF Vision OCR" ✅ (fallback)

## 🔍 Troubleshooting

### "OCR extraction failed"
**Possible causes:**
- Google Cloud Vision credentials not set or invalid
- Claude API key not set
- PDF is corrupted or heavily encrypted

**Solutions:**
1. Check Railway environment variables are set correctly
2. Verify Google Cloud Vision API is enabled
3. Check Railway logs for specific error messages

### "Google Vision unavailable, trying Claude PDF Vision"
**This is normal!** It means:
- Google Cloud Vision credentials not configured, OR
- Google API call failed

The system automatically falls back to Claude PDF Vision.

### Google Cloud Vision JSON format error
**Make sure you:**
- Copy the ENTIRE JSON file contents (including `{` and `}`)
- Don't add extra quotes around it in Railway
- The JSON should start with `{` and end with `}`

## 💰 Cost Considerations

### Google Cloud Vision Pricing
- **Free tier:** 1,000 pages/month
- **After free:** $1.50 per 1,000 pages
- Most standards are 20-100 pages → very affordable

### Claude PDF Vision Pricing
- Uses your existing Claude API tokens
- Approximately same cost as regular Claude requests
- Good for occasional OCR use

## 🎯 Recommendations

**For Production:**
- Set up Google Cloud Vision (better accuracy + free tier)
- Keep Claude as automatic fallback (already configured!)

**For Development/Testing:**
- Claude PDF Vision works great (no extra setup!)
- Upgrade to Google Cloud Vision when you need higher volume

## 📝 Environment Variables Summary

Required for OCR to work:

```bash
# Option 1: Google Cloud Vision (recommended)
GOOGLE_CLOUD_VISION_CREDENTIALS={"type":"service_account","project_id":"..."}

# Option 2: Claude PDF Vision (automatic fallback)
ANTHROPIC_API_KEY=sk-ant-...  # Already configured!
```

## ✅ Success Indicators

When OCR is working correctly, you'll see:

1. ✅ Image-based PDFs extract successfully
2. ✅ Blue info box: "📷 Image-based PDF detected - OCR extraction was used"
3. ✅ Extraction method shows "Google Cloud Vision OCR" or "Claude PDF Vision OCR"
4. ✅ Requirements table populated with extracted data

---

**Need help?** Check Railway logs or contact the dev team!
