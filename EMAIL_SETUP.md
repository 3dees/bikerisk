# Email Feedback Setup Guide

The BikeRisk app can now send user feedback directly to your email!

## Quick Setup (5 minutes)

### 1. Create Gmail App Password

**Important:** You need a Gmail App Password (NOT your regular Gmail password)

1. Go to [Google App Passwords](https://myaccount.google.com/apppasswords)
2. You may need to enable 2-Factor Authentication first if you haven't
3. Select:
   - **App:** Mail
   - **Device:** Other (Custom name)
4. Enter name: `BikeRisk Feedback`
5. Click **Generate**
6. Copy the 16-character password (it looks like: `abcd efgh ijkl mnop`)

### 2. Configure Local Environment

1. Create a `.env` file in your project root (if it doesn't exist)
2. Add these lines:

```bash
# Feedback Email Configuration
FEEDBACK_EMAIL_USER=your-gmail@gmail.com
FEEDBACK_EMAIL_PASSWORD=abcd efgh ijkl mnop
FEEDBACK_RECIPIENT_EMAIL=vanessajambois@gmail.com
```

**Replace:**
- `your-gmail@gmail.com` with the Gmail account you'll send FROM
- `abcd efgh ijkl mnop` with the App Password you generated
- The recipient email is already set to `vanessajambois@gmail.com`

### 3. Configure Railway (Production)

1. Go to your Railway dashboard
2. Select your BikeRisk project
3. Click **Variables** tab
4. Add these environment variables:
   - `FEEDBACK_EMAIL_USER`: your-gmail@gmail.com
   - `FEEDBACK_EMAIL_PASSWORD`: (your 16-char app password)
   - `FEEDBACK_RECIPIENT_EMAIL`: vanessajambois@gmail.com

### 4. Test It!

1. Restart your Streamlit app (locally or Railway)
2. Submit test feedback via the sidebar widget
3. Check `vanessajambois@gmail.com` for the email

## Email Format

You'll receive nicely formatted HTML emails with:
- **Subject:** BikeRisk Feedback: [Type] - [Page]
- **Content:**
  - Feedback type (Bug, Feature Request, etc.)
  - Page/section it relates to
  - User's email (if provided)
  - Rating (if applicable)
  - Full feedback message
  - Timestamp

## Troubleshooting

**Email not arriving?**
- ✅ Check spam folder
- ✅ Verify App Password (no spaces when pasting)
- ✅ Ensure 2FA is enabled on your Gmail account
- ✅ Check Railway environment variables are set correctly

**"Email not configured" message?**
- The app will still save feedback to `feedback.jsonl` locally
- Add the environment variables to enable email sending

## Security Notes

- ✅ `.env` file is in `.gitignore` (credentials never committed)
- ✅ Use App Passwords, not your main Gmail password
- ✅ Railway environment variables are encrypted
- ✅ Email credentials only used for sending feedback notifications

## Fallback

Even if email fails, feedback is always saved to `feedback.jsonl` locally, so you'll never lose user feedback!
