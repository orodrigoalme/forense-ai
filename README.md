# Forense AI - Forensic Analysis API for AI-Generated Images

🌐 Access web page at: forenseai.orodrigoalme.com

## 📋 What is this application?

**Forense AI** is a FastAPI-based REST API that performs forensic analysis on images to detect if they were generated or manipulated by Artificial Intelligence. The application combines multiple digital forensic techniques with generative AI (Google Gemini) to provide a consolidated verdict on image authenticity.

### 🆕 Version 2.0 - What's New

- ✅ **Anonymous Authentication** - Use without registration via JWT tokens
- ✅ **Dynamic Limits** - Increase limits using your own Gemini key
- ✅ **Anti-Abuse Protection** - Smart rate limiting per IP/session
- ✅ **Quota System** - Usage control per API key and anonymous sessions
- ✅ **Budget Caps** - Automatic Gemini cost protection
- ✅ **Flexible Auth** - API Key OR Anonymous Token

---

## 🎯 Main Features

### 1. **Noise Analysis (NOISE)**
Examines the natural noise pattern produced by camera sensors. AI-generated images tend to have:
- Abnormally low or perfectly consistent noise
- "Overly smooth" regions (skin, sky, backgrounds)
- Absence of natural sensor noise patterns

### 2. **Fourier Transform Analysis (FFT)**
Analyzes the image's frequency spectrum to detect:
- Excessive symmetry in the spectrum (AI generates near-perfect symmetrical patterns)
- Periodic anomalous peaks (grid artifacts, checkerboard patterns)
- Unnatural spectral uniformity
- High-frequency grid patterns (upscaling artifacts)

### 3. **Error Level Analysis (ELA)**
A technique that recompresses the JPEG image and analyzes differences to detect:
- Regions with inconsistent error levels (selective manipulation)
- Areas with abnormally low error (AI insertions)
- Edges with inconsistent error (splicing, copy-move)
- Uniform error patterns (full AI generation)

### 4. **Analysis with Gemini AI**
Integrates the Google Gemini API for advanced contextual analysis:
- Interprets results from technical analyses
- Provides explanations in accessible language for non-technical users
- Generates a final verdict with a confidence level
- Identifies key indicators in a simple format

### 5. **Annotated Images**
Generates annotated visualizations that highlight:
- Suspicious areas identified by each method
- Anomaly heatmaps
- Risk score per region

---

## 🔧 Technical Architecture

### Technologies Used
- **Framework:** FastAPI 0.109.0
- **Image Processing:** OpenCV, NumPy, Pillow
- **Scientific Analysis:** SciPy
- **Generative AI:** Google Gemini (google-genai 0.3.0)
- **Authentication:** JWT (PyJWT)
- **Rate Limiting:** SlowAPI
- **Server:** Uvicorn

### Project Structure
```
forense-ai/
├── app/
│   ├── main.py                       # API Endpoints
│   ├── services/
│   │   ├── analysis_service.py       # Analysis Orchestration + Gemini
│   │   └── image_annotator.py        # Annotated Image Generation
│   ├── analyzers/
│   │   ├── noise.py                  # Noise Analysis
│   │   ├── fft.py                    # FFT Analysis
│   │   └── ela.py                    # Error Level Analysis
│   ├── middleware/
│   │   ├── anonymous_auth.py         # Anonymous JWT System
│   │   ├── auth.py                   # API Key Authentication
│   │   ├── rate_limiter.py           # Dynamic Rate Limiting
│   │   ├── quota.py                  # Quota System
│   │   ├── cost_tracker.py           # Gemini Cost Tracking
│   │   └── captcha.py                # reCAPTCHA Verification (optional)
│   └── utils.py                      # Validation and Utilities
├── uploads/                          # Temporary directory for uploads
├── cost_tracking.json                # Cost registration (auto-generated)
├── .env                              # Environment variables
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker configuration
└── README.md                         # This documentation
```

---

## 🔐 Authentication

API v2.0 offers **2 flexible authentication modes**:

### Option 1: API Key (Recommended for Integration)

**Advantages:**
- ✅ No anonymous session limitations
- ✅ Personalized quotas per client
- ✅ Higher rate limits
- ✅ Ideal for production applications

**How to use:**
```bash
curl -X POST "http://localhost:8001/api/analyze-image" \
  -H "X-API-Key: aidet_demo_hackathon_2026" \
  -F "file=@image.jpg"
```

**Demo API Key (for testing):**
- **Key:** `aidet_demo_hackathon_2026`
- **Rate Limit:** 20 req/min, 200 req/hour
- **Quota:** Unlimited

### Option 2: Anonymous Token (Public Access)

**Advantages:**
- ✅ No registration required
- ✅ Immediate access
- ✅ Ideal for tests and public demos

**Limitations (without own Gemini key):**
- 📊 50 requests per session
- 📊 5,000 quota credits
- 📊 3 requests/minute

**Limitations (WITH own Gemini key):**
- 📊 200 requests per session (4x more!)
- 📊 Unlimited quota
- 📊 20 requests/minute (6x more!)

**Usage Workflow:**

```bash
# 1. Get tokens (access + refresh)
TOKEN_DATA=$(curl -X POST "http://localhost:8001/api/auth/anonymous")
ACCESS_TOKEN=$(echo $TOKEN_DATA | jq -r .access_token)
REFRESH_TOKEN=$(echo $TOKEN_DATA | jq -r .refresh_token)

# 2. Use access token (valid for 1h)
curl -X POST "http://localhost:8001/api/analyze-image" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "file=@image.jpg"

# 3. Renew when expired (after 1h)
NEW_TOKENS=$(curl -X POST "http://localhost:8001/api/auth/refresh" \
  -H "X-Refresh-Token: $REFRESH_TOKEN")
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- `GEMINI_API_KEY` environment variable (for server-side Gemini analysis)
- Configured `.env` file

### Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd forense-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file (see section below)
```

### `.env` Configuration

Create a `.env` file in the project root:

```env
# ========================================
# AUTHENTICATION
# ========================================

# Valid API Keys (comma-separated)
API_KEYS=aidet_demo_hackathon_2026,aidet_prod_key_xyz123

# Premium keys (higher quotas)
PREMIUM_API_KEYS=aidet_prod_key_xyz123

# JWT Secret (generate with: openssl rand -hex 32)
JWT_SECRET=your_very_long_and_random_secrey_key_here

# Anonymous token lifetime
ACCESS_TOKEN_LIFETIME_MINUTES=60
SESSION_LIFETIME_DAYS=7

# ========================================
# QUOTAS AND LIMITS
# ========================================

# Daily quotas per tier
FREE_TIER_DAILY_LIMIT=10
PREMIUM_TIER_DAILY_LIMIT=100

# Anonymous session limits WITHOUT own Gemini key
ANON_REQUESTS_LIMIT=50
ANON_QUOTA_LIMIT=5000

# Anonymous session limits WITH own Gemini key
ANON_REQUESTS_LIMIT_CUSTOM_KEY=200
ANON_QUOTA_LIMIT_CUSTOM_KEY=0  # 0 = unlimited

# ========================================
# RATE LIMITING
# ========================================

# Full Analysis (with Gemini)
RATE_LIMIT_ANALYZE_SERVER_KEY=3/minute
RATE_LIMIT_ANALYZE_CUSTOM_KEY=20/minute

# Individual Analyses (FFT, NOISE, ELA)
RATE_LIMIT_INDIVIDUAL_SERVER_KEY=10/minute
RATE_LIMIT_INDIVIDUAL_CUSTOM_KEY=30/minute

# ========================================
# ANTI-ABUSE PROTECTION
# ========================================

# Session creation limits per IP
MAX_SESSIONS_PER_IP_HOUR=3
MAX_SESSIONS_PER_IP_DAY=10
MAX_ACTIVE_SESSIONS_PER_IP=5

# ========================================
# GOOGLE GEMINI
# ========================================

# SERVER Gemini API Key (optional)
GEMINI_API_KEY=your_gemini_key_here

# Budget caps (cost protection)
MAX_DAILY_GEMINI_COST=5.0
MAX_MONTHLY_GEMINI_COST=50.0

# ========================================
# reCAPTCHA (Optional)
# ========================================

# Enforcement: "required", "optional", "disabled"
CAPTCHA_ENFORCEMENT=optional
RECAPTCHA_SECRET_KEY=
RECAPTCHA_MIN_SCORE=0.5
```

### Local Execution

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

The API will be available at: `http://localhost:8001`

Interactive documentation (Swagger): `http://localhost:8001/docs`

### Docker

```bash
# Build
docker build -t forense-ai .

# Run
docker run -p 8001:8001 --env-file .env forense-ai
```

---

## 🌐 API Endpoints

### 🔐 Authentication

#### **POST /api/auth/anonymous**
Generates JWT tokens for anonymous access (no registration required).

**Request:**
```bash
curl -X POST "http://localhost:8001/api/auth/anonymous"
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "access_expires_in": 3600,
  "refresh_expires_in": 604800,
  "access_expires_at": "2026-02-09T14:27:00",
  "refresh_expires_at": "2026-02-16T13:27:00",
  "session_id": "anon_a1b2c3d4e5f6",
  "limits": {
    "default": {
      "requests_limit": 50,
      "quota_limit": 5000,
      "grad_description": "Limits when using server Gemini key"
    },
    "custom_key": {
      "requests_limit": 200,
      "quota_limit": "unlimited",
      "grad_description": "Limits when using your own Gemini key (X-Gemini-Key)"
    },
    "current_usage": {
      "requests_used": 0,
      "quota_used": 0
    }
  }
}
```

---

#### **POST /api/auth/refresh**
Refreshes the access token using the refresh token.

**Request:**
```bash
curl -X POST "http://localhost:8001/api/auth/refresh" \
  -H "X-Refresh-Token: eyJhbGciOiJIUzI1NiIs..."
```

**Response (200):**
```json
{
  "access_token": "new_access_token_here",
  "refresh_token": "new_refresh_token_here",
  "token_type": "Bearer",
  "access_expires_in": 3600,
  ...
}
```

**Errors:**
- `401 Unauthorized` - Refresh token expired or invalid
- `401 Unauthorized` - Session not found

---

#### **GET /api/auth/session**
Check current anonymous session statistics.

**Request:**
```bash
curl -X GET "http://localhost:8001/api/auth/session" \
  -H "Authorization: Bearer <access_token>"
```

**Response (200):**
```json
{
  "session_id": "anon_a1b2c3d4e5f6",
  "type": "anonymous",
  "stats": {
    "requests_used": 12,
    "requests_remaining": 38,
    "requests_limit": 50,
    "quota_used": 1200,
    "quota_remaining": 3800,
    "quota_limit": 5000,
    "created_at": "2026-02-09T10:00:00",
    "session_age_hours": 3.45,
    "limit_type": "server_key",
    "tip": "Use X-Gemini-Key header with your own key for higher limits"
  }
}
```

---

#### **DELETE /api/auth/session**
Ends the current anonymous session.

**Request:**
```bash
curl -X DELETE "http://localhost:8001/api/auth/session" \
  -H "Authorization: Bearer <access_token>"
```

**Response (200):**
```json
{
  "message": "Session ended successfully",
  "session_id": "anon_a1b2c3d4e5f6",
  "stats": {
    "requests_used": 15,
    "quota_used": 1500
  }
}
```

---

### 🔍 Image Analysis

#### **POST /api/analyze-image** ⭐ (Primary Endpoint)
Executes consolidated FULL analysis (FFT + NOISE + ELA + Gemini).

**Authentication (choose ONE):**

**Option 1 - API Key:**
```bash
curl -X POST "http://localhost:8001/api/analyze-image" \
  -H "X-API-Key: aidet_demo_hackathon_2026" \
  -F "file=@image.jpg"
```

**Option 2 - Anonymous Token:**
```bash
curl -X POST "http://localhost:8001/api/analyze-image" \
  -H "Authorization: Bearer <access_token>" \
  -F "file=@image.jpg"
```

**Optional Headers:**
- `X-Gemini-Key` - Your Gemini key (increases limits and uses your credits)
- `X-Captcha-Token` - reCAPTCHA token (if enabled)

**Response (200):**
```json
{
  "automated_analysis": {
    "final_score": 0.72,
    "interpretation": "Probably AI",
    "confidence": "high",
    "methods_used": ["FFT", "NOISE", "ELA"],
    "individual_scores": {
      "fft": 0.42,
      "noise": 0.85,
      "ela": 0.68
    },
    "key_evidence": [
      "NOISE: Synthetic noise detected (consistency=0.85)",
      "ELA: Excessive uniformity (mean_error=0.012)"
    ],
    "recommendation": "⚠️ MANUAL ANALYSIS - Ambiguous evidence"
  },
  "gemini_analysis": {
    "verdict": "AI",
    "full_analysis": "Full text of Gemini analysis...",
    "explanation": "This image presents typical characteristics...",
    "confidence": "high",
    "key_indicators": [
      "Uniform noise pattern typical of generators",
      "Absence of natural JPEG compression artifacts"
    ]
  },
  "annotated_image": "base64_encoded_annotated_image",
  "details": {
    "fft": { /* Full FFT results */ },
    "noise": { /* Full NOISE results */ },
    "ela": { /* Full ELA results */ }
  },
  "session_usage": {
    "requests_used": 13,
    "requests_remaining": 37,
    "quota_used": 1300,
    "quota_remaining": 3700,
    "limit_type": "server_key"
  }
}
```

**Rate Limits:**
- **Demo API Key:** 20 req/min
- **Anonymous token (without Gemini key):** 3 req/min
- **Anonymous token (with Gemini key):** 20 req/min

**Errors:**
- `401 Unauthorized` - Invalid or missing Token/API key
- `429 Too Many Requests` - Rate limit or quota exceeded
- `400 Bad Request` - Invalid file
- `500 Internal Server Error` - Analysis error

---

## 📊 Limits Comparison Table

| Feature | Demo API Key | Anonymous Token (Server) | Anonymous Token (Own Key) |
|---|---|---|---|
| **Authentication** | `X-API-Key: aidet_demo_...` | `Authorization: Bearer ...` | `Authorization: Bearer ...` + `X-Gemini-Key` |
| **Requests/session** | Unlimited | 50 | 200 |
| **Credit Quota** | Unlimited | 5,000 | Unlimited |
| **Rate Limit** | 20 req/min | 3 req/min | 20 req/min |
| **Session Duration** | Permanent | 7 days | 7 days |
| **Gemini Cost** | Server | Server | Client |
| **Budget cap** | N/A | $5/day, $50/month | N/A |
| **Ideal for** | Prod Integration | Quick tests | Intense usage |

---

## 🛡️ Anti-Abuse Protection

### IP Limits (Anonymous Sessions)

**PerHour:**
- Maximum 3 new sessions created per IP

**PerDay:**
- Maximum 10 new sessions created per IP
- Maximum 5 simultaneous active sessions per IP

**Active Sessions:**
- Sessions are automatically cleared after 7 days
- Use `DELETE /api/auth/session` to end manually

**Bypass:**
- Use API Key to avoid session creation limits

---

## 💰 Cost and Quota System

### Budget Caps (Server Gemini Key)

Automatic cost protection when clients use the server's Gemini key:

- **Daily limit:** $5.00 USD
- **Monthly limit:** $50.00 USD
- **Cost per request:** ~$0.002 USD

**Tracking file:** `cost_tracking.json` (auto-generated)

**Automatic cleanup:**
- Keeps the last 7 days of daily data
- Keeps the last 3 months of monthly data

**Bypass:**
- Use `X-Gemini-Key` with your own key to avoid server budget cap

---

## 🧪 Use Cases

### 1. Quick Test (No Registration)

```bash
# 1. Get token
TOKEN=$(curl -s -X POST "http://localhost:8001/api/auth/anonymous" | jq -r .access_token)

# 2. Analyze image
curl -X POST "http://localhost:8001/api/analyze-image" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@suspicious_photo.jpg" \
  | jq .automated_analysis.interpretation
```

### 2. Production Integration

```python
import requests

API_KEY = "aidet_demo_hackathon_2026"
API_URL = "http://localhost:8001/api/analyze-image"

def analyze_image(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(
            API_URL,
            headers={"X-API-Key": API_KEY},
            files={"file": f}
        )
    
    if response.status_code == 200:
        result = response.json()
        return result["automated_analysis"]["interpretation"]
    else:
        raise Exception(f"Error: {response.status_code}")

# Usage
verdict = analyze_image("image.jpg")
print(f"Verdict: {verdict}")
```

### 3. Using Own Gemini Key (Higher Limits)

```bash
# Get anonymous token
TOKEN=$(curl -s -X POST "http://localhost:8001/api/auth/anonymous" | jq -r .access_token)

# Analyze with your own Gemini key (200 req/session instead of 50!)
curl -X POST "http://localhost:8001/api/analyze-image" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Gemini-Key: YOUR_GEMINI_KEY_HERE" \
  -F "file=@image.jpg"
```

---

## 📊 Interpreting Results

### Risk Score (0.0 - 1.0)
- **0.00 - 0.15:** Very likely REAL
- **0.15 - 0.35:** Probably REAL
- **0.35 - 0.55:** INCONCLUSIVE - Manual analysis recommended
- **0.55 - 0.75:** Probably AI
- **0.75 - 1.00:** Very likely AI

### Confidence Levels
- **very_high:** All 3 methods agree + score far from grey area
- **high:** All methods analyzed + consistent results
- **medium:** Some methods failed or partial conflicting results
- **low:** Only 1-2 methods worked
- **very_low:** Analysis compromised or insufficient data

### Gemini Verdict
- **REAL:** Authentic image, captured by camera
- **AI:** Image generated or manipulated by AI
- **INCONCLUSIVE:** Conflicting or insufficient evidence
- **DISABLED:** Gemini not configured
- **ERROR:** Error in Gemini analysis

---

## ⚠️ Limitations

1. **Gemini Disabled without API Key**
   - If `GEMINI_API_KEY` is not configured AND the client doesn't send `X-Gemini-Key`, the `gemini_analysis.verdict` field will be `"DISABLED"`.

2. **Image Formats**
   - ELA works best with JPEG images (PNG images are temporarily converted).

3. **Heavily Compressed Images**
   - Heavy compression can generate false positives in all methods.

4. **Screenshots and Legitimate Edits**
   - Screenshots and basic edits (crop, resize) may be flagged as suspicious.

5. **Budget Caps**
   - When using the server's Gemini key, there are limits of $5/day and $50/month.
   - Use your own key (`X-Gemini-Key`) to avoid these limits.

---

## 🔧 Advanced Configuration

### Full Environment Variables

```env
# Authentication
API_KEYS=key1,key2,key3
PREMIUM_API_KEYS=key2
JWT_SECRET=generate_with_openssl_rand_hex_32
ACCESS_TOKEN_LIFETIME_MINUTES=60
SESSION_LIFETIME_DAYS=7

# Quotas
FREE_TIER_DAILY_LIMIT=10
PREMIUM_TIER_DAILY_LIMIT=100
ANON_REQUESTS_LIMIT=50
ANON_QUOTA_LIMIT=5000
ANON_REQUESTS_LIMIT_CUSTOM_KEY=200
ANON_QUOTA_LIMIT_CUSTOM_KEY=0

# Rate Limiting
RATE_LIMIT_ANALYZE_SERVER_KEY=3/minute
RATE_LIMIT_ANALYZE_CUSTOM_KEY=20/minute
RATE_LIMIT_INDIVIDUAL_SERVER_KEY=10/minute
RATE_LIMIT_INDIVIDUAL_CUSTOM_KEY=30/minute

# Anti-Abuse
MAX_SESSIONS_PER_IP_HOUR=3
MAX_SESSIONS_PER_IP_DAY=10
MAX_ACTIVE_SESSIONS_PER_IP=5

# Gemini
GEMINI_API_KEY=your_key_here
MAX_DAILY_GEMINI_COST=5.0
MAX_MONTHLY_GEMINI_COST=50.0

# reCAPTCHA (Optional)
CAPTCHA_ENFORCEMENT=optional  # required, optional, disabled
RECAPTCHA_SECRET_KEY=
RECAPTCHA_MIN_SCORE=0.5
```

---

## 🐛 Troubleshooting

### Error: "Session limit reached"

**Cause:** IP created too many sessions in a short time (anti-abuse protection).

**Solutions:**
1. Wait 1 hour (automatic reset)
2. Use demo API Key (`X-API-Key: aidet_demo_hackathon_2026`)
3. End old sessions: `DELETE /api/auth/session`

### Error: "Budget cap reached"

**Cause:** Server Gemini cost limits exceeded ($5/day or $50/month).

**Solutions:**
1. Use your own Gemini key: `-H "X-Gemini-Key: YOUR_KEY"`
2. Wait for reset (Midnight UTC for daily)
3. Increase limits in `.env` if you manage the server

### Error: "Token expired"

**Cause:** 1h access token has expired.

**Solution:**
```bash
# Renew with refresh token
curl -X POST "/api/auth/refresh" \
  -H "X-Refresh-Token: YOUR_REFRESH_TOKEN"
```

### Error: "Session not found"

**Cause:** Session expired (7 days) or was terminated.

**Solution:**
```bash
# Create new session
curl -X POST "/api/auth/anonymous"
```

---

## 📝 Changelog

### v2.0 (2026-02-09)
- ✅ Anonymous authentication system with JWT
- ✅ Dynamic limits based on own Gemini key
- ✅ Anti-abuse protection per IP
- ✅ Quota system and cost tracking
- ✅ Smart rate limiting
- ✅ reCAPTCHA support (optional)

### v1.0 (2025-12-01)
- ✅ Forensic analyses: FFT, NOISE, ELA
- ✅ Gemini AI integration
- ✅ Annotated image generation
- ✅ Basic API with FastAPI

---

## 📧 Contact and Support

- **Interactive Documentation:** `/docs` (Swagger UI)
- **Health Check:** `/health`
- **Repository:** [GitHub](https://github.com/your-repo)

---

## 📄 License

This project is provided as-is, without warranties. Use at your own risk.

---

## 🤝 Contributions

Contributions are welcome! Areas for improvement:
- New analysis methods (DWT, CFA, Metadata Analysis)
- Improvements in detection thresholds
- Support for videos and GIFs
- Web interface for upload
- Analysis caching system
- Redis integration for distributed quotas
