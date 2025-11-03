# 🔐 INTELLISEARCH API Key Authentication

## ✅ Implementation Complete!

Your INTELLISEARCH system now has **API Key Authentication** implemented with the following features:

### 🎯 **Security Features Implemented**

- ✅ **API Key Authentication** - All research endpoints require valid API keys
- ✅ **Rate Limiting** - 20 requests/hour, 5 requests/minute per API key
- ✅ **Session Ownership** - Users can only access their own research sessions
- ✅ **Admin User Support** - Special privileges for admin users
- ✅ **Automatic Key Validation** - Invalid keys are rejected immediately

### 🚀 **Quick Start Guide**

#### **1. Start the Backend**
```bash
cd web-app/backend
python main.py
```

#### **2. Start the Frontend**
```bash
cd web-app/frontend
npm run dev
```

#### **3. Access the Application**
- Open: http://localhost:3000
- You'll see an API key authentication screen

#### **4. Use Demo API Keys**
- **Regular User**: `demo-key-research-123`
- **Admin User**: `demo-key-admin-456`

### 🔧 **For Production Deployment**

#### **1. Generate Secure API Keys**
```bash
# Generate a secure API key
python -c "import secrets; print('intellisearch-' + secrets.token_urlsafe(32))"
```

#### **2. Set Environment Variables**
```bash
# Add to your .env file or Render environment variables

# Example 1: Simple user allocation
INTELLISEARCH_API_KEY_1=intellisearch-abc123def456:john_doe
INTELLISEARCH_API_KEY_2=intellisearch-xyz789uvw012:sarah_smith
INTELLISEARCH_API_KEY_3=intellisearch-admin999888:admin_user

# Example 2: Department-based allocation
INTELLISEARCH_API_KEY_1=intellisearch-research001:research_team
INTELLISEARCH_API_KEY_2=intellisearch-marketing002:marketing_dept
INTELLISEARCH_API_KEY_3=intellisearch-legal003:legal_counsel
INTELLISEARCH_API_KEY_4=intellisearch-admin004:system_admin

# Example 3: Email-based allocation
INTELLISEARCH_API_KEY_1=intellisearch-key001:john.doe@company.com
INTELLISEARCH_API_KEY_2=intellisearch-key002:sarah.smith@company.com
INTELLISEARCH_API_KEY_3=intellisearch-key003:admin@company.com

# Admin users (must match user_id after the colon)
ADMIN_USERS=admin_user,system_admin,admin@company.com

# Rate limiting
RATE_LIMIT_HOURLY=50
RATE_LIMIT_MINUTE=10
```

#### **3. Deploy to Render**
- Set environment variables in Render dashboard
- API keys are automatically loaded
- Rate limiting is automatically applied

### 🧪 **Test the Authentication**

Run the test suite to verify everything works:

```bash
cd web-app/backend
python test_auth.py
```

### 📋 **How It Works**

#### **Backend Changes:**
- `auth.py` - Complete authentication module
- `main.py` - All research endpoints now require API keys
- Session storage includes user ownership
- Built-in rate limiting per user

#### **Frontend Changes:**
- `AuthContext.tsx` - Manages authentication state
- `ApiKeyAuth.tsx` - API key input component
- `ResearchContext.tsx` - Uses authenticated requests
- `App.tsx` - Shows auth screen before main app

#### **Security Flow:**
1. User enters API key in frontend
2. Frontend validates key with backend `/api/auth/info`
3. Valid keys are stored and used for all requests
4. Backend validates every request and applies rate limits
5. Users can only access their own sessions

### 🎯 **API Endpoints**

- `GET /api/health` - No auth required
- `GET /api/auth/info` - Returns user info (requires API key)
- `POST /api/research/start` - Start research (requires API key)
- `GET /api/research/{id}/status` - Check status (requires API key + ownership)
- `GET /api/research/{id}/result` - Get results (requires API key + ownership)

### 💡 **Rate Limits**

- **Default**: 20 requests/hour, 5 requests/minute
- **Configurable** via environment variables
- **Per API key** (not per IP address)
- **Returns 429** status when exceeded

### 🔄 **User Experience**

1. **First Visit**: User sees authentication screen
2. **Enter API Key**: Demo keys provided for testing
3. **Authenticated**: Full research interface available
4. **Session Management**: Only user's sessions visible
5. **Rate Limiting**: Friendly error messages when limits hit

### 🛡️ **Security Best Practices Implemented**

- ✅ No hardcoded secrets in code
- ✅ Environment variable configuration
- ✅ Rate limiting to prevent abuse
- ✅ Session isolation between users
- ✅ Secure API key storage in localStorage
- ✅ Automatic session expiration handling
- ✅ Admin user privilege separation

### 🧪 **Testing Your Render Deployment**

#### **Test Your Live API Keys**
```bash
# Run the deployment test script
python test_render_deployment.py
```

This will:
- ✅ Test your Render backend health
- ✅ Validate your API keys work in production
- ✅ Check admin user privileges
- ✅ Test research endpoint access

#### **Quick Manual Test**
```bash
# Test health (no auth needed)
curl https://your-app.onrender.com/api/health

# Test API key authentication
curl -H "X-API-Key: your-actual-api-key" https://your-app.onrender.com/api/auth/info
```

### 👑 **Admin User Setup**

#### **Step 1: Configure Environment Variables in Render**

In your Render dashboard, set these environment variables:

```bash
# Your API keys (user_id comes after the colon)
INTELLISEARCH_API_KEY_1=your-secure-key-123:john_doe
INTELLISEARCH_API_KEY_2=your-secure-key-456:sarah_smith
INTELLISEARCH_API_KEY_3=your-secure-key-789:admin_user

# Make admin_user an admin (must match user_id exactly)
ADMIN_USERS=admin_user
```

#### **Step 2: Multiple Admins Example**
```bash
# Email-based user IDs
INTELLISEARCH_API_KEY_1=intellisearch-key001:john.doe@company.com
INTELLISEARCH_API_KEY_2=intellisearch-key002:admin@company.com
INTELLISEARCH_API_KEY_3=intellisearch-key003:ceo@company.com

# Multiple admins (comma-separated, no spaces)
ADMIN_USERS=admin@company.com,ceo@company.com
```

#### **Step 3: Verify Admin Status**
```bash
# Test admin privileges
curl -H "X-API-Key: your-admin-key" https://your-app.onrender.com/api/auth/info

# Should return: "is_admin": true
```

### 🔍 **Admin User Privileges**

- ✅ **Access All Sessions**: Can view any user's research sessions
- ✅ **Full Research Access**: All research endpoints available
- ✅ **Session Management**: Can check status/results for any session ID
- ✅ **Unlimited Rate Limits**: No rate limiting restrictions (bypass all limits)
- ✅ **Clear Identification**: `/api/auth/info` shows `"is_admin": true` and `"rate_limits": "unlimited"`

### ⚠️ **Important Admin Notes**

1. **Case Sensitive**: User IDs in `ADMIN_USERS` must exactly match API key user IDs
2. **No Spaces**: Use `user1,user2,user3` not `user1, user2, user3`
3. **Exact Match**: Admin list must match the part after `:` in API keys
4. **Redeploy Required**: Changes to environment variables require redeployment

### 🎉 **Ready for Production!**

Your INTELLISEARCH system is now secured and ready for production deployment. Users will need valid API keys to access the research functionality, and the system will automatically manage rate limiting and session security.

**Happy researching! 🔍✨**