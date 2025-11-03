# INTELLISEARCH Access Control Strategy

## 🔒 Current State Analysis

**Current Security Status: OPEN ACCESS**
- No authentication required
- No user management system
- All endpoints publicly accessible
- Session management based on UUIDs only
- API keys stored in environment variables (good)

## 🎯 Recommended Access Control Approaches

### **Option 1: Simple API Key Authentication (Quick Implementation)**

**Best for**: Small teams, internal use, proof of concept

#### Implementation:
```python
# web-app/backend/auth.py
from fastapi import HTTPException, Depends, Header
from typing import Optional
import os

VALID_API_KEYS = {
    os.getenv("CLIENT_API_KEY_1", "demo-key-123"): "client1",
    os.getenv("CLIENT_API_KEY_2", "demo-key-456"): "client2",
    # Add more as needed
}

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401, 
            detail="Invalid or missing API key"
        )
    return VALID_API_KEYS[x_api_key]

# Usage in endpoints:
@app.post("/api/research/start")
async def start_research(
    request: ResearchRequest, 
    user_id: str = Depends(verify_api_key)
):
    # Now you know which user made the request
    session["user_id"] = user_id
    # ... rest of implementation
```

#### Frontend Changes:
```typescript
// Add API key to requests
const apiKey = localStorage.getItem('intellisearch_api_key');
const response = await fetch('/api/research/start', {
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': apiKey
  },
  body: JSON.stringify(request)
});
```

---

### **Option 2: JWT Token Authentication (Professional)**

**Best for**: Multi-user applications, commercial use

#### Backend Implementation:
```python
# web-app/backend/auth.py
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
JWT_ALGORITHM = "HS256"

def create_access_token(user_id: str, expires_delta: timedelta = None):
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    to_encode = {"sub": user_id, "exp": expire}
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Login endpoint
@app.post("/api/auth/login")
async def login(username: str, password: str):
    # Verify credentials (implement your logic)
    if verify_credentials(username, password):
        token = create_access_token(user_id=username)
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")
```

---

### **Option 3: User Registration System (Full-Featured)**

**Best for**: Public applications, SaaS deployment

#### Database Schema:
```sql
-- Add to your database (PostgreSQL/SQLite)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    api_quota_daily INTEGER DEFAULT 10,
    api_quota_used INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE research_sessions (
    id UUID PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    query TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    report_content TEXT,
    word_count INTEGER
);
```

#### Implementation:
```python
# web-app/backend/models.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from passlib.context import CryptContext

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    api_quota_daily = Column(Integer, default=10)
    api_quota_used = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    def verify_password(self, password: str):
        return pwd_context.verify(password, self.password_hash)
    
    @staticmethod
    def hash_password(password: str):
        return pwd_context.hash(password)
```

---

## 🚀 Implementation Recommendations by Use Case

### **For Internal/Team Use (Recommended: Option 1)**
- Simple API key authentication
- Shared keys per team/department
- Quick to implement
- No user registration needed

### **For Client Services (Recommended: Option 2)**
- JWT token authentication
- Time-limited sessions
- Professional security
- User management

### **For Public SaaS (Recommended: Option 3)**
- Full user registration
- Usage quotas and billing
- Email verification
- Password reset functionality

---

## 🔧 Enhanced Session Security

### **Current Session Management Enhancement:**
```python
# web-app/backend/main.py - Enhanced session security
import secrets
from datetime import datetime, timedelta

class SecureSession:
    def __init__(self, user_id: str, session_id: str = None):
        self.session_id = session_id or secrets.token_urlsafe(32)
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.expires_at = datetime.now() + timedelta(hours=24)
    
    def is_valid(self):
        return datetime.now() < self.expires_at
    
    def refresh(self):
        self.last_accessed = datetime.now()
        self.expires_at = datetime.now() + timedelta(hours=24)

# Enhanced research sessions with user ownership
secure_sessions: Dict[str, SecureSession] = {}

async def get_session(session_id: str, user_id: str):
    session = secure_sessions.get(session_id)
    if not session or not session.is_valid() or session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Invalid or expired session")
    session.refresh()
    return session
```

---

## 🛡️ Rate Limiting and Quotas

### **User-Based Rate Limiting:**
```python
# web-app/backend/rate_limiting.py
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
    
    async def check_rate_limit(self, user_id: str, limit: int = 5, window: int = 3600):
        now = datetime.now()
        # Clean old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id] 
            if now - req_time < timedelta(seconds=window)
        ]
        
        if len(self.requests[user_id]) >= limit:
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded. Max {limit} requests per hour."
            )
        
        self.requests[user_id].append(now)

rate_limiter = RateLimiter()

# Usage in endpoints
@app.post("/api/research/start")
async def start_research(
    request: ResearchRequest,
    user_id: str = Depends(verify_api_key)
):
    await rate_limiter.check_rate_limit(user_id)
    # ... rest of implementation
```

---

## 🎮 Frontend Authentication Components

### **Login Component:**
```typescript
// web-app/frontend/src/components/Login.tsx
import React, { useState } from 'react';

interface LoginProps {
  onLogin: (token: string) => void;
}

export const Login: React.FC<LoginProps> = ({ onLogin }) => {
  const [credentials, setCredentials] = useState({
    username: '',
    password: ''
  });

  const handleLogin = async () => {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials)
    });
    
    if (response.ok) {
      const { access_token } = await response.json();
      localStorage.setItem('auth_token', access_token);
      onLogin(access_token);
    }
  };

  return (
    <div className="login-form">
      <input 
        type="text" 
        placeholder="Username"
        value={credentials.username}
        onChange={(e) => setCredentials({...credentials, username: e.target.value})}
      />
      <input 
        type="password" 
        placeholder="Password"
        value={credentials.password}
        onChange={(e) => setCredentials({...credentials, password: e.target.value})}
      />
      <button onClick={handleLogin}>Login</button>
    </div>
  );
};
```

---

## 🔗 Integration with Existing System

### **Minimal Changes Required:**

1. **Add authentication middleware to `main.py`**
2. **Enhance session storage with user ownership**
3. **Add rate limiting per user**
4. **Update frontend to handle authentication**

### **Recommended Starting Point:**
```python
# web-app/backend/main.py - Add this to your existing code

# Add at the top
from functools import wraps

def require_auth(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Add your authentication logic here
        return await func(*args, **kwargs)
    return wrapper

# Apply to sensitive endpoints
@app.post("/api/research/start")
@require_auth
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    # Your existing implementation
    pass
```

---

## 🎯 Next Steps Recommendation

**For immediate deployment**: Start with **Option 1 (API Key)** - it's quick to implement and provides basic security.

**For production**: Plan migration to **Option 2 (JWT)** or **Option 3 (Full User System)** based on your user base.

Would you like me to implement any of these options in your codebase?