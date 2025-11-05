#!/usr/bin/env python3
"""
INTELLISEARCH Render Deployment Test
Tests your live API keys against your Render deployment.
"""

import requests
import json
import time
import os

def test_render_deployment():
    """Test authentication against your Render deployment"""
    
    # Your Render URL - UPDATE THIS to your actual Render URL
    RENDER_URL = input("Enter your Render backend URL (e.g., https://your-app.onrender.com): ").strip()
    if not RENDER_URL:
        print("❌ No URL provided")
        return False
    
    # Remove trailing slash if present
    RENDER_URL = RENDER_URL.rstrip('/')
    
    print(f"\n🔐 Testing INTELLISEARCH Authentication on Render")
    print(f"🌐 Target URL: {RENDER_URL}")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{RENDER_URL}/api/health", timeout=10)
        if response.status_code == 200:
            print("✅ Backend is running on Render")
            health_data = response.json()
            print(f"   Status: {health_data.get('status', 'unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Render backend: {e}")
        return False
    
    # Test 2: Test your API keys
    print("\n2️⃣ Testing your API keys...")
    
    # Get API keys from user
    print("\n📝 Enter your API keys to test:")
    api_keys = []
    
    while True:
        key = input(f"API Key #{len(api_keys)+1} (or press Enter to finish): ").strip()
        if not key:
            break
        api_keys.append(key)
    
    if not api_keys:
        print("❌ No API keys provided")
        return False
    
    # Test each API key
    for i, api_key in enumerate(api_keys, 1):
        print(f"\n🔑 Testing API Key #{i}: {api_key[:20]}...")
        
        try:
            headers = {"X-API-Key": api_key}
            response = requests.get(f"{RENDER_URL}/api/auth/info", headers=headers, timeout=10)
            
            if response.status_code == 200:
                auth_data = response.json()
                user_info = auth_data.get('user', {})
                
                print(f"✅ API Key #{i} is VALID")
                print(f"   User ID: {user_info.get('user_id', 'unknown')}")
                print(f"   Is Admin: {'YES' if user_info.get('is_admin', False) else 'NO'}")
                print(f"   Rate Limits: {user_info.get('rate_limits', {})}")
                
                # Test research endpoint
                print(f"   Testing research access...")
                test_request = {
                    "query": "Test query for API validation",
                    "report_type": "concise",
                    "prompt_type": "general",
                    "automation_level": "full"
                }
                
                research_response = requests.post(
                    f"{RENDER_URL}/api/research/start",
                    headers={**headers, "Content-Type": "application/json"},
                    json=test_request,
                    timeout=15
                )
                
                if research_response.status_code == 200:
                    result = research_response.json()
                    session_id = result.get('data', {}).get('session_id')
                    print(f"   ✅ Research endpoint accessible - Session: {session_id}")
                else:
                    print(f"   ❌ Research endpoint failed: {research_response.status_code}")
                    print(f"   Error: {research_response.text}")
                    
            elif response.status_code == 401:
                print(f"❌ API Key #{i} is INVALID")
                error_detail = response.json().get('detail', 'No detail provided')
                print(f"   Error: {error_detail}")
            else:
                print(f"❌ API Key #{i} test failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ API Key #{i} test failed: {e}")
    
    # Test 3: Frontend access
    print("\n3️⃣ Testing frontend access...")
    frontend_url = RENDER_URL.replace('/api', '').replace(':8000', ':3000')
    print(f"   Frontend should be accessible at: {frontend_url}")
    print(f"   Visit this URL and test your API keys in the browser")
    
    print("\n" + "=" * 60)
    print("🎯 Render Deployment Test Complete!")
    print("\n📋 Next Steps:")
    print("1. Visit your frontend URL to test in browser")
    print("2. Use your validated API keys to access the system")
    print("3. Admin users can access all sessions")
    
    return True

def show_admin_setup_guide():
    """Show how to set up admin users"""
    
    print("\n" + "=" * 60)
    print("👑 ADMIN USER SETUP GUIDE")
    print("=" * 60)
    
    print("\n🎯 To make a user an admin, you need to:")
    print("1. Add their user_id to the ADMIN_USERS environment variable")
    print("2. The user_id comes from their API key configuration")
    
    print("\n📝 Environment Variable Format:")
    print("# In your Render environment variables:")
    print("ADMIN_USERS=user1,user2,user3")
    
    print("\n🔑 Example API Key Setup:")
    print("# Regular users")
    print("INTELLISEARCH_API_KEY_1=your-secure-key-123:john_doe")
    print("INTELLISEARCH_API_KEY_2=your-secure-key-456:sarah_smith")
    print("# Admin user - user_id after the colon")
    print("INTELLISEARCH_API_KEY_3=your-secure-key-789:admin_user")
    print("")
    print("# Then set admin users (must match user_id)")
    print("ADMIN_USERS=admin_user")
    
    print("\n🏢 Real-world Example:")
    print("# Company setup")
    print("INTELLISEARCH_API_KEY_1=intellisearch-key001:john.doe@company.com")
    print("INTELLISEARCH_API_KEY_2=intellisearch-key002:sarah.smith@company.com") 
    print("INTELLISEARCH_API_KEY_3=intellisearch-key003:admin@company.com")
    print("INTELLISEARCH_API_KEY_4=intellisearch-key004:ceo@company.com")
    print("")
    print("# Make admin@company.com and ceo@company.com admins")
    print("ADMIN_USERS=admin@company.com,ceo@company.com")
    
    print("\n🔧 Steps in Render Dashboard:")
    print("1. Go to your service dashboard")
    print("2. Click 'Environment' tab")
    print("3. Add/Edit environment variables:")
    print("   - Your API keys (INTELLISEARCH_API_KEY_X)")
    print("   - Admin users list (ADMIN_USERS)")
    print("4. Save and redeploy")
    
    print("\n👑 Admin User Privileges:")
    print("✅ Can access ALL research sessions (not just their own)")
    print("✅ Can view any user's research results")
    print("✅ Same rate limits as regular users")
    print("✅ Identified as admin in /api/auth/info response")
    
    print("\n⚠️  Security Notes:")
    print("• Admin user_ids are case-sensitive")
    print("• Must exactly match the user_id in API key")
    print("• Comma-separated list for multiple admins")
    print("• No spaces around commas in ADMIN_USERS")

if __name__ == "__main__":
    print("🚀 INTELLISEARCH Render Testing & Admin Setup")
    print("\nChoose an option:")
    print("1. Test my Render deployment")
    print("2. Show admin setup guide")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice in ["1", "3"]:
        test_render_deployment()
    
    if choice in ["2", "3"]:
        show_admin_setup_guide()
    
    print("\n🎉 Done! Your INTELLISEARCH system is ready for production use!")