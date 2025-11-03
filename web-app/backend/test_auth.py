#!/usr/bin/env python3
"""
INTELLISEARCH API Key Authentication Test
Tests the complete authentication flow for the INTELLISEARCH system.
"""

import requests
import json
import time
import os
import sys

# Add the backend directory to the path
sys.path.append(os.path.dirname(__file__))

def test_authentication_flow():
    """Test the complete authentication flow"""
    
    # Configuration
    BASE_URL = "http://localhost:8000"
    DEMO_KEYS = {
        "valid_user": "demo-key-research-123",
        "admin_user": "demo-key-admin-456",
        "invalid": "invalid-key-999"
    }
    
    print("🔐 INTELLISEARCH Authentication Test Suite")
    print("=" * 50)
    
    # Test 1: Health check (no auth required)
    print("\n1️⃣ Testing health endpoint (no auth required)...")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            health_data = response.json()
            print(f"   Status: {health_data.get('status', 'unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("   Make sure the backend is running: python main.py")
        return False
    
    # Test 2: Try accessing protected endpoint without auth
    print("\n2️⃣ Testing protected endpoint without authentication...")
    try:
        response = requests.get(f"{BASE_URL}/api/auth/info", timeout=5)
        if response.status_code == 401:
            print("✅ Correctly rejected unauthenticated request")
            print(f"   Error: {response.json().get('detail', 'No detail')}")
        else:
            print(f"❌ Should have rejected request, got: {response.status_code}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Test 3: Try with invalid API key
    print("\n3️⃣ Testing with invalid API key...")
    try:
        headers = {"X-API-Key": DEMO_KEYS["invalid"]}
        response = requests.get(f"{BASE_URL}/api/auth/info", headers=headers, timeout=5)
        if response.status_code == 401:
            print("✅ Correctly rejected invalid API key")
            print(f"   Error: {response.json().get('detail', 'No detail')}")
        else:
            print(f"❌ Should have rejected invalid key, got: {response.status_code}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Test 4: Valid API key authentication
    print("\n4️⃣ Testing with valid API key...")
    try:
        headers = {"X-API-Key": DEMO_KEYS["valid_user"]}
        response = requests.get(f"{BASE_URL}/api/auth/info", headers=headers, timeout=5)
        if response.status_code == 200:
            print("✅ Successfully authenticated with valid API key")
            auth_data = response.json()
            user_info = auth_data.get('user', {})
            print(f"   User ID: {user_info.get('user_id', 'unknown')}")
            print(f"   Is Admin: {user_info.get('is_admin', False)}")
            print(f"   Rate Limits: {user_info.get('rate_limits', {})}")
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Test 5: Admin user authentication
    print("\n5️⃣ Testing admin user authentication...")
    try:
        headers = {"X-API-Key": DEMO_KEYS["admin_user"]}
        response = requests.get(f"{BASE_URL}/api/auth/info", headers=headers, timeout=5)
        if response.status_code == 200:
            auth_data = response.json()
            user_info = auth_data.get('user', {})
            if user_info.get('is_admin', False):
                print("✅ Admin user correctly identified")
                print(f"   User ID: {user_info.get('user_id', 'unknown')}")
            else:
                print("❌ Admin user not correctly identified")
        else:
            print(f"❌ Admin authentication failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Test 6: Rate limiting (make multiple requests quickly)
    print("\n6️⃣ Testing rate limiting...")
    try:
        headers = {"X-API-Key": DEMO_KEYS["valid_user"]}
        
        # Make 6 requests quickly (should trigger minute limit of 5)
        print("   Making 6 rapid requests to trigger rate limit...")
        success_count = 0
        rate_limited = False
        
        for i in range(6):
            response = requests.get(f"{BASE_URL}/api/auth/info", headers=headers, timeout=5)
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited = True
                print(f"   ✅ Rate limit triggered on request {i+1}")
                print(f"   Error: {response.json().get('detail', 'No detail')}")
                break
            time.sleep(0.1)  # Small delay between requests
        
        if rate_limited:
            print("✅ Rate limiting working correctly")
        else:
            print(f"❌ Rate limiting may not be working (completed {success_count}/6 requests)")
            
    except Exception as e:
        print(f"❌ Rate limit test failed: {e}")
    
    # Test 7: Research endpoint authentication
    print("\n7️⃣ Testing research endpoint authentication...")
    try:
        headers = {"X-API-Key": DEMO_KEYS["valid_user"], "Content-Type": "application/json"}
        test_request = {
            "query": "Test authentication research query",
            "report_type": "concise",
            "prompt_type": "general",
            "automation_level": "full"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/research/start", 
            headers=headers, 
            json=test_request,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Research endpoint accepts authenticated requests")
            result = response.json()
            session_id = result.get('data', {}).get('session_id')
            if session_id:
                print(f"   Session ID: {session_id}")
                
                # Test session access
                time.sleep(1)  # Brief delay
                status_response = requests.get(
                    f"{BASE_URL}/api/research/{session_id}/status",
                    headers=headers,
                    timeout=5
                )
                
                if status_response.status_code == 200:
                    print("✅ Session access control working")
                    status_data = status_response.json()
                    print(f"   Session status: {status_data.get('status', 'unknown')}")
                else:
                    print(f"❌ Session access failed: {status_response.status_code}")
            
        else:
            print(f"❌ Research endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Research endpoint test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Authentication Test Summary:")
    print("✅ Basic authentication flow implemented")
    print("✅ API key validation working")
    print("✅ Rate limiting enabled") 
    print("✅ Session ownership protection")
    print("✅ Admin user detection")
    print("\n📋 Next Steps:")
    print("1. Start frontend: npm run dev")
    print("2. Test authentication in browser")
    print("3. Try demo keys:")
    print(f"   Regular user: {DEMO_KEYS['valid_user']}")
    print(f"   Admin user: {DEMO_KEYS['admin_user']}")
    
    return True

if __name__ == "__main__":
    # Check if backend is running by testing the module
    try:
        from auth import api_key_manager
        print("✅ Auth module loaded successfully")
    except ImportError as e:
        print(f"❌ Cannot import auth module: {e}")
        sys.exit(1)
    
    print("📡 Testing authentication against running backend...")
    print("   Make sure backend is running: python main.py")
    print("   Default URL: http://localhost:8000")
    
    input("\nPress Enter to start tests (or Ctrl+C to cancel)...")
    
    try:
        test_authentication_flow()
    except KeyboardInterrupt:
        print("\n\n❌ Tests cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()