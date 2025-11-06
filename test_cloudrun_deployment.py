#!/usr/bin/env python3
"""
test_cloudrun_deployment.py - Test script for Cloud Run deployment
Tests the deployed INTELLISEARCH service endpoints
"""

import requests
import json
import time
import sys
from typing import Dict, Any

class CloudRunTester:
    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip('/')
        self.session = requests.Session()
        
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint."""
        try:
            response = self.session.get(f"{self.service_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data.get('status')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_ready_endpoint(self) -> bool:
        """Test the readiness check endpoint."""
        try:
            response = self.session.get(f"{self.service_url}/ready", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Readiness check passed: {data.get('status')}")
                return True
            else:
                print(f"❌ Readiness check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Readiness check error: {e}")
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test the root endpoint."""
        try:
            response = self.session.get(f"{self.service_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Root endpoint working: {data.get('service')}")
                return True
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            return False
    
    def test_direct_research(self, query: str = "What is artificial intelligence?") -> bool:
        """Test the direct research endpoint."""
        try:
            payload = {
                "query": query,
                "prompt_type": "general",
                "enable_automation": True
            }
            
            print(f"🔍 Testing direct research with query: '{query}'")
            response = self.session.post(
                f"{self.service_url}/research/direct",
                json=payload,
                timeout=300  # 5 minutes for research
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Direct research completed: {data.get('status')}")
                if data.get('executed_nodes'):
                    print(f"   Executed nodes: {' → '.join(data['executed_nodes'])}")
                return True
            else:
                print(f"❌ Direct research failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Direct research error: {e}")
            return False
    
    def test_async_research(self, query: str = "Benefits of renewable energy") -> bool:
        """Test the async research endpoint."""
        try:
            payload = {
                "query": query,
                "prompt_type": "general",
                "enable_automation": True
            }
            
            print(f"🔍 Testing async research with query: '{query}'")
            
            # Start research task
            response = self.session.post(
                f"{self.service_url}/research",
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to start async research: {response.status_code}")
                return False
            
            task_data = response.json()
            task_id = task_data.get('task_id')
            print(f"✅ Research task started: {task_id}")
            
            # Poll for completion
            max_polls = 60  # 5 minutes max
            for i in range(max_polls):
                time.sleep(5)  # Wait 5 seconds between polls
                
                status_response = self.session.get(
                    f"{self.service_url}/task/{task_id}",
                    timeout=10
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get('status')
                    progress = status_data.get('progress', '')
                    
                    print(f"   Status: {status} - {progress}")
                    
                    if status == 'completed':
                        print(f"✅ Async research completed successfully")
                        return True
                    elif status == 'failed':
                        error = status_data.get('error', 'Unknown error')
                        print(f"❌ Async research failed: {error}")
                        return False
                else:
                    print(f"❌ Failed to get task status: {status_response.status_code}")
                    return False
            
            print(f"❌ Async research timed out after {max_polls * 5} seconds")
            return False
            
        except Exception as e:
            print(f"❌ Async research error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        print(f"🧪 Testing INTELLISEARCH Cloud Run deployment: {self.service_url}")
        print("=" * 70)
        
        results = {}
        
        # Basic endpoint tests
        print("\n1. Testing health endpoints...")
        results['health'] = self.test_health_endpoint()
        results['ready'] = self.test_ready_endpoint()
        results['root'] = self.test_root_endpoint()
        
        # Research functionality tests
        if all([results['health'], results['ready'], results['root']]):
            print("\n2. Testing research functionality...")
            results['direct_research'] = self.test_direct_research()
            results['async_research'] = self.test_async_research()
        else:
            print("\n❌ Skipping research tests due to failed basic checks")
            results['direct_research'] = False
            results['async_research'] = False
        
        return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_cloudrun_deployment.py <service_url>")
        print("Example: python test_cloudrun_deployment.py https://intellisearch-abc123-uc.a.run.app")
        sys.exit(1)
    
    service_url = sys.argv[1]
    
    tester = CloudRunTester(service_url)
    results = tester.run_all_tests()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your deployment is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()