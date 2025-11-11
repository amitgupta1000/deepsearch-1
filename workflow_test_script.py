
import json
import uvicorn
from fastapi import FastAPI
from main import app
from fastapi.testclient import TestClient
import threading
import time


def run_app(app_instance):
    uvicorn.run(app_instance, host="127.0.0.1", port=8000, log_level="info")

def run_sync_test():
    """
    Runs a synchronous test of the research workflow.
    """
    print("--- Starting Synchronous Workflow Test ---")
    
    # Run the FastAPI app in a separate thread
    server_thread = threading.Thread(target=run_app, args=(app,))
    server_thread.daemon = True
    server_thread.start()
    time.sleep(5)  # Give the server time to start

    # Create a TestClient
    client = TestClient(app)
    
    # Define the test request
    test_request = {
        "query": "What are the latest trends in renewable energy?",
        "prompt_type": "general"
    }
    
    print(f"Request Body: {json.dumps(test_request, indent=2)}")

    # Make the API call
    response = client.post("/api/research/test-workflow", json=test_request)
    
    # Print the results
    print(f"Status Code: {response.status_code}")
    
    print("--- Response ---")
    if response.status_code == 200:
        try:
            response_data = response.json()
            print(json.dumps(response_data, indent=2))
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from response.")
            print(f"Response Text: {response.text}")
    else:
        print(f"Error: {response.text}")
        
    print("--- Test Complete ---")

if __name__ == "__main__":
    run_sync_test()
