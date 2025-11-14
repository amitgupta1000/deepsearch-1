
import requests

def test_backend_connection():
    """
    Tests the connection to the backend health check endpoint.
    """
    try:
        response = requests.get("http://localhost:8080/api/health")
        if response.status_code == 200:
            print("Backend connection successful!")
            print("Response:", response.json())
        else:
            print(f"Backend connection failed with status code: {response.status_code}")
            print("Response:", response.text)
            
            # Attempt to connect to the root endpoint to check for a different API path
            print("\nChecking for API path mismatch...")
            try:
                response = requests.get("http://localhost:8080/health")
                if response.status_code == 200:
                    print("Connection successful at /health. The API path may be misconfigured in the frontend.")
                else:
                    print("Connection failed at /health as well.")
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error at /health: {e}")

    except requests.exceptions.ConnectionError as e:
        print(f"Failed to connect to the backend: {e}")
        print("Please ensure the backend server is running and accessible at http://localhost:8080")

if __name__ == "__main__":
    test_backend_connection()
