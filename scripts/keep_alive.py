import urllib.request
import urllib.error
import time
import os

# Set your backend URL here, or pass it as an environment variable
BACKEND_URL = os.getenv("RENDER_BACKEND_URL", "https://your-backend.onrender.com")

def ping_server():
    print(f"Pinging {BACKEND_URL}...")
    try:
        req = urllib.request.Request(BACKEND_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            print(f"Success! Status code: {status}")
    except urllib.error.HTTPError as e:
        # We still count this as a success in terms of waking the server, 
        # since the server responded with an HTTP error.
        print(f"Server responded with error code: {e.code}")
    except urllib.error.URLError as e:
        print(f"Failed to reach server: {e.reason}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if BACKEND_URL == "https://your-backend.onrender.com":
        print("WARNING: Please set the RENDER_BACKEND_URL environment variable or update the script.")
    ping_server()
