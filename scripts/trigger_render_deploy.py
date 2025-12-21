"""
Trigger Render deployment via API
Monitors deployment status and reports results
"""

import os
import requests
import time
import sys

# Configuration
RENDER_API_KEY = os.getenv("RENDER_API_KEY", "")
RENDER_SERVICE_ID = os.getenv("RENDER_SERVICE_ID_API", "")

if not RENDER_API_KEY:
    print("⚠️  RENDER_API_KEY not set - skipping cloud deployment")
    print("   Model is ready in backend/models/ for local deployment")
    sys.exit(0)

if not RENDER_SERVICE_ID:
    print("⚠️  RENDER_SERVICE_ID_API not set - skipping cloud deployment")
    print("   To enable deployment, create a Render service and set RENDER_SERVICE_ID_API")
    sys.exit(0)

print("=" * 80)
print("RENDER DEPLOYMENT")
print("=" * 80)

# Render API endpoint
RENDER_API_URL = f"https://api.render.com/v1/services/{RENDER_SERVICE_ID}/deploys"

headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {RENDER_API_KEY}"
}

print(f"\n🚀 Triggering deployment for service: {RENDER_SERVICE_ID}")

try:
    # Trigger deployment
    response = requests.post(
        RENDER_API_URL,
        headers=headers,
        json={"clearCache": "clear"}
    )
    
    if response.status_code == 201:
        deploy_data = response.json()
        deploy_id = deploy_data.get("id", "unknown")
        
        print(f"✅ Deployment triggered successfully")
        print(f"   Deploy ID: {deploy_id}")
        print(f"   Status: {deploy_data.get('status', 'unknown')}")
        
        # Monitor deployment (optional - can be removed for faster pipeline)
        print(f"\n⏳ Monitoring deployment status...")
        
        max_wait = 300  # 5 minutes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # Get deployment status
            status_url = f"https://api.render.com/v1/services/{RENDER_SERVICE_ID}/deploys/{deploy_id}"
            status_response = requests.get(status_url, headers=headers)
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get("status", "unknown")
                
                print(f"   Status: {status}")
                
                if status == "live":
                    print("\n✅ Deployment successful!")
                    print("=" * 80)
                    sys.exit(0)
                elif status in ["build_failed", "deactivated"]:
                    print(f"\n❌ Deployment failed with status: {status}")
                    sys.exit(1)
            
            time.sleep(10)  # Check every 10 seconds
        
        print("\n⚠️  Deployment monitoring timeout - check Render dashboard")
        print("   Deployment may still be in progress")
        
    else:
        print(f"❌ Failed to trigger deployment")
        print(f"   Status code: {response.status_code}")
        print(f"   Response: {response.text}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
