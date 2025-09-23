#!/usr/bin/env python3
"""
Test script for the XGBoost heatmap generation endpoint.
"""

import requests
import json
import time

def test_heatmap_endpoint():
    """Test the XGBoost heatmap generation endpoint."""
    
    # Endpoint URL
    url = "http://localhost:8000/api/v1/predictions/generate-xgboost-heatmap"
    
    # Test parameters - small grid for quick testing
    params = {
        "region": "stellenbosch",
        "grid_size": 10,  # Small grid for quick test
        "month": 6,
        "batch_size": 10,
        "rate_limit_delay": 0.5,
        "include_stats": True,
        "return_html": False,  # Don't return HTML content for this test
        "save_file": True  # Save file so we can verify it works
    }
    
    print("Testing XGBoost heatmap generation endpoint...")
    print(f"URL: {url}")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    print()
    
    try:
        # Make the request
        print("Sending request...")
        start_time = time.time()
        
        response = requests.post(url, params=params)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Response received in {processing_time:.1f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS!")
            print(f"Region: {result['region']}")
            print(f"Grid size: {result['parameters']['grid_size']} ({result['parameters']['total_points']} points)")
            print(f"Month: {result['parameters']['month_name']}")
            print(f"Processing time: {result['processing']['processing_time_seconds']:.1f}s")
            print(f"Mean risk: {result['statistics']['mean_risk']:.3f}")
            print(f"High risk points: {result['statistics']['high_risk_points']}")
            
            if result['output']['file_saved']:
                print(f"File saved to: {result['output']['file_saved']}")
            
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error detail: {error_detail}")
            except:
                print(f"Error text: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to FastAPI server.")
        print("Make sure the server is running with: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_heatmap_endpoint()