"""
Generate grid-based heatmaps for Pyracantha invasion risk prediction using XGBoost Enhanced model and real API data.

This script:
1. Creates a grid of coordinates for the specified geographic area
2. Fetches real environmental data from the local API endpoints for each grid point
3. Processes and formats the data for compatibility with the trained XGBoost Enhanced model
4. Predicts invasion risk for each grid point
5. Creates an interactive grid-based choropleth map of invasion risk

Usage:
    python -m experiments.xgboost_enhanced.generate_heatmap_api [options]

Requirements:
    - FastAPI server running on localhost:8000 (uvicorn app.main:app)
    - Trained XGBoost Enhanced model (model.pkl)
    - Python packages: folium, numpy, pandas, requests, xgboost
"""

import os
import sys
import argparse
import datetime
import pickle
import numpy as np
import pandas as pd
from math import pi
import folium
from branca.colormap import LinearColormap
from folium.plugins import MarkerCluster, MeasureControl, Fullscreen
import aiohttp
import asyncio
import json
import time
import random

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Directory where model files are stored
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(model_file='model.pkl'):
    """Load the trained XGBoost Enhanced model from disk."""
    try:
        model_path = os.path.join(MODEL_DIR, model_file)
        print(f"Loading XGBoost Enhanced model from {model_path}...")
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print(f"Available files in {MODEL_DIR}:")
            for file in os.listdir(MODEL_DIR):
                print(f"  - {file}")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded successfully: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Continue with the rest of the implementation...
# This is a placeholder - the full implementation would be similar to the original
# but adapted for the XGBoost Enhanced model