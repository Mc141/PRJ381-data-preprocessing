import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB connection settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "invasive_db")

# Create MongoDB client (singleton)
client = MongoClient(MONGO_URI)

# Access the database
db = client[DB_NAME]

# Collections
inat_collection = db["inat_observations"]
weather_collection = db["weather_data"]
stats_collection = db["system_stats"]

# Helper getters
def get_database():
    """Return the database object."""
    return db

def get_inat_collection():
    """Return the iNaturalist observations collection."""
    return inat_collection

def get_weather_collection():
    """Return the weather data collection."""
    return weather_collection

def get_stats_collection():
    """Return the system stats collection."""
    return stats_collection
