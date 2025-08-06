import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB connection settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "invasive_db")

# Global variables for client and database
client: MongoClient = None
db = None


def connect_to_mongo():
    """Establish connection to MongoDB."""
    global client, db
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print(f"Connected to MongoDB: {MONGO_URI}/{DB_NAME}")


def close_mongo_connection():
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
        print("MongoDB connection closed")


# Helper getters
def get_database():
    return db

def get_inat_collection():
    return db["inat_observations"]

def get_weather_collection():
    return db["weather_data"]

def get_stats_collection():
    return db["system_stats"]
