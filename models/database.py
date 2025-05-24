from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection string
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'my_law_users')

# Create MongoDB client
client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]

def get_db():
    """Get database instance"""
    return db

def close_connection():
    """Close MongoDB connection"""
    if client:
        client.close() 