from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# MongoDB connection string from environment variable
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')

def get_db():
    """Get MongoDB database connection"""
    try:
        client = MongoClient(MONGODB_URI)
        db = client['my_law_users']
        logger.info("Successfully connected to MongoDB")
        return db
    
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise 