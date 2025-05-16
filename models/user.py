from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # MongoDB connection - Local instance
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    # Force a connection to verify it works
    client.server_info()
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

class User:
    # Class-level database connection
    client = client
    db = client['my_law_users']
    
    def __init__(self, username, password=None):
        self.username = username
        self.password = password

    def save(self):
        try:
            if not self.db.users.find_one({'username': self.username}):
                self.db.users.insert_one({
                    'username': self.username,
                    'password': generate_password_hash(self.password, method='pbkdf2:sha256')
                })
                logger.info(f"Successfully created user: {self.username}")
                return True
            logger.warning(f"User already exists: {self.username}")
            return False
        except Exception as e:
            logger.error(f"Error saving user {self.username}: {str(e)}")
            raise

    @staticmethod
    def get_by_username(username):
        try:
            user_data = User.db.users.find_one({'username': username})
            if user_data:
                user = User(username)
                return user
            return None
        except Exception as e:
            logger.error(f"Error getting user {username}: {str(e)}")
            raise

    @staticmethod
    def check_password(username, password):
        try:
            user_data = User.db.users.find_one({'username': username})
            if not user_data:
                logger.warning(f"User not found: {username}")
                return False
                
            # Check if the hash is using scrypt
            if user_data['password'].startswith('scrypt:'):
                try:
                    return check_password_hash(user_data['password'], password)
                except ValueError:
                    # If scrypt verification fails, try pbkdf2:sha256
                    return check_password_hash(generate_password_hash(password, method='pbkdf2:sha256'), password)
            else:
                # For non-scrypt hashes, use normal verification
                return check_password_hash(user_data['password'], password)
        except Exception as e:
            logger.error(f"Error checking password for user {username}: {str(e)}")
            raise

    @staticmethod
    def get_all_attorneys():
        """Get all users who are attorneys"""
        try:
            # For now, return all users as attorneys since we don't have a role field yet
            # TODO: Add role field to users and filter by role='attorney'
            return list(User.db.users.find({}, {'username': 1, '_id': 0}))
        except Exception as e:
            logger.error(f"Error getting attorneys: {str(e)}")
            raise 