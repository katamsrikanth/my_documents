from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BaseModel:
    """Base model class that provides common database functionality."""
    
    # MongoDB connection
    client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
    db = client[os.getenv('MONGODB_DB', 'law_firm_db')]
    
    @classmethod
    def create_collection(cls):
        """Create the collection if it doesn't exist."""
        if cls.collection_name not in cls.db.list_collection_names():
            cls.db.create_collection(cls.collection_name)
    
    @classmethod
    def get_all(cls):
        """Get all documents from the collection."""
        return list(cls.db[cls.collection_name].find())
    
    @classmethod
    def get_by_id(cls, doc_id):
        """Get a document by its ID."""
        from bson import ObjectId
        return cls.db[cls.collection_name].find_one({'_id': ObjectId(doc_id)})
    
    @classmethod
    def delete(cls, doc_id):
        """Delete a document by its ID."""
        from bson import ObjectId
        return cls.db[cls.collection_name].delete_one({'_id': ObjectId(doc_id)})
    
    @classmethod
    def update(cls, doc_id, update_data):
        """Update a document by its ID."""
        from bson import ObjectId
        return cls.db[cls.collection_name].update_one(
            {'_id': ObjectId(doc_id)},
            {'$set': update_data}
        ) 