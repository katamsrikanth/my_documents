from datetime import datetime
from bson import ObjectId
from models.base import BaseModel

class Attorney(BaseModel):
    collection_name = 'attorneys'

    def __init__(self, name, email=None, phone=None, specialization=None, bar_number=None, status='Active'):
        self.name = name
        self.email = email
        self.phone = phone
        self.specialization = specialization
        self.bar_number = bar_number
        self.status = status
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    @classmethod
    def create_collection(cls):
        """Create the attorneys collection if it doesn't exist."""
        if cls.collection_name not in cls.db.list_collection_names():
            cls.db.create_collection(cls.collection_name)
            # Create indexes
            cls.db[cls.collection_name].create_index('name')
            cls.db[cls.collection_name].create_index('email', unique=True)
            cls.db[cls.collection_name].create_index('bar_number', unique=True)

    def save(self):
        """Save the attorney to the database."""
        attorney_data = {
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'specialization': self.specialization,
            'bar_number': self.bar_number,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        result = self.db[self.collection_name].insert_one(attorney_data)
        self._id = result.inserted_id
        return self._id

    @classmethod
    def get_all(cls):
        """Get all attorneys."""
        return list(cls.db[cls.collection_name].find())

    @classmethod
    def get_by_id(cls, attorney_id):
        """Get an attorney by ID."""
        return cls.db[cls.collection_name].find_one({'_id': ObjectId(attorney_id)})

    @classmethod
    def get_active(cls):
        """Get all active attorneys."""
        return list(cls.db[cls.collection_name].find({'status': 'Active'}))

    @classmethod
    def search(cls, query):
        """Search attorneys by name, email, or specialization."""
        search_query = {
            '$or': [
                {'name': {'$regex': query, '$options': 'i'}},
                {'email': {'$regex': query, '$options': 'i'}},
                {'specialization': {'$regex': query, '$options': 'i'}}
            ]
        }
        return list(cls.db[cls.collection_name].find(search_query))

    @classmethod
    def update(cls, attorney_id, update_data):
        """Update an attorney's information."""
        update_data['updated_at'] = datetime.now()
        return cls.db[cls.collection_name].update_one(
            {'_id': ObjectId(attorney_id)},
            {'$set': update_data}
        )

    @classmethod
    def delete(cls, attorney_id):
        """Delete an attorney."""
        return cls.db[cls.collection_name].delete_one({'_id': ObjectId(attorney_id)}) 