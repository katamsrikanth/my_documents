from datetime import datetime
from bson import ObjectId
import uuid
from models.user import User
import logging

logger = logging.getLogger(__name__)

class Client:
    def __init__(self, first_name, last_name, email, phone_number=None, alternate_number=None,
                 address=None, city=None, state=None, zip_code=None, country=None,
                 preferred_contact='email', client_type='individual'):
        self.client_id = str(uuid.uuid4())  # Generate UUID string
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone_number = phone_number
        self.alternate_number = alternate_number
        self.address = address
        self.city = city
        self.state = state
        self.zip_code = zip_code
        self.country = country
        self.preferred_contact = preferred_contact
        self.client_type = client_type
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        return {
            'client_id': self.client_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'phone_number': self.phone_number,
            'alternate_number': self.alternate_number,
            'address': self.address,
            'city': self.city,
            'state': self.state,
            'zip_code': self.zip_code,
            'country': self.country,
            'preferred_contact': self.preferred_contact,
            'client_type': self.client_type,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @classmethod
    def from_dict(cls, data):
        client = cls(
            first_name=data['first_name'],
            last_name=data['last_name'],
            email=data['email'],
            phone_number=data.get('phone_number'),
            alternate_number=data.get('alternate_number'),
            address=data.get('address'),
            city=data.get('city'),
            state=data.get('state'),
            zip_code=data.get('zip_code'),
            country=data.get('country'),
            preferred_contact=data.get('preferred_contact', 'email'),
            client_type=data.get('client_type', 'individual')
        )
        client.client_id = data.get('client_id', str(uuid.uuid4()))
        client.created_at = data.get('created_at', datetime.utcnow())
        client.updated_at = data.get('updated_at', datetime.utcnow())
        return client

    def save(self, db=None):
        """Save the client to the database."""
        db = db or User.db
        client_data = self.to_dict()
        result = db.clients.insert_one(client_data)
        return str(result.inserted_id)

    @classmethod
    def get_by_id(cls, client_id):
        """Get a client by ID."""
        try:
            client_data = User.db.clients.find_one({'client_id': client_id})
            if client_data:
                return cls.from_dict(client_data)
            return None
        except Exception as e:
            logger.error(f"Error getting client by ID: {str(e)}")
            return None

    @classmethod
    def get_all(cls):
        """Get all clients."""
        try:
            clients = []
            for client_data in User.db.clients.find():
                client = cls.from_dict(client_data)
                clients.append(client)
            return clients
        except Exception as e:
            logger.error(f"Error getting all clients: {str(e)}")
            return []

    def update(self, db=None):
        """Update the client in the database."""
        db = db or User.db
        self.updated_at = datetime.utcnow()
        client_data = self.to_dict()
        db.clients.update_one(
            {'client_id': self.client_id},
            {'$set': client_data}
        )

    def delete(self, db=None):
        """Delete the client from the database."""
        db = db or User.db
        db.clients.delete_one({'client_id': self.client_id})

    @staticmethod
    def create_collection():
        """Create the clients collection if it doesn't exist"""
        if 'clients' not in User.db.list_collection_names():
            User.db.create_collection('clients')
            # Create indexes
            User.db.clients.create_index('client_id', unique=True)
            User.db.clients.create_index('email', unique=True)
            User.db.clients.create_index('phone_number')

    @staticmethod
    def search(query):
        """Search clients by name, email, or phone"""
        search_query = {
            '$or': [
                {'first_name': {'$regex': query, '$options': 'i'}},
                {'last_name': {'$regex': query, '$options': 'i'}},
                {'email': {'$regex': query, '$options': 'i'}},
                {'phone_number': {'$regex': query, '$options': 'i'}}
            ]
        }
        return list(User.db.clients.find(search_query)) 