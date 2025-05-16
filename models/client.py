from datetime import datetime
import uuid
from models.user import User

class Client:
    def __init__(self, first_name, last_name, email, phone_number, alternate_number=None,
                 address=None, city=None, state=None, zip_code=None, country=None,
                 preferred_contact='email', client_type='Individual', status='Active'):
        self.client_id = str(uuid.uuid4())
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
        self.date_added = datetime.utcnow()
        self.status = status

    @staticmethod
    def create_collection():
        """Create the clients collection if it doesn't exist"""
        if 'clients' not in User.db.list_collection_names():
            User.db.create_collection('clients')
            # Create indexes
            User.db.clients.create_index('client_id', unique=True)
            User.db.clients.create_index('email', unique=True)
            User.db.clients.create_index('phone_number')

    def save(self):
        """Save the client to the database"""
        self.create_collection()
        client_data = self.__dict__
        User.db.clients.insert_one(client_data)

    @staticmethod
    def get_all():
        """Get all clients"""
        return list(User.db.clients.find())

    @staticmethod
    def get_by_id(client_id):
        """Get a client by ID"""
        return User.db.clients.find_one({'client_id': client_id})

    @staticmethod
    def update(client_id, update_data):
        """Update a client's information"""
        return User.db.clients.update_one(
            {'client_id': client_id},
            {'$set': update_data}
        )

    @staticmethod
    def delete(client_id):
        """Delete a client"""
        return User.db.clients.delete_one({'client_id': client_id})

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