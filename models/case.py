import uuid
from datetime import datetime
from models.user import User
from bson import ObjectId
from database import get_db

class Case:
    def __init__(self, client_id, case_number, case_type, status, description, start_date, title=None, court_name=None, end_date=None, 
                 assigned_to=None, assigned_attorney_id=None, priority='Medium', notes=None, documents=None):
        self.client_id = client_id
        self.case_number = case_number
        self.case_type = case_type
        self.status = status
        self.description = description
        self.title = title
        self.court_name = court_name
        self.start_date = start_date if isinstance(start_date, datetime) else datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = end_date if isinstance(end_date, datetime) else datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
        self.assigned_to = assigned_to
        self.assigned_attorney_id = assigned_attorney_id
        self.priority = priority
        self.notes = notes or []
        self.documents = documents or []
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    @staticmethod
    def create_collection():
        """Create the cases collection if it doesn't exist"""
        if 'cases' not in User.db.list_collection_names():
            User.db.create_collection('cases')
            # Create indexes
            User.db.cases.create_index('case_id', unique=True)
            User.db.cases.create_index('client_id')
            User.db.cases.create_index('case_number', unique=True)
            User.db.cases.create_index('assigned_attorney_id')

    def save(self):
        db = get_db()
        case_data = {
            'client_id': self.client_id,
            'case_number': self.case_number,
            'case_type': self.case_type,
            'status': self.status,
            'description': self.description,
            'title': self.title,
            'court_name': self.court_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'assigned_to': self.assigned_to,
            'assigned_attorney_id': self.assigned_attorney_id,
            'priority': self.priority,
            'notes': self.notes,
            'documents': self.documents,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        result = db.cases.insert_one(case_data)
        return str(result.inserted_id)

    @staticmethod
    def get_all():
        db = get_db()
        cases = list(db.cases.find())
        for case in cases:
            case['_id'] = str(case['_id'])
            if 'client_id' in case:
                case['client_id'] = str(case['client_id'])
            if case.get('start_date'):
                case['start_date'] = case['start_date'] if isinstance(case['start_date'], datetime) else datetime.strptime(case['start_date'], '%Y-%m-%d')
            if case.get('end_date'):
                case['end_date'] = case['end_date'] if isinstance(case['end_date'], datetime) else datetime.strptime(case['end_date'], '%Y-%m-%d')
        return cases

    @staticmethod
    def get_by_id(case_id):
        db = get_db()
        try:
            # First try with ObjectId
            case = db.cases.find_one({'_id': ObjectId(case_id)})
        except:
            # If that fails, try with the string ID directly
            case = db.cases.find_one({'_id': case_id})
            
        if case:
            case['_id'] = str(case['_id'])
            if 'client_id' in case:
                case['client_id'] = str(case['client_id'])
            if case.get('start_date'):
                case['start_date'] = case['start_date'] if isinstance(case['start_date'], datetime) else datetime.strptime(case['start_date'], '%Y-%m-%d')
            if case.get('end_date'):
                case['end_date'] = case['end_date'] if isinstance(case['end_date'], datetime) else datetime.strptime(case['end_date'], '%Y-%m-%d')
        return case

    @staticmethod
    def update(case_id, update_data):
        db = get_db()
        if 'start_date' in update_data and isinstance(update_data['start_date'], str):
            update_data['start_date'] = datetime.strptime(update_data['start_date'], '%Y-%m-%d')
        if 'end_date' in update_data and isinstance(update_data['end_date'], str):
            update_data['end_date'] = datetime.strptime(update_data['end_date'], '%Y-%m-%d')
        update_data['updated_at'] = datetime.utcnow()
        result = db.cases.update_one(
            {'_id': ObjectId(case_id)},
            {'$set': update_data}
        )
        return result.modified_count > 0

    @staticmethod
    def delete(case_id):
        db = get_db()
        result = db.cases.delete_one({'_id': ObjectId(case_id)})
        return result.deleted_count > 0

    @staticmethod
    def get_by_client_id(client_id):
        db = get_db()
        cases = list(db.cases.find({'client_id': client_id}))
        for case in cases:
            case['_id'] = str(case['_id'])
            case['client_id'] = str(case['client_id'])
            if case.get('start_date'):
                case['start_date'] = case['start_date'] if isinstance(case['start_date'], datetime) else datetime.strptime(case['start_date'], '%Y-%m-%d')
            if case.get('end_date'):
                case['end_date'] = case['end_date'] if isinstance(case['end_date'], datetime) else datetime.strptime(case['end_date'], '%Y-%m-%d')
        return cases

    @staticmethod
    def search(query):
        db = get_db()
        cases = list(db.cases.find({
            '$or': [
                {'case_number': {'$regex': query, '$options': 'i'}},
                {'case_type': {'$regex': query, '$options': 'i'}},
                {'description': {'$regex': query, '$options': 'i'}}
            ]
        }))
        for case in cases:
            case['_id'] = str(case['_id'])
            case['client_id'] = str(case['client_id'])
            if case.get('start_date'):
                case['start_date'] = case['start_date'] if isinstance(case['start_date'], datetime) else datetime.strptime(case['start_date'], '%Y-%m-%d')
            if case.get('end_date'):
                case['end_date'] = case['end_date'] if isinstance(case['end_date'], datetime) else datetime.strptime(case['end_date'], '%Y-%m-%d')
        return cases 