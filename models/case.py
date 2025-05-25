import uuid
from datetime import datetime
from models.user import User
from bson import ObjectId
from models.database import get_db
import os
import shutil
from werkzeug.utils import secure_filename
import json

class Case:
    def __init__(self, **kwargs):
        self._id = kwargs.get('_id')
        self.client_id = kwargs.get('client_id')  # Store as UUID string
        self.client_name = kwargs.get('client_name')
        self.title = kwargs.get('title')
        self.description = kwargs.get('description')
        self.case_type = kwargs.get('case_type')
        self.court_name = kwargs.get('court_name')
        self.case_number = kwargs.get('case_number')
        self.status = kwargs.get('status')
        # Convert string dates to datetime objects
        self.start_date = kwargs.get('start_date')
        if isinstance(self.start_date, str):
            try:
                self.start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
            except (ValueError, TypeError):
                self.start_date = None
        self.end_date = kwargs.get('end_date')
        if isinstance(self.end_date, str):
            try:
                self.end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
            except (ValueError, TypeError):
                self.end_date = None
        self.priority = kwargs.get('priority')
        self.assigned_attorney_id = kwargs.get('assigned_attorney_id')
        self.documents = kwargs.get('documents', [])  # List of document metadata
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())

    @staticmethod
    def create_collection():
        """Create the cases collection if it doesn't exist"""
        db = get_db()
        if 'cases' not in db.list_collection_names():
            db.create_collection('cases')
            # Create indexes
            db.cases.create_index('client_id')  # Index on client_id for faster lookups
            db.cases.create_index('case_number', unique=True)
            db.cases.create_index('assigned_attorney_id')

    def save(self):
        collection = self.get_collection()
        # Convert datetime objects to strings for storage
        save_data = self.__dict__.copy()
        if isinstance(save_data.get('start_date'), datetime):
            save_data['start_date'] = save_data['start_date'].strftime('%Y-%m-%d')
        if isinstance(save_data.get('end_date'), datetime):
            save_data['end_date'] = save_data['end_date'].strftime('%Y-%m-%d')
        
        # Remove _id from save_data if it's None to let MongoDB generate it
        if save_data.get('_id') is None:
            save_data.pop('_id', None)
        
        if self._id:
            self.updated_at = datetime.now()
            collection.update_one(
                {'_id': ObjectId(self._id)},
                {'$set': save_data}
            )
        else:
            self.created_at = datetime.now()
            self.updated_at = self.created_at
            result = collection.insert_one(save_data)
            self._id = str(result.inserted_id)
        return self

    @staticmethod
    def get_collection():
        return get_db().cases

    @staticmethod
    def get_all():
        collection = Case.get_collection()
        cases = list(collection.find())
        for case in cases:
            case['_id'] = str(case['_id'])
        return cases

    @staticmethod
    def get_by_id(case_id):
        collection = Case.get_collection()
        try:
            # Try to convert string to ObjectId if it's a valid ObjectId
            if isinstance(case_id, str):
                try:
                    case_id = ObjectId(case_id)
                except:
                    # If conversion fails, use the string as is
                    pass
            
            case = collection.find_one({'_id': case_id})
            if case:
                case['_id'] = str(case['_id'])
                # Convert string dates to datetime objects
                if isinstance(case.get('start_date'), str):
                    try:
                        case['start_date'] = datetime.strptime(case['start_date'], '%Y-%m-%d')
                    except (ValueError, TypeError):
                        case['start_date'] = None
                if isinstance(case.get('end_date'), str):
                    try:
                        case['end_date'] = datetime.strptime(case['end_date'], '%Y-%m-%d')
                    except (ValueError, TypeError):
                        case['end_date'] = None
            return case
        except Exception as e:
            print(f"Error in get_by_id: {str(e)}")
            return None

    @staticmethod
    def update(case_id, update_data):
        collection = Case.get_collection()
        update_data['updated_at'] = datetime.now()
        return collection.update_one(
            {'_id': ObjectId(case_id)},
            {'$set': update_data}
        )

    @staticmethod
    def delete(case_id):
        collection = Case.get_collection()
        # Delete associated documents from storage
        case = Case.get_by_id(case_id)
        if case and 'documents' in case:
            for doc in case['documents']:
                doc_path = os.path.join('/Users/srikanthkatam/Documents/srikanth/Apps/legal_storage/case_documents', doc['filename'])
                if os.path.exists(doc_path):
                    os.remove(doc_path)
        return collection.delete_one({'_id': ObjectId(case_id)})

    @staticmethod
    def get_by_client_id(client_id):
        collection = Case.get_collection()
        cases = list(collection.find({'client_id': client_id}))  # Using client_id directly as UUID string
        for case in cases:
            case['_id'] = str(case['_id'])
        return cases

    @staticmethod
    def search(query):
        collection = Case.get_collection()
        cases = list(collection.find({
            '$or': [
                {'title': {'$regex': query, '$options': 'i'}},
                {'case_number': {'$regex': query, '$options': 'i'}},
                {'description': {'$regex': query, '$options': 'i'}}
            ]
        }))
        for case in cases:
            case['_id'] = str(case['_id'])
        return cases

    @staticmethod
    def get_document_storage_path():
        base_path = os.getenv('DOCUMENT_STORAGE_PATH', 'documents')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        return base_path

    def get_case_document_path(self):
        case_path = os.path.join(self.get_document_storage_path(), str(self._id))
        if not os.path.exists(case_path):
            os.makedirs(case_path)
        return case_path

    def add_document(self, file, description=''):
        if not self._id:
            raise ValueError("Case must be saved before adding documents")

        filename = secure_filename(file.filename)
        case_path = self.get_case_document_path()
        
        # Get the latest version number for this document
        existing_versions = [doc['version'] for doc in self.documents if doc['filename'] == filename]
        version = max(existing_versions) + 1 if existing_versions else 1
        
        # Create version directory
        version_dir = os.path.join(case_path, f"v{version}")
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)
        
        # Save the file
        file_path = os.path.join(version_dir, filename)
        file.save(file_path)
        
        # Create document metadata
        document = {
            '_id': str(ObjectId()),
            'filename': filename,
            'original_filename': file.filename,
            'version': version,
            'description': description,
            'uploaded_at': datetime.now(),
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'mime_type': file.content_type
        }
        
        # Add to documents list
        self.documents.append(document)
        
        # Update in database
        db = get_db()
        db.cases.update_one(
            {'_id': ObjectId(self._id)},
            {'$push': {'documents': document}}
        )
        
        return document

    def delete_document(self, document_id):
        if not self._id:
            raise ValueError("Case must be saved before deleting documents")
        
        # Find the document
        document = next((doc for doc in self.documents if doc['_id'] == document_id), None)
        if not document:
            raise ValueError("Document not found")
        
        # Delete the file
        if os.path.exists(document['file_path']):
            os.remove(document['file_path'])
        
        # Remove from documents list
        self.documents = [doc for doc in self.documents if doc['_id'] != document_id]
        
        # Update in database
        db = get_db()
        db.cases.update_one(
            {'_id': ObjectId(self._id)},
            {'$pull': {'documents': {'_id': document_id}}}
        )

    def get_document(self, document_id):
        return next((doc for doc in self.documents if doc['_id'] == document_id), None)

    @staticmethod
    def get_document_path(case_id, filename):
        """Get the full path to a document"""
        return os.path.join('/Users/srikanthkatam/Documents/srikanth/Apps/legal_storage/case_documents', 
                           str(case_id), filename) 