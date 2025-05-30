import uuid
from datetime import datetime
from models.user import User
from bson import ObjectId
from models.database import get_db
import os
import shutil
from werkzeug.utils import secure_filename
import json
from models.base import BaseModel
from models.attorney import Attorney
from models.attorney_case_history import AttorneyCaseHistory
import logging

logger = logging.getLogger(__name__)

class Case(BaseModel):
    collection_name = 'cases'

    def __init__(self, **kwargs):
        # Don't call super().__init__() since BaseModel's __init__ is causing issues
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
        self.court_visits = kwargs.get('court_visits', [])  # List of court visits with notes
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
        self.attorney_ids = kwargs.get('attorney_ids', [])
        self.attorneys = kwargs.get('attorneys', [])

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
            # Remove _id from save_data before update
            save_data.pop('_id', None)
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

    @classmethod
    def get_by_id(cls, case_id):
        collection = cls.get_collection()
        try:
            # Try to convert string to ObjectId if it's a valid ObjectId
            if isinstance(case_id, str):
                try:
                    case_id = ObjectId(case_id)
                except:
                    # If conversion fails, try to find by case_id field
                    case_data = collection.find_one({'case_id': case_id})
                    if case_data:
                        case_data['_id'] = str(case_data['_id'])
                        case = cls(**case_data)
                        # Load attorneys
                        case.attorneys = [Attorney.get_by_id(str(attorney_id)) for attorney_id in case.attorney_ids if attorney_id]
                        return case
                    return None
            
            case_data = collection.find_one({'_id': case_id})
            if case_data:
                case_data['_id'] = str(case_data['_id'])
                # Convert string dates to datetime objects
                if isinstance(case_data.get('start_date'), str):
                    try:
                        case_data['start_date'] = datetime.strptime(case_data['start_date'], '%Y-%m-%d')
                    except (ValueError, TypeError):
                        case_data['start_date'] = None
                if isinstance(case_data.get('end_date'), str):
                    try:
                        case_data['end_date'] = datetime.strptime(case_data['end_date'], '%Y-%m-%d')
                    except (ValueError, TypeError):
                        case_data['end_date'] = None
                case = cls(**case_data)
                # Load attorneys
                case.attorneys = [Attorney.get_by_id(str(attorney_id)) for attorney_id in case.attorney_ids if attorney_id]
                return case
            return None
        except Exception as e:
            logger.error(f"Error in get_by_id: {str(e)}")
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
        if case and hasattr(case, 'documents') and case.documents:
            for doc in case.documents:
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

    def add_court_visit(self, visit_date, notes):
        """Add a new court visit with notes"""
        # Convert string date to datetime if needed
        if isinstance(visit_date, str):
            try:
                visit_date = datetime.strptime(visit_date, '%Y-%m-%d')
            except (ValueError, TypeError):
                visit_date = datetime.utcnow()

        visit = {
            '_id': str(ObjectId()),
            'date': visit_date,
            'notes': notes,
            'created_at': datetime.utcnow()
        }
        self.court_visits.append(visit)
        return visit

    def update_court_visit(self, visit_id, notes):
        """Update notes for a specific court visit"""
        for visit in self.court_visits:
            if visit['_id'] == visit_id:
                visit['notes'] = notes
                visit['updated_at'] = datetime.utcnow()
                return visit
        return None

    def delete_court_visit(self, visit_id):
        """Delete a court visit"""
        self.court_visits = [visit for visit in self.court_visits if visit['_id'] != visit_id]

    def associate_attorney(self, attorney_id, user=None):
        if attorney_id not in self.attorney_ids:
            self.attorney_ids.append(attorney_id)
            self.save()
            AttorneyCaseHistory.log(self._id, attorney_id, 'associate', user)

    def deassociate_attorney(self, attorney_id, user=None):
        if attorney_id in self.attorney_ids:
            self.attorney_ids.remove(attorney_id)
            self.save()
            AttorneyCaseHistory.log(self._id, attorney_id, 'deassociate', user)

    def update_attorneys(self, new_attorney_ids, user=None):
        """Update attorneys and log changes in history"""
        old_attorney_ids = set(str(attorney._id) for attorney in self.attorneys) if self.attorneys else set()
        new_attorney_ids = set(new_attorney_ids)

        # Log removals
        for attorney_id in old_attorney_ids - new_attorney_ids:
            AttorneyCaseHistory.log(str(self._id), attorney_id, 'deassociate', user)

        # Log additions
        for attorney_id in new_attorney_ids - old_attorney_ids:
            AttorneyCaseHistory.log(str(self._id), attorney_id, 'associate', user)

        # Update attorneys
        self.attorneys = [Attorney.get_by_id(attorney_id) for attorney_id in new_attorney_ids if attorney_id]
        self.updated_at = datetime.utcnow()
        self.save()

    def get_attorney_history(self):
        """Get the history of attorney assignments"""
        return AttorneyCaseHistory.get_history_for_case(str(self._id)) 