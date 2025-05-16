from datetime import datetime
import uuid
from bson import ObjectId
from database import get_db
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class InitialInquiry:
    def __init__(self, fullName, phoneNumber, email=None, appointmentType=None, 
                 preferredDate=None, preferredTime=None, referralSource=None,
                 assignedAttorney=None, caseDescription=None, notes=None):
        logger.debug(f"Initializing InitialInquiry with name: {fullName}")
        self.inquiry_id = str(uuid.uuid4())
        self.fullName = fullName
        self.phoneNumber = phoneNumber
        self.email = email
        self.appointmentType = appointmentType
        self.preferredDate = preferredDate
        self.preferredTime = preferredTime
        self.referralSource = referralSource
        self.assignedAttorney = assignedAttorney
        self.caseDescription = caseDescription
        self.notes = notes
        self.callDateTime = datetime.now()
        self.status = "New"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        logger.debug(f"Created InitialInquiry with ID: {self.inquiry_id}")

    @staticmethod
    def create_collection():
        logger.debug("Attempting to create initial_inquiries collection")
        db = get_db()
        logger.debug(f"Current database: {db.name}")
        # Switch to my_law_users database
        db = db.client['my_law_users']
        logger.debug(f"Switched to database: {db.name}")
        if 'initial_inquiries' not in db.list_collection_names():
            logger.debug("Creating initial_inquiries collection")
            db.create_collection('initial_inquiries')
            # Create indexes
            logger.debug("Creating indexes for initial_inquiries collection")
            db.initial_inquiries.create_index('fullName')
            db.initial_inquiries.create_index('phoneNumber')
            db.initial_inquiries.create_index('email')
            db.initial_inquiries.create_index('callDateTime')
            db.initial_inquiries.create_index('status')
            logger.debug("Collection and indexes created successfully")
        else:
            logger.debug("Collection already exists")

    def save(self):
        logger.debug(f"Attempting to save inquiry with ID: {self.inquiry_id}")
        db = get_db()
        # Switch to my_law_users database
        db = db.client['my_law_users']
        logger.debug(f"Using database: {db.name}")
        inquiry_data = {
            'inquiry_id': self.inquiry_id,
            'fullName': self.fullName,
            'phoneNumber': self.phoneNumber,
            'email': self.email,
            'appointmentType': self.appointmentType,
            'preferredDate': self.preferredDate,
            'preferredTime': self.preferredTime,
            'referralSource': self.referralSource,
            'assignedAttorney': self.assignedAttorney,
            'caseDescription': self.caseDescription,
            'notes': self.notes,
            'callDateTime': self.callDateTime,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        logger.debug(f"Inquiry data to save: {inquiry_data}")
        result = db.initial_inquiries.insert_one(inquiry_data)
        logger.debug(f"Save result: {result.inserted_id}")
        return result.inserted_id

    @staticmethod
    def get_all():
        logger.debug("Fetching all inquiries")
        db = get_db()
        # Switch to my_law_users database
        db = db.client['my_law_users']
        logger.debug(f"Using database: {db.name}")
        inquiries = list(db.initial_inquiries.find().sort('callDateTime', -1))
        logger.debug(f"Found {len(inquiries)} inquiries")
        return inquiries

    @staticmethod
    def get_recent(limit=5):
        logger.debug(f"Fetching {limit} recent inquiries")
        db = get_db()
        # Switch to my_law_users database
        db = db.client['my_law_users']
        logger.debug(f"Using database: {db.name}")
        inquiries = list(db.initial_inquiries.find().sort('callDateTime', -1).limit(limit))
        logger.debug(f"Found {len(inquiries)} recent inquiries")
        return inquiries

    @classmethod
    def get_by_id(cls, inquiry_id):
        """Get an inquiry by ID"""
        try:
            logger.debug(f"Fetching inquiry with ID: {inquiry_id}")
            db = get_db()
            # Switch to my_law_users database
            db = db.client['my_law_users']
            logger.debug(f"Using database: {db.name}")
            inquiry = db.initial_inquiries.find_one({'inquiry_id': inquiry_id})
            if inquiry:
                logger.debug(f"Found inquiry: {inquiry}")
                # Convert ObjectId to string
                inquiry['_id'] = str(inquiry['_id'])
                
                # Convert datetime fields to strings
                datetime_fields = ['callDateTime', 'created_at', 'updated_at']
                for field in datetime_fields:
                    if field in inquiry and inquiry[field] is not None:
                        if isinstance(inquiry[field], datetime):
                            inquiry[field] = inquiry[field].strftime('%Y-%m-%d %H:%M:%S')
                return inquiry
            logger.debug(f"No inquiry found with ID: {inquiry_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting inquiry by ID: {str(e)}")
            return None

    @staticmethod
    def update(inquiry_id, update_data):
        logger.debug(f"Updating inquiry with ID: {inquiry_id}")
        logger.debug(f"Update data: {update_data}")
        db = get_db()
        # Switch to my_law_users database
        db = db.client['my_law_users']
        logger.debug(f"Using database: {db.name}")
        update_data['updated_at'] = datetime.now()
        result = db.initial_inquiries.update_one(
            {'inquiry_id': inquiry_id},
            {'$set': update_data}
        )
        logger.debug(f"Update result: {result.modified_count} documents modified")
        return result.modified_count > 0

    @staticmethod
    def delete(inquiry_id):
        logger.debug(f"Deleting inquiry with ID: {inquiry_id}")
        db = get_db()
        # Switch to my_law_users database
        db = db.client['my_law_users']
        logger.debug(f"Using database: {db.name}")
        result = db.initial_inquiries.delete_one({'inquiry_id': inquiry_id})
        logger.debug(f"Delete result: {result.deleted_count} documents deleted")
        return result.deleted_count > 0

    @staticmethod
    def search(query):
        logger.debug(f"Searching inquiries with query: {query}")
        db = get_db()
        # Switch to my_law_users database
        db = db.client['my_law_users']
        logger.debug(f"Using database: {db.name}")
        search_query = {
            '$or': [
                {'fullName': {'$regex': query, '$options': 'i'}},
                {'phoneNumber': {'$regex': query, '$options': 'i'}},
                {'email': {'$regex': query, '$options': 'i'}},
                {'caseDescription': {'$regex': query, '$options': 'i'}}
            ]
        }
        logger.debug(f"Search query: {search_query}")
        results = list(db.initial_inquiries.find(search_query).sort('callDateTime', -1))
        logger.debug(f"Found {len(results)} matching inquiries")
        return results 