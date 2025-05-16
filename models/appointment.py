from datetime import datetime
import uuid
from models.user import User

class Appointment:
    def __init__(self, case_id, date_time, location, purpose, status="Scheduled"):
        self.appointment_id = str(uuid.uuid4())
        self.case_id = case_id
        self.date_time = date_time
        self.location = location
        self.purpose = purpose
        self.status = status
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    @staticmethod
    def create_collection():
        """Create the appointments collection if it doesn't exist"""
        db = User.db
        if 'appointments' not in db.list_collection_names():
            appointments_collection = db['appointments']
            appointments_collection.create_index('appointment_id', unique=True)
            appointments_collection.create_index('case_id')
            appointments_collection.create_index('date_time')
            appointments_collection.create_index('status')
            return appointments_collection
        return db['appointments']

    def save(self):
        """Save the appointment to the database"""
        appointments_collection = self.create_collection()
        appointment_data = {
            'appointment_id': self.appointment_id,
            'case_id': self.case_id,
            'date_time': self.date_time,
            'location': self.location,
            'purpose': self.purpose,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        appointments_collection.insert_one(appointment_data)
        return self

    @staticmethod
    def get_all():
        """Get all appointments"""
        appointments_collection = Appointment.create_collection()
        return list(appointments_collection.find())

    @staticmethod
    def get_by_id(appointment_id):
        """Get an appointment by ID"""
        appointments_collection = Appointment.create_collection()
        return appointments_collection.find_one({'appointment_id': appointment_id})

    @staticmethod
    def get_by_case_id(case_id):
        """Get all appointments for a specific case"""
        appointments_collection = Appointment.create_collection()
        return list(appointments_collection.find({'case_id': case_id}))

    @staticmethod
    def update(appointment_id, update_data):
        """Update an appointment"""
        appointments_collection = Appointment.create_collection()
        update_data['updated_at'] = datetime.utcnow()
        appointments_collection.update_one(
            {'appointment_id': appointment_id},
            {'$set': update_data}
        )
        return Appointment.get_by_id(appointment_id)

    @staticmethod
    def delete(appointment_id):
        """Delete an appointment"""
        appointments_collection = Appointment.create_collection()
        appointments_collection.delete_one({'appointment_id': appointment_id})

    @staticmethod
    def search(query):
        """Search appointments by purpose or location"""
        appointments_collection = Appointment.create_collection()
        return list(appointments_collection.find({
            '$or': [
                {'purpose': {'$regex': query, '$options': 'i'}},
                {'location': {'$regex': query, '$options': 'i'}}
            ]
        })) 