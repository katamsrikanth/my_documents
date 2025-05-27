from datetime import datetime
from models.base import BaseModel

class AttorneyFeedback(BaseModel):
    collection_name = 'attorney_feedback'

    @classmethod
    def submit_feedback(cls, attorney_id, client_name, feedback, rating=None):
        entry = {
            'attorney_id': str(attorney_id),
            'client_name': client_name,
            'feedback': feedback,
            'rating': rating,
            'timestamp': datetime.utcnow(),
            'approved': False  # Admin can approve for display
        }
        cls.db[cls.collection_name].insert_one(entry)

    @classmethod
    def get_feedback_for_attorney(cls, attorney_id, approved_only=True):
        query = {'attorney_id': str(attorney_id)}
        if approved_only:
            query['approved'] = True
        return list(cls.db[cls.collection_name].find(query).sort('timestamp', -1))

    @classmethod
    def approve_feedback(cls, feedback_id):
        from bson import ObjectId
        cls.db[cls.collection_name].update_one({'_id': ObjectId(feedback_id)}, {'$set': {'approved': True}}) 