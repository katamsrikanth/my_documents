from datetime import datetime
from models.base import BaseModel

class AttorneyCaseHistory(BaseModel):
    collection_name = 'attorney_case_history'

    @classmethod
    def log(cls, case_id, attorney_id, action, user=None):
        entry = {
            'case_id': str(case_id),
            'attorney_id': str(attorney_id),
            'action': action,  # 'associate' or 'deassociate'
            'user': user,
            'timestamp': datetime.utcnow()
        }
        cls.db[cls.collection_name].insert_one(entry)

    @classmethod
    def get_history_for_case(cls, case_id):
        return list(cls.db[cls.collection_name].find({'case_id': str(case_id)}).sort('timestamp', -1))

    @classmethod
    def get_history_for_attorney(cls, attorney_id):
        return list(cls.db[cls.collection_name].find({'attorney_id': str(attorney_id)}).sort('timestamp', -1)) 