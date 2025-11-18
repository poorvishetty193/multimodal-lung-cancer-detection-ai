from backend.services.db_service import get_db
from bson import ObjectId

db = get_db()

class UserService:

    def create_patient(self, data):
        result = db.patients.insert_one(data)
        return str(result.inserted_id)

    def get_patient(self, pid):
        patient = db.patients.find_one({"_id": ObjectId(pid)})
        if patient:
            patient["id"] = str(patient["_id"])
            del patient["_id"]
        return patient

user_service = UserService()
