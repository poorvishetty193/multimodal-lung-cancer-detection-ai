from pymongo import MongoClient

MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "lung_ai"

client = MongoClient(MONGO_URL)
db = client[DB_NAME]

def get_db():
    return db
