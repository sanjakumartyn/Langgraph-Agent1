import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv("d:/Langgraph-Agent1/.env", override=True)
MONGODB_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

print("URI:", MONGODB_URI)
print("DB:", MONGO_DB_NAME)

try:
    client = MongoClient(MONGODB_URI)
    db = client[MONGO_DB_NAME]
    collections = db.list_collection_names()
    print("Collections:", collections)
    
    for c in collections:
        count = db[c].count_documents({})
        print(f"Collection '{c}' has {count} documents.")
except Exception as e:
    print("Error:", e)
