"""
Script to inspect actual MongoDB and Qdrant data structure.
Run from agentic_rag directory.
"""
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

MONGODB_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print("=" * 60)
print("MONGODB INSPECTION")
print("=" * 60)

from pymongo import MongoClient
from pymongo.server_api import ServerApi

try:
    mongo_client = MongoClient(str(MONGODB_URI), server_api=ServerApi('1'), serverSelectionTimeoutMS=8000)
    mongo_client.admin.command('ping')
    mongo_db = mongo_client[str(MONGO_DB_NAME)]
    print(f"[OK] MongoDB connected: {MONGO_DB_NAME}")

    collections = mongo_db.list_collection_names()
    print(f"\nCollections ({len(collections)}):")
    for col_name in sorted(collections):
        col = mongo_db[col_name]
        count = col.count_documents({})
        print(f"  [{count} docs] {col_name}")

    print("\n--- Sample docs from each collection ---")
    for col_name in sorted(collections):
        col = mongo_db[col_name]
        sample = col.find_one({}, {"_id": 0})
        if sample:
            keys = list(sample.keys())
            print(f"\n[{col_name}] fields: {keys}")
            # Show first sample (compact)
            sample_str = json.dumps(sample, default=str, indent=2)
            if len(sample_str) > 800:
                sample_str = sample_str[:800] + "..."
            print(sample_str)

    print("\n--- Aggregation metrics ---")
    # CRM distinct statuses
    try:
        crm_statuses = mongo_db["crm records"].distinct("status")
        print(f"CRM statuses: {crm_statuses}")
    except: pass

    # Past sales distinct deal statuses
    try:
        ps_statuses = mongo_db["past_sales"].distinct("dealStatus")
        print(f"Past sales dealStatus: {ps_statuses}")
        total_sales = mongo_db["past_sales"].count_documents({})
        won_sales = mongo_db["past_sales"].count_documents({"dealStatus": "Won"})
        print(f"Total sales: {total_sales}, Won: {won_sales}")
    except: pass

    # Products categories
    try:
        prod_cats = mongo_db["products"].distinct("category")
        print(f"Product categories: {prod_cats}")
        total_products = mongo_db["products"].count_documents({})
        print(f"Total products: {total_products}")
    except: pass

    # Case studies count
    try:
        cs_count = mongo_db["case studies"].count_documents({})
        print(f"Case studies: {cs_count}")
    except: pass

    # Proposal fields
    try:
        prop_sample = mongo_db["proposal documents"].find_one({}, {"_id": 0})
        if prop_sample:
            print(f"Proposal fields: {list(prop_sample.keys())}")
    except: pass

    # Past meetings fields
    try:
        mtg_sample = mongo_db["past meeting records"].find_one({}, {"_id": 0})
        if mtg_sample:
            print(f"Meeting fields: {list(mtg_sample.keys())}")
    except: pass

except Exception as e:
    print(f"[ERR] MongoDB error: {e}")

print("\n" + "=" * 60)
print("QDRANT INSPECTION")
print("=" * 60)

try:
    from qdrant_client import QdrantClient
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collections = qdrant_client.get_collections()
    print(f"[OK] Qdrant connected")
    print(f"Collections: {[c.name for c in collections.collections]}")

    for col in collections.collections:
        info = qdrant_client.get_collection(col.name)
        count = info.points_count
        vec_size = info.config.params.vectors
        print(f"\n[{col.name}] points: {count}, vector_config: {vec_size}")

        # Sample a point
        results = qdrant_client.query_points(
            collection_name=col.name,
            query=[0.0] * 3072,
            limit=3
        )
        for hit in results.points[:2]:
            payload = hit.payload or {}
            payload_keys = list(payload.keys())
            print(f"  Sample point payload keys: {payload_keys}")
            # Show payload without the big vector
            print(f"  company_name: {payload.get('company_name')}")
            chunk_preview = str(payload.get('chunk_text', ''))[:200]
            print(f"  chunk_text preview: {chunk_preview}")
            break

except Exception as e:
    print(f"[ERR] Qdrant error: {e}")
