"""
More detailed MongoDB aggregation queries to power the dashboard.
"""
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

MONGODB_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

from pymongo import MongoClient
from pymongo.server_api import ServerApi

mongo_client = MongoClient(str(MONGODB_URI), server_api=ServerApi('1'), serverSelectionTimeoutMS=8000)
mongo_client.admin.command('ping')
mongo_db = mongo_client[str(MONGO_DB_NAME)]

print("=== DASHBOARD METRICS ===")

# Stats
total_products = mongo_db["products"].count_documents({})
case_studies = mongo_db["case studies"].count_documents({})
active_opps = mongo_db["crm records"].count_documents({"status": {"$in": ["Evaluation","Interested","Proposal Sent","Qualified","Negotiation"]}})
total_sales = mongo_db["past_sales"].count_documents({})
won_sales = mongo_db["past_sales"].count_documents({"dealStatus": "Won"})
success_rate = int((won_sales / total_sales * 100)) if total_sales > 0 else 0

print(f"total_products: {total_products}")
print(f"case_studies: {case_studies}")
print(f"active_opps: {active_opps}")
print(f"total_sales: {total_sales}, won: {won_sales}, success_rate: {success_rate}%")

# Portfolio breakdown
print("\n=== PRODUCT PORTFOLIO (by category) ===")
portfolio_pipeline = [
    {"$group": {"_id": "$category", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
for p in mongo_db["products"].aggregate(portfolio_pipeline):
    pct = int(p["count"] / total_products * 100)
    print(f"  {p['_id']}: {p['count']} ({pct}%)")

# Top solutions by proposal value
print("\n=== TOP SOLUTIONS (by proposal value) ===")
solutions_pipeline = [
    {"$group": {
        "_id": "$serviceName",
        "opportunities": {"$sum": 1},
        "value": {"$sum": "$estimatedCost"}
    }},
    {"$sort": {"value": -1}},
    {"$limit": 8}
]
for s in mongo_db["proposal documents"].aggregate(solutions_pipeline):
    print(f"  {s['_id']}: {s['opportunities']} opps, ${s['value']:,}")

# CRM pipeline stages
print("\n=== CRM PIPELINE (by status) ===")
crm_pipeline = [
    {"$group": {"_id": "$status", "count": {"$sum": 1}, "value": {"$sum": 0}}},
    {"$sort": {"count": -1}}
]
for c in mongo_db["crm records"].aggregate(crm_pipeline):
    print(f"  {c['_id']}: {c['count']} records")

# Sales breakdown
print("\n=== PAST SALES STATUS ===")
for s in mongo_db["past_sales"].find({}, {"_id":0}):
    print(f"  {s}")

# Proposal statuses
print("\n=== PROPOSAL STATUSES ===")
prop_statuses = mongo_db["proposal documents"].distinct("proposalStatus")
print(f"  Proposal statuses: {prop_statuses}")

prop_status_pipeline = [
    {"$group": {"_id": "$proposalStatus", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
for p in mongo_db["proposal documents"].aggregate(prop_status_pipeline):
    print(f"  {p['_id']}: {p['count']}")

# Proposal total value
print("\n=== TOTAL PROPOSAL VALUE ===")
total_val_pipeline = [
    {"$group": {"_id": None, "total": {"$sum": "$estimatedCost"}}}
]
for t in mongo_db["proposal documents"].aggregate(total_val_pipeline):
    print(f"  Total proposal value: ${t['total']:,}")

# CRM industries
print("\n=== CRM by INDUSTRY ===")
ind_pipeline = [
    {"$group": {"_id": "$industry", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
for i in mongo_db["crm records"].aggregate(ind_pipeline):
    print(f"  {i['_id']}: {i['count']}")

# Meeting types
print("\n=== MEETING TYPES ===")
mtg_pipeline = [
    {"$group": {"_id": "$meetingType", "count": {"$sum": 1}}}
]
for m in mongo_db["past meeting records"].aggregate(mtg_pipeline):
    print(f"  {m['_id']}: {m['count']}")

# Meeting sentiments
print("\n=== MEETING SENTIMENTS ===")
sent_pipeline = [
    {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}}
]
for s in mongo_db["past meeting records"].aggregate(sent_pipeline):
    print(f"  {s['_id']}: {s['count']}")

# Case study industries
print("\n=== CASE STUDIES by INDUSTRY ===")
cs_pipeline = [
    {"$group": {"_id": "$industry", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 8}
]
for c in mongo_db["case studies"].aggregate(cs_pipeline):
    print(f"  {c['_id']}: {c['count']}")

# Sample CRM records
print("\n=== CRM RECORDS SAMPLE ===")
for r in mongo_db["crm records"].find({}, {"_id": 0, "crmId": 1, "company": 1, "status": 1, "dealValue": 1, "probability": 1, "industry": 1, "salesStage": 1, "owner": 1}):
    print(f"  {r}")
