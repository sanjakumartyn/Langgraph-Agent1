import asyncio
from rag_app.database import mongo_db

def test():
    if mongo_db is not None:
        print("CRM Statuses:", mongo_db["crm records"].distinct("status"))
        print("Proposal Statuses:", mongo_db["proposal documents"].distinct("proposalStatus"))
        print("Past Sales Statuses:", mongo_db["past_sales"].distinct("dealStatus"))
        print("Past Sales Sale Statuses:", mongo_db["past_sales"].distinct("saleStatus"))
        print("Product Categories:", mongo_db["products"].distinct("category"))

if __name__ == "__main__":
    test()
