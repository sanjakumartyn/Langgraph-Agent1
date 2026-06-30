import sqlite3
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

mongo_client: Optional[MongoClient] = None
mongo_db = None
if MONGODB_URI and MONGO_DB_NAME:
    try:
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        mongo_db = mongo_client[MONGO_DB_NAME]
    except Exception as e:
        print(f"Failed to initialize MongoDB client in app/database: {str(e)}")

DB_PATH = os.path.join(os.path.dirname(__file__), "company_reports.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create company_reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS company_reports (
            company_name TEXT PRIMARY KEY,
            website TEXT,
            report_data TEXT,
            created_at TEXT
        )
    """)
    
    # Create report_embeddings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS report_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            chunk_text TEXT,
            embedding TEXT,
            FOREIGN KEY(company_name) REFERENCES company_reports(company_name) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()

def save_report_mongodb(company_name: str, website: str, report_data: Dict[str, Any], chunks_with_embeddings: List[Dict[str, Any]]):
    if mongo_db is not None:
        try:
            col_reports = mongo_db["company_reports"]
            # Ensure report_data is saved as dict
            doc_data = report_data
            if isinstance(doc_data, str):
                try:
                    doc_data = json.loads(doc_data)
                except Exception:
                    pass
            report_doc = {
                "company_name": company_name,
                "website": website,
                "report_data": doc_data,
                "created_at": datetime.utcnow().isoformat()
            }
            col_reports.replace_one({"company_name": company_name}, report_doc, upsert=True)
            
            # Clear old embeddings and save new ones
            col_embeddings = mongo_db["report_embeddings"]
            col_embeddings.delete_many({"company_name": company_name})
            
            if chunks_with_embeddings:
                records = []
                for chunk in chunks_with_embeddings:
                    records.append({
                        "company_name": company_name,
                        "chunk_text": chunk["chunk_text"],
                        "embedding": chunk["embedding"]
                    })
                col_embeddings.insert_many(records)
            print(f"Successfully saved report and embeddings to MongoDB cache for '{company_name}'.")
        except Exception as e:
            print(f"MongoDB cache save error: {str(e)}")

def get_cached_report_mongodb(company_name: str) -> Optional[Dict[str, Any]]:
    if mongo_db is not None:
        try:
            col = mongo_db["company_reports"]
            doc = col.find_one({"company_name": {"$regex": f"^{company_name.strip()}$", "$options": "i"}})
            if doc:
                report_data = doc.get("report_data")
                if isinstance(report_data, str):
                    return json.loads(report_data)
                return report_data
        except Exception as e:
            print(f"MongoDB cache lookup error: {str(e)}")
    return None

def search_vector_store_mongodb(query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    if mongo_db is not None:
        try:
            col = mongo_db["report_embeddings"]
            rows = col.find({}, {"company_name": 1, "chunk_text": 1, "embedding": 1})
            results = []
            for row in rows:
                company_name = row["company_name"]
                chunk_text = row["chunk_text"]
                embedding = row["embedding"]
                if embedding:
                    similarity = cosine_similarity(query_embedding, embedding)
                    results.append({
                        "company_name": company_name,
                        "chunk_text": chunk_text,
                        "similarity": similarity
                    })
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]
        except Exception as e:
            print(f"MongoDB vector search error: {str(e)}")
    return []

def save_report(company_name: str, website: str, report_data: Dict[str, Any], chunks_with_embeddings: List[Dict[str, Any]]):
    # Save to MongoDB first
    save_report_mongodb(company_name, website, report_data, chunks_with_embeddings)

    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Upsert company report
    now = datetime.utcnow().isoformat()
    cursor.execute(
        """
        INSERT INTO company_reports (company_name, website, report_data, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(company_name) DO UPDATE SET
            website=excluded.website,
            report_data=excluded.report_data,
            created_at=excluded.created_at
        """,
        (company_name, website, json.dumps(report_data), now)
    )
    
    # Clear old embeddings for this company
    cursor.execute("DELETE FROM report_embeddings WHERE company_name = ?", (company_name,))
    
    # Insert new embeddings
    for item in chunks_with_embeddings:
        chunk_text = item["chunk_text"]
        embedding = item["embedding"]
        cursor.execute(
            """
            INSERT INTO report_embeddings (company_name, chunk_text, embedding)
            VALUES (?, ?, ?)
            """,
            (company_name, chunk_text, json.dumps(embedding))
        )
        
    conn.commit()
    conn.close()

def get_cached_report(company_name: str) -> Optional[Dict[str, Any]]:
    # Try MongoDB cache first
    mongo_report = get_cached_report_mongodb(company_name)
    if mongo_report:
        return mongo_report

    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT report_data FROM company_reports WHERE LOWER(company_name) = LOWER(?)", (company_name.strip(),))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        return json.loads(row["report_data"])
    return None

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))
    magnitude1 = sum(x * x for x in v1) ** 0.5
    magnitude2 = sum(x * x for x in v2) ** 0.5
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def search_vector_store(query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    # Try MongoDB vector search first
    mongo_results = search_vector_store_mongodb(query_embedding, limit)
    if mongo_results:
        return mongo_results

    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT company_name, chunk_text, embedding FROM report_embeddings")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        company_name = row["company_name"]
        chunk_text = row["chunk_text"]
        try:
            embedding = json.loads(row["embedding"])
            similarity = cosine_similarity(query_embedding, embedding)
            results.append({
                "company_name": company_name,
                "chunk_text": chunk_text,
                "similarity": similarity
            })
        except Exception:
            continue
            
    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]
