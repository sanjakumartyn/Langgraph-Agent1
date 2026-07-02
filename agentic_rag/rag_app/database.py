import os
import sys
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
import psycopg2
from upstash_redis import Redis
# pyrefly: ignore [missing-import]
from pymongo import MongoClient
from pymongo.server_api import ServerApi
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# Ensure root directory is in sys.path so we can import from app.database
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# pyrefly: ignore [missing-import]
from app.database import (
    get_cached_report as get_cached_report_sqlite,
    save_report as save_report_sqlite,
    search_vector_store as search_vector_store_sqlite
)

# Load environment — override=True ensures updated .env values take precedence
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path, override=True)

NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
UPSTASH_REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN")
MONGODB_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

# Initialize Neon PostgreSQL connection
def get_neon_connection(register=True):
    if not NEON_DATABASE_URL:
        return None
    try:
        conn = psycopg2.connect(NEON_DATABASE_URL)
        # Register pgvector only if requested (skip during init before extension exists)
        if register:
            try:
                register_vector(conn)
            except Exception as e:
                # If vector type doesn't exist yet, ignore registration failure
                pass
        return conn
    except Exception as e:
        print(f"Failed to connect to NeonDB: {str(e)}")
        return None

redis_client: Optional[Redis] = None
if UPSTASH_REDIS_URL and UPSTASH_REDIS_TOKEN:
    redis_client = Redis(url=str(UPSTASH_REDIS_URL), token=str(UPSTASH_REDIS_TOKEN))

mongo_client: Optional[MongoClient] = None
mongo_db = None
if MONGODB_URI and MONGO_DB_NAME:
    try:
        mongo_client = MongoClient(
            str(MONGODB_URI),
            server_api=ServerApi('1'),
            serverSelectionTimeoutMS=5000
        )
        if mongo_client is not None:
            # Quick ping to verify connection
            mongo_client.admin.command('ping')
            mongo_db = mongo_client[str(MONGO_DB_NAME)]
            print(f"MongoDB connected successfully. DB: {MONGO_DB_NAME}")
    except Exception as e:
        print(f"Failed to initialize MongoDB client in agentic_rag/rag_app/database: {str(e)}")

# Initialize Qdrant Client
qdrant_client: Optional[QdrantClient] = None
try:
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    if QDRANT_URL:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print("Qdrant cloud client initialized successfully")
    else:
        qdrant_client = QdrantClient(path=os.path.join(root_dir, "qdrant_db"))
        print("Qdrant client initialized successfully at ./qdrant_db")
except Exception as e:
    print(f"Failed to initialize Qdrant client: {str(e)}")


def init_neon_tables():
    conn = get_neon_connection(register=False)
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS company_reports (
                    company_name TEXT PRIMARY KEY,
                    website TEXT,
                    report_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS report_embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    company_name TEXT,
                    chunk_text TEXT,
                    embedding VECTOR(3072)
                );
            """)
            conn.commit()
            print("NeonDB tables verified/created successfully.")
    except Exception as e:
        print(f"Failed to initialize Neon tables: {e}")
    finally:
        conn.close()


def save_report_mongodb(company_name: str, website: str, report_data: Dict[str, Any], chunks_with_embeddings: List[Dict[str, Any]]):
    if mongo_db is not None:
        try:
            col_reports = mongo_db["company_reports"]
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
                "created_at": "now()"
            }
            col_reports.replace_one({"company_name": company_name}, report_doc, upsert=True)
            
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
                company_name = row.get("company_name", "Unknown")
                chunk_text = row.get("chunk_text", "")
                embedding = row.get("embedding")
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

def save_report_qdrant(company_name: str, chunks_with_embeddings: List[Dict[str, Any]]):
    if qdrant_client is None:
        return
    try:
        collection_name = "company_reports"
        if not chunks_with_embeddings:
            return
        vector_size = len(chunks_with_embeddings[0]["embedding"])
        
        collections = qdrant_client.get_collections()
        existing = [c.name for c in collections.collections]
        if collection_name not in existing:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection: {collection_name} with size {vector_size}")
            
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="company_name",
                        match=MatchValue(value=company_name)
                    )
                ]
            )
        )
        
        points = []
        for idx, chunk in enumerate(chunks_with_embeddings):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk["embedding"],
                    payload={
                        "company_name": company_name,
                        "chunk_text": chunk["chunk_text"]
                    }
                )
            )
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"Successfully saved {len(points)} embeddings to Qdrant for '{company_name}'.")
    except Exception as e:
        print(f"Qdrant cache save error: {str(e)}")

def search_qdrant_vectors(query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    if qdrant_client is None:
        return []
    try:
        collection_name = "company_reports"
        collections = qdrant_client.get_collections()
        existing = [c.name for c in collections.collections]
        if collection_name not in existing:
            return []
            
        response = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=limit
        )
        results = []
        for hit in response.points:
            payload = hit.payload or {}
            results.append({
                "company_name": payload.get("company_name"),
                "chunk_text": payload.get("chunk_text"),
                "similarity": hit.score
            })
        return results
    except Exception as e:
        print(f"Qdrant vector search error: {str(e)}")
    return []

def search_mongodb_internal(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    results = []
    if mongo_db is not None:
        try:
            collections = [c for c in mongo_db.list_collection_names() if not c.startswith("system.")]
            for col_name in collections:
                if col_name in ["report_embeddings", "company_reports"]:
                    continue
                col = mongo_db[col_name]
                common_fields = [
                    "company_name", "company", "name", "title", "text", "description", 
                    "summary", "content", "body", "client", "challenge", "solution", 
                    "results", "painPoint", "proposedSolution", "agenda", "discussionSummary",
                    "clientConcerns", "serviceName", "technology", "useCase", "dealName"
                ]
                or_clause = []
                for field in common_fields:
                    or_clause.append({field: {"$regex": query, "$options": "i"}})
                
                cursor = col.find({"$or": or_clause}).limit(limit)
                for doc in cursor:
                    doc_dict = dict(doc)
                    doc_id = str(doc_dict.pop("_id", ""))
                    doc_dict["_id"] = doc_id
                    doc_dict["_collection"] = col_name
                    results.append(doc_dict)
                    if len(results) >= limit:
                        break
                if len(results) >= limit:
                    break
        except Exception as e:
            print(f"MongoDB internal data search error: {str(e)}")
    return results

def get_cached_report_neon(company_name: str) -> Optional[Dict[str, Any]]:
    conn = get_neon_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT report_data FROM company_reports WHERE company_name ILIKE %s",
                (company_name.strip(),)
            )
            row = cur.fetchone()
            if row and row[0]:
                data = row[0]
                if isinstance(data, str):
                    return json.loads(data)
                return data
    except Exception as e:
        print(f"Neon cache lookup error: {str(e)}")
    finally:
        conn.close()
    
    # Try MongoDB cache
    print("Trying MongoDB cache lookup...")
    mongo_report = get_cached_report_mongodb(company_name)
    if mongo_report:
        return mongo_report
    
    # Fallback to local SQLite database cache
    print("Falling back to local SQLite cache lookup...")
    try:
        return get_cached_report_sqlite(company_name)
    except Exception as sq_err:
        print(f"SQLite cache lookup error: {sq_err}")
    return None

def save_report_neon(company_name: str, website: str, report_data: Dict[str, Any], chunks_with_embeddings: List[Dict[str, Any]]):
    # Always save to local fallback first
    save_report_sqlite(company_name, website, report_data, chunks_with_embeddings)
    save_report_mongodb(company_name, website, report_data, chunks_with_embeddings)
    save_report_qdrant(company_name, chunks_with_embeddings)

    conn = get_neon_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            # Upsert company report
            cur.execute(
                """
                INSERT INTO company_reports (company_name, website, report_data)
                VALUES (%s, %s, %s)
                ON CONFLICT (company_name) DO UPDATE SET
                    website = EXCLUDED.website,
                    report_data = EXCLUDED.report_data,
                    created_at = CURRENT_TIMESTAMP;
                """,
                (company_name, website, json.dumps(report_data))
            )

            # Re-insert embeddings
            cur.execute("DELETE FROM report_embeddings WHERE company_name = %s", (company_name,))
            
            if chunks_with_embeddings:
                records = []
                for chunk in chunks_with_embeddings:
                    # chunk["embedding"] is a list of floats
                    records.append((company_name, chunk["chunk_text"], chunk["embedding"]))
                
                from psycopg2.extras import execute_values
                execute_values(
                    cur,
                    "INSERT INTO report_embeddings (company_name, chunk_text, embedding) VALUES %s",
                    records
                )
            conn.commit()
        print(f"Successfully saved report and embeddings to Neon cache for '{company_name}'.")
    except Exception as e:
        conn.rollback()
        print(f"Neon cache save error: {str(e)}")
    finally:
        conn.close()

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))
    magnitude1 = sum(x * x for x in v1) ** 0.5
    magnitude2 = sum(x * x for x in v2) ** 0.5
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def search_neon_vectors(query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    # Try Qdrant vector search first
    print("Trying Qdrant vector search...")
    results = search_qdrant_vectors(query_embedding, limit)
    if results:
        return results
        
    conn = get_neon_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT company_name, chunk_text, 1 - (embedding <=> %s::vector) AS similarity
                    FROM report_embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_embedding, query_embedding, limit)
                )
                rows = cur.fetchall()
                for row in rows:
                    results.append({
                        "company_name": row[0],
                        "chunk_text": row[1],
                        "similarity": float(row[2])
                    })
                return results
        except Exception as e:
            print(f"Neon vector search error: {str(e)}")
        finally:
            conn.close()
            
    if not results:
        print("Trying MongoDB vector search...")
        try:
            results = search_vector_store_mongodb(query_embedding, limit)
        except Exception as mongo_err:
            print(f"MongoDB vector search error: {mongo_err}")
            
    if not results:
        print("Falling back to local SQLite vector search...")
        try:
            return search_vector_store_sqlite(query_embedding, limit)
        except Exception as sq_err:
            print(f"SQLite vector search error: {sq_err}")
            
    return results

def get_dynamic_dashboard_metrics() -> Dict[str, Any]:
    """Dynamically aggregates real-time metrics from MongoDB for the dashboard."""
    if mongo_db is None:
        return {"error": "MongoDB not connected"}
        
    try:
        # 1. Basic Counts
        total_products = mongo_db["products"].count_documents({})
        case_studies = mongo_db["case studies"].count_documents({})
        
        # Active Pipeline (Not won or lost)
        active_opps = mongo_db["crm records"].count_documents({
            "status": {"$in": ["Evaluation", "Interested", "Proposal Sent", "Qualified", "Negotiation"]}
        })
        
        # Win Rate
        total_sales = mongo_db["past_sales"].count_documents({})
        won_sales = mongo_db["past_sales"].count_documents({"dealStatus": "Won"})
        success_rate = int((won_sales / total_sales * 100)) if total_sales > 0 else 0
        
        # 2. Portfolio Breakdown
        portfolio_pipeline = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        portfolio_aggs = list(mongo_db["products"].aggregate(portfolio_pipeline))
        
        portfolio = []
        for p in portfolio_aggs:
            cat_name = p["_id"] or "Unknown"
            count = p["count"]
            percentage = int((count / total_products * 100)) if total_products > 0 else 0
            portfolio.append({"name": cat_name, "count": count, "percentage": percentage})
            
        # 3. Top Solutions
        solutions_pipeline = [
            {"$group": {
                "_id": "$serviceName", 
                "opportunities": {"$sum": 1},
                "value": {"$sum": "$estimatedCost"}
            }},
            {"$sort": {"value": -1}},
            {"$limit": 5}
        ]
        solutions_aggs = list(mongo_db["proposal documents"].aggregate(solutions_pipeline))
        
        top_solutions = []
        for s in solutions_aggs:
            srv_name = s["_id"] or "Unknown Service"
            top_solutions.append({
                "name": srv_name,
                "opportunities": s["opportunities"],
                "value": s["value"] or 0
            })
            
        return {
            "success": True,
            "company": "Global Enterprise Dashboard",
            "tagline": "Dynamic analysis powered by real MongoDB CRM & Product data",
            "stats": {
                "total_products": total_products,
                "case_studies": case_studies,
                "active_opportunities": active_opps,
                "success_rate": success_rate
            },
            "portfolio": portfolio,
            "top_solutions": top_solutions
        }
    except Exception as e:
        print(f"Error aggregating metrics: {e}")
        return {"error": str(e)}
