import os
import sys
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from upstash_redis import Redis

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

# Load environment
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
UPSTASH_REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN")

# Initialize clients
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

redis_client: Optional[Redis] = None
if UPSTASH_REDIS_URL and UPSTASH_REDIS_TOKEN:
    redis_client = Redis(url=UPSTASH_REDIS_URL, token=UPSTASH_REDIS_TOKEN)


def init_supabase_tables():
    """
    Since DDL commands are restricted through the anonymous publishable API,
    this function performs verification or instructs the user on how tables should be defined.
    If the tables do not exist, they can be created in the Supabase Dashboard with columns:
    1. company_reports:
       - company_name: text (Primary Key)
       - website: text
       - report_data: text / jsonb
       - created_at: timestamp
    2. report_embeddings:
       - id: bigint (Primary Key, Identity)
       - company_name: text
       - chunk_text: text
       - embedding: jsonb / text
    """
    pass


def get_cached_report_supabase(company_name: str) -> Optional[Dict[str, Any]]:
    if supabase:
        try:
            response = supabase.table("company_reports").select("report_data").ilike("company_name", company_name.strip()).execute()
            if response.data:
                report_raw = response.data[0]["report_data"]
                if isinstance(report_raw, str):
                    return json.loads(report_raw)
                return report_raw
        except Exception as e:
            print(f"Supabase cache lookup error: {str(e)}")
    
    # Fallback to local SQLite database cache
    print("Falling back to local SQLite cache lookup...")
    try:
        return get_cached_report_sqlite(company_name)
    except Exception as sq_err:
        print(f"SQLite cache lookup error: {sq_err}")
    return None


def save_report_supabase(company_name: str, website: str, report_data: Dict[str, Any], chunks_with_embeddings: List[Dict[str, Any]]):
    # Always save to local SQLite cache first to guarantee availability
    try:
        save_report_sqlite(company_name, website, report_data, chunks_with_embeddings)
        print("Successfully saved report to local SQLite cache database.")
    except Exception as sq_err:
        print(f"SQLite cache save error: {sq_err}")

    if not supabase:
        return
    try:
        # Upsert report data
        supabase.table("company_reports").upsert({
            "company_name": company_name,
            "website": website,
            "report_data": json.dumps(report_data) if isinstance(report_data, dict) else report_data,
            "created_at": "now()"
        }).execute()
        
        # Clear old embeddings
        supabase.table("report_embeddings").delete().eq("company_name", company_name).execute()
        
        # Write new chunk embeddings
        records = []
        for chunk in chunks_with_embeddings:
            records.append({
                "company_name": company_name,
                "chunk_text": chunk["chunk_text"],
                "embedding": json.dumps(chunk["embedding"]) if isinstance(chunk["embedding"], list) else chunk["embedding"]
            })
            
        if records:
            supabase.table("report_embeddings").insert(records).execute()
            
    except Exception as e:
        print(f"Supabase cache save error (will use SQLite cache instead): {str(e)}")


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))
    magnitude1 = sum(x * x for x in v1) ** 0.5
    magnitude2 = sum(x * x for x in v2) ** 0.5
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def search_supabase_vectors(query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    results = []
    if supabase:
        try:
            response = supabase.table("report_embeddings").select("company_name, chunk_text, embedding").execute()
            for row in response.data:
                company_name = row["company_name"]
                chunk_text = row["chunk_text"]
                
                embedding_raw = row["embedding"]
                if isinstance(embedding_raw, str):
                    embedding = json.loads(embedding_raw)
                else:
                    embedding = embedding_raw
                    
                sim = cosine_similarity(query_embedding, embedding)
                results.append({
                    "company_name": company_name,
                    "chunk_text": chunk_text,
                    "similarity": sim
                })
                
            results.sort(key=lambda x: x["similarity"], reverse=True)
            results = results[:limit]
        except Exception as e:
            print(f"Supabase vector search error: {str(e)}")
            
    if not results:
        # Fallback to local SQLite vector search
        print("Falling back to local SQLite vector search...")
        try:
            return search_vector_store_sqlite(query_embedding, limit)
        except Exception as sq_err:
            print(f"SQLite vector search error: {sq_err}")
            
    return results
