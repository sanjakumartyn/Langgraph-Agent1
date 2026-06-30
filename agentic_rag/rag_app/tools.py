import os
import sys
import json
import httpx
from typing import List, Dict, Any, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embeddings():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    if provider == "mistral":
        from langchain_mistralai import MistralAIEmbeddings
        return MistralAIEmbeddings(
            model="mistral-embed",
            api_key=os.getenv("MISTRAL_API_KEY")
        )
    else:
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

# Ensure root directory is in sys.path so we can import from root app
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app.graph import run_company_agent
from .database import search_neon_vectors, get_cached_report_neon, save_report_neon, search_mongodb_internal

async def safe_get_async(url: str, params=None, headers=None, timeout=15):
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": str(e)}


def chunk_dossier(report: Dict[str, Any]) -> List[str]:
    chunks = []
    company_name = report.get("company_name", "Unknown Company")
    
    # 1. Company Overview
    summary = report.get("company_summary") or ""
    industry = report.get("industry") or "Unknown"
    website = report.get("website") or ""
    if summary:
        chunks.append(f"Company: {company_name}\nWebsite: {website}\nIndustry: {industry}\nOverview: {summary}")
        
    # 2. Products or Services
    products = report.get("products_or_services") or []
    if products:
        products_str = "\n".join(f"- {p}" if isinstance(p, str) else json.dumps(p) for p in products)
        chunks.append(f"Company: {company_name}\nProducts and Services:\n{products_str}")
        
    # 3. Leadership and Contacts
    leadership = report.get("leadership_or_contact_signals") or []
    if leadership:
        leadership_str = "\n".join(f"- {l}" if isinstance(l, str) else json.dumps(l) for l in leadership)
        chunks.append(f"Company: {company_name}\nLeadership & Contacts:\n{leadership_str}")
        
    # 4. Locations
    locations = report.get("locations") or []
    if locations:
        locations_str = ", ".join(locations) if isinstance(locations, list) else str(locations)
        chunks.append(f"Company: {company_name}\nLocations: {locations_str}")
        
    # 5. Business Priorities & Pain Points
    priorities = report.get("business_priorities") or []
    pain_points = report.get("pain_points") or []
    if priorities or pain_points:
        p_str = "\n".join(f"- Priority: {p}" for p in priorities)
        pp_str = "\n".join(f"- Pain Point: {p}" for p in pain_points)
        chunks.append(f"Company: {company_name}\nBusiness Priorities:\n{p_str}\nPain Points:\n{pp_str}")
        
    # 6. Technology Signals
    tech = report.get("technology_signals") or []
    if tech:
        tech_str = "\n".join(f"- {t}" if isinstance(t, str) else json.dumps(t) for t in tech)
        chunks.append(f"Company: {company_name}\nTechnology Signals:\n{tech_str}")
        
    # 7. Market News
    news = report.get("market_news") or []
    if news:
        news_str = "\n".join(f"- {n}" if isinstance(n, str) else json.dumps(n) for n in news)
        chunks.append(f"Company: {company_name}\nMarket News & Updates:\n{news_str}")
        
    # 8. History
    history = report.get("company_history") or []
    if history:
        history_str = "\n".join(f"- {h}" if isinstance(h, str) else json.dumps(h) for h in history)
        chunks.append(f"Company: {company_name}\nHistory & Milestones:\n{history_str}")
        
    return [c.strip() for c in chunks if c.strip()]


async def ingest_company_to_neon(company_name: str, company_website: Optional[str] = None) -> Dict[str, Any]:
    """
    Crawls web sources using the Company Intelligence Agent, chunks the report,
    generates embeddings, and caches the results in Supabase.
    """
    # 0. Mock Ingestion Fallback for local UI and workflow testing without API limits
    if company_name.lower().strip() == "mock":
        report_data = {
            "company_name": "MockCorp AI Solutions",
            "website": "https://www.mockcorp.ai",
            "industry": "Business Intelligence & MLOps",
            "company_summary": "MockCorp is a global pioneer in developing agentic workflow solutions and enterprise RAG systems. They specialize in integrating large language models with existing databases to streamline corporate decision making.",
            "products_or_services": [
                "Agentic Customer Support Chatbots",
                "Predictive Supply Chain Copilots",
                "Cognitive Document Discovery RAG",
                "MLOps Pipeline Scaling Consulting"
            ],
            "leadership_or_contact_signals": [
                "Jane Doe (Chief Technology Officer) - Active on AI regulations",
                "John Smith (Director of Sales) - Looking for cognitive partners"
            ],
            "locations": ["San Francisco, CA", "Bengaluru, India"],
            "business_priorities": [
                "Reduce operational latency in customer query processing",
                "Migrate legacy CRM databases to vector-indexed cloud systems"
            ],
            "pain_points": [
                "High API billing costs for commercial LLMs",
                "Hallucinations in custom medical record discovery system",
                "Data compliance restrictions with off-premise training data"
            ],
            "technology_signals": [
                "Uses Python, FastAPI, LangGraph, and PostgreSQL",
                "Currently scaling vector infrastructure using Pinecone"
            ],
            "market_news": [
                "MockCorp raises $50M in Series B funding to expand MLOps services",
                "Announces strategic partnership with major healthcare providers"
            ],
            "company_history": [
                "Founded in 2024 by alumni of major research labs",
                "Launched RAG platform in late 2024 with 10 enterprise clients"
            ],
            "source_evidence": [],
            "confidence_score": 100,
            "missing_data": []
        }
        
        # Chunk the report
        text_chunks = chunk_dossier(report_data)
        
        # Generate dummy embeddings based on provider dimension (1024 for Mistral, 3072 for Gemini)
        provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
        dim = 1024 if provider == "mistral" else 3072
        chunks_with_embeddings = []
        for chunk in text_chunks:
            chunks_with_embeddings.append({
                "chunk_text": chunk,
                "embedding": [0.0] * dim
            })
            
        # Cache to Neon
        save_report_neon(
            company_name="MockCorp AI Solutions",
            website="https://www.mockcorp.ai",
            report_data=report_data,
            chunks_with_embeddings=chunks_with_embeddings
        )
        return report_data

    # 1. Run the Company Intelligence Agent in a dedicated thread/loop to bypass Windows Uvicorn event loop restrictions
    import asyncio
    
    def _run_agent_sync():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_company_agent(company_name, company_website))
        finally:
            loop.close()
            
    result = await asyncio.to_thread(_run_agent_sync)
    
    report_data = result.get("final_result", {})
    if not report_data or not report_data.get("company_summary"):
        # If scraper returned an empty report or failed, construct a basic profile
        report_data = {
            "company_name": company_name,
            "website": company_website or f"https://www.{company_name.lower().replace(' ', '')}.com",
            "company_summary": f"Real-time search profile for {company_name}.",
            "industry": "Unknown",
            "products_or_services": [],
            "locations": [],
            "business_priorities": [],
            "pain_points": [],
            "technology_signals": [],
            "market_news": [],
            "company_history": [],
            "source_evidence": [],
            "confidence_score": 10,
            "missing_data": ["Full scraping profile could not be completed."]
        }

    # 2. Chunk the report
    text_chunks = chunk_dossier(report_data)
    
    # 3. Generate embeddings
    embeddings_model = get_embeddings()
    
    chunks_with_embeddings = []
    for chunk in text_chunks:
        try:
            emb = embeddings_model.embed_query(chunk)
            chunks_with_embeddings.append({
                "chunk_text": chunk,
                "embedding": emb
            })
        except Exception as e:
            print(f"Error embedding chunk for {company_name}: {str(e)}")
            
    # 4. Save to NeonDB
    save_report_neon(
        company_name=company_name,
        website=report_data.get("website", ""),
        report_data=report_data,
        chunks_with_embeddings=chunks_with_embeddings
    )
    
    return report_data


async def vector_search_tool(query: str, limit: int = 3) -> str:
    """Search the vector database of stored company reports for matching context."""
    try:
        # Embed query
        embeddings_model = get_embeddings()
        query_emb = await embeddings_model.aembed_query(query)
        # Query Neon vector index
        raw_results = search_neon_vectors(query_emb, limit=limit)
        
        if not raw_results:
            return "No matching records found in the vector database."
            
        formatted = []
        for r in raw_results:
            formatted.append(
                f"--- Result (Similarity: {round(r['similarity'], 4)}) ---\n"
                f"Company: {r['company_name']}\n"
                f"Excerpt: {r['chunk_text']}\n"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"Error executing vector search tool: {str(e)}"


async def get_company_dossier_tool(company_name: str) -> str:
    """Retrieve the full intelligence report dossier for a specific company from the database.
    If not cached, trigger real-time scraping and ingestion."""
    report = get_cached_report_neon(company_name)
    if not report:
        print(f"No stored intelligence dossier found for '{company_name}' in the database. Triggering real-time ingestion...")
        try:
            report = await ingest_company_to_neon(company_name)
        except Exception as e:
            return f"No stored intelligence dossier found for '{company_name}', and automatic ingestion failed: {str(e)}"
    return json.dumps(report, indent=2)


async def live_news_lookup_tool(company_name: str) -> str:
    """Fetch live news from NewsAPI and MediaStack for a company to get updated context."""
    newsapi_key = os.getenv("NEWS_API_KEY")
    mediastack_key = os.getenv("MEDIASTACK_API_KEY")
    
    articles = []
    
    # 1. Fetch NewsAPI
    if newsapi_key:
        status, data = await safe_get_async(
            "https://newsapi.org/v2/everything",
            params={
                "q": f'"{company_name}"',
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 5,
                "apiKey": newsapi_key
            }
        )
        if status == 200 and isinstance(data, dict):
            for item in data.get("articles", []):
                if isinstance(item, dict):
                    articles.append({
                        "title": item.get("title"),
                        "source": item.get("source", {}).get("name") if isinstance(item.get("source"), dict) else None,
                        "url": item.get("url"),
                        "description": item.get("description")
                    })
                
    # 2. Fetch MediaStack
    if mediastack_key:
        status, data = await safe_get_async(
            "http://api.mediastack.com/v1/news",
            params={
                "access_key": mediastack_key,
                "keywords": company_name,
                "languages": "en",
                "limit": 5
            }
        )
        if status == 200 and isinstance(data, dict) and "data" in data:
            for item in data["data"]:
                if isinstance(item, dict):
                    articles.append({
                        "title": item.get("title"),
                        "source": item.get("source"),
                        "url": item.get("url"),
                        "description": item.get("description")
                    })
                
    if not articles:
        # Fallback to a quick Wikipedia lookup
        status, data = await safe_get_async(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": company_name,
                "format": "json",
            }
        )
        if status == 200 and isinstance(data, dict):
            query_data = data.get("query")
            if isinstance(query_data, dict):
                search_results = query_data.get("search")
                if isinstance(search_results, list) and len(search_results) > 0:
                    first_result = search_results[0]
                    if isinstance(first_result, dict) and "title" in first_result:
                        title = first_result["title"]
            status, summary = await safe_get_async(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
            )
            if status == 200 and summary:
                return f"Wikipedia Summary for '{company_name}':\n{summary.get('extract')}"
        return f"No live search results or news found for '{company_name}'."
        
    # Deduplicate and format articles
    seen = set()
    formatted = []
    for a in articles:
        url = a.get("url")
        if url and url not in seen:
            seen.add(url)
            formatted.append(f"- {a['title']} ({a['source']}): {a['description'] or ''}")
            
    return "\n".join(formatted[:6])


async def search_mongodb_tool(query: str, limit: int = 5) -> str:
    """Search the company's internal MongoDB database collections for documents/information matching the query."""
    try:
        results = search_mongodb_internal(query, limit=limit)
        if not results:
            return f"No matching internal company documents found in MongoDB for query '{query}'."
            
        formatted = []
        for r in results:
            col = r.pop("_collection", "unknown")
            formatted.append(
                f"--- Result from collection '{col}' ---\n"
                f"{json.dumps(r, indent=2)}\n"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"Error executing search_mongodb_tool: {str(e)}"

