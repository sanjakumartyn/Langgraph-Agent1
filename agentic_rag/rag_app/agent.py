import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from .tools import (
    vector_search_tool,
    get_company_dossier_tool,
    live_news_lookup_tool,
    search_mongodb_tool,
    get_ai_products_from_mongodb,
    get_internal_company_context,
)
from .database import get_cached_report_neon


# ─── LLM Provider ────────────────────────────────────────────────────────────
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
            temperature=0,
            api_key=os.getenv("MISTRAL_API_KEY")
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

llm = get_llm()


# ─── State ────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    query: str
    entities: Dict[str, Any]
    is_internal: bool
    is_fast_mode: bool
    steps_taken: List[str]
    retrieved_context: List[str]
    sources_used: List[Dict[str, str]]
    next_action: str
    current_param: str
    final_answer: str
    # New: carries real product + internal data for analysis
    ai_products: List[Dict[str, Any]]
    internal_context: Dict[str, Any]


# ─── Utilities ────────────────────────────────────────────────────────────────
async def ainvoke_with_retry(prompt: str, max_retries: int = 3) -> Any:
    """Wrapper to automatically retry LLM calls on 429 Rate Limits with Mistral fallback."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return await llm.ainvoke(prompt)
        except Exception as e:
            last_error = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt == max_retries - 1:
                    break
                await asyncio.sleep(2 ** attempt * 2)
            else:
                break
                
    # Fallback to Mistral AI if primary fails
    print("Primary LLM failed. Falling back to Mistral AI...")
    try:
        from langchain_mistralai import ChatMistralAI
        mistral_fallback = ChatMistralAI(
            model=os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
            temperature=0,
            api_key=os.getenv("MISTRAL_API_KEY")
        )
        return await mistral_fallback.ainvoke(prompt)
    except Exception as fallback_e:
        print(f"Mistral fallback also failed: {fallback_e}")
        if last_error:
            raise last_error
        raise fallback_e


def clean_json_response(content: str) -> str:
    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return content.strip()


def _build_guaranteed_fallback(primary_company: str, ai_products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a guaranteed non-empty fallback response using our real AI products.
    Used when LLM fails entirely.
    """
    # Pick top 3 AI products for solution mapping
    top_products = ai_products[:3] if ai_products else [
        {"serviceName": "AI Sales Intelligence Platform", "pricing": 52000, "unit": "Annual License",
         "description": "Enterprise-grade AI sales intelligence platform.", "category": "AI & Generative AI"},
        {"serviceName": "Enterprise Knowledge Assistant", "pricing": 107000, "unit": "Monthly Subscription",
         "description": "AI-powered enterprise knowledge management.", "category": "AI & Generative AI"},
        {"serviceName": "Predictive Analytics Engine", "pricing": 40000, "unit": "Project",
         "description": "ML-powered predictive analytics for business forecasting.", "category": "Data & Analytics"},
    ]

    solution_mapping = []
    for p in top_products:
        price = p.get("pricing", 50000)
        unit = p.get("unit", "Annual License")
        solution_mapping.append({
            "requirement": p.get("useCase", p.get("description", "Enterprise AI modernization")),
            "solution": p.get("serviceName", "AI Solution"),
            "match": 82,
            "value": f"${price:,}/{unit}"
        })

    return {
        "company": primary_company,
        "strategic_fit": 80,
        "overview": (
            f"{primary_company} is a target company being analyzed for AI solution opportunities. "
            "Based on market intelligence, they have strong potential for AI-driven modernization "
            "across their core business operations. Real-time scraping data is being processed."
        ),
        "needs_prediction": [
            "AI-powered automation to reduce manual operational overhead",
            "Predictive analytics for data-driven business decisions",
            "Enterprise knowledge management using Generative AI",
            "Sales intelligence platform to improve conversion rates"
        ],
        "meeting_prep": {
            "priorities": [
                "Operational efficiency improvement via AI automation",
                "Data modernization and analytics capability building",
                "Cost reduction through intelligent process automation"
            ],
            "growth_initiatives": [
                "Digital transformation of core business processes",
                "AI/ML integration into existing workflows"
            ],
            "risks": [
                "Budget approval timelines for enterprise AI projects",
                "Internal resistance to new technology adoption"
            ],
            "buying_signals": [
                "Active exploration of AI vendor partnerships",
                "Recent digital transformation announcements"
            ],
            "objections": [
                "Integration complexity with existing legacy systems",
                "Data security and compliance concerns with external AI"
            ],
            "stakeholders": [
                "Chief Technology Officer (CTO)",
                "VP of Digital Transformation",
                "Head of IT Infrastructure"
            ]
        },
        "solution_mapping": solution_mapping
    }


async def safe_run(coro, name: str, kind: str, sources_log: List[Dict[str, str]]) -> str:
    try:
        result = await coro
        if result and len(str(result).strip()) > 20:
            sources_log.append({"name": name, "type": kind})
            return str(result)
        print(f"[safe_run] '{name}' returned empty/short result.")
        return ""
    except Exception as e:
        print(f"[safe_run] ERROR in '{name}' ({kind}): {type(e).__name__}: {e}")
        return ""


# ─── Node 1: Intent Detection ──────────────────────────────────────────────────
async def query_understanding(state: RAGState) -> Dict[str, Any]:
    """Extracts structured entities: companies, intent, metrics, rephrased query."""
    prompt = f"""You are an Intent Detection Agent for a Sales Intelligence system.
Parse the user query and extract structured entities.

User Query: {state["query"]}

Return ONLY valid JSON:
{{
  "company_names": ["list of company names mentioned, empty list if none"],
  "intent": "one of: compare | analyze | lookup | summarize | news | general",
  "metrics": ["list from: products | leadership | news | history | financials | industry | competitors | technology | locations"],
  "time_frame": "date range if mentioned, else null",
  "rephrased_query": "a clean precise version of the query optimized for database search"
}}"""
    try:
        response = await ainvoke_with_retry(prompt)
        entities = json.loads(clean_json_response(response.content))
        return {
            **state,
            "entities": entities,
            "steps_taken": state["steps_taken"] + [
                f"Intent Detection: Intent='{entities.get('intent')}', Companies={entities.get('company_names')}"
            ]
        }
    except Exception as e:
        return {
            **state,
            "entities": {"company_names": [], "intent": "lookup", "metrics": ["general"],
                         "time_frame": None, "rephrased_query": state["query"]},
            "steps_taken": state["steps_taken"] + [f"Intent Detection fallback: {e}"]
        }


# ─── Node 2: Company Routing Decision ──────────────────────────────────────────
async def company_routing_decision(state: RAGState) -> Dict[str, Any]:
    """Checks if the company exists internally (MongoDB/CRM)."""
    entities = state["entities"]
    company_names = entities.get("company_names", [])
    primary_company = company_names[0] if company_names else state["query"]
    
    existing_report = get_cached_report_neon(primary_company)
    is_internal = existing_report is not None
    
    status = "Internal (Company Exists in DB)" if is_internal else "External (New Company — will scrape)"
    return {
        **state,
        "is_internal": is_internal,
        "current_param": primary_company,
        "steps_taken": state["steps_taken"] + [f"Routing Decision: {status}"]
    }


# ─── Node 3A: Internal Retrieval ───────────────────────────────────────────────
async def internal_retrieval(state: RAGState) -> Dict[str, Any]:
    """Fetches data from MongoDB / CRM / Previous Meetings / Sales History."""
    primary_company = state["current_param"]
    sources_log: List[Dict[str, str]] = []
    
    tasks = [
        safe_run(search_mongodb_tool(primary_company), "MongoDB / CRM Records", "mongodb", sources_log),
        safe_run(get_company_dossier_tool(primary_company), "Cached Intelligence Dossier", "dossier", sources_log)
    ]
    
    results = await asyncio.gather(*tasks)
    context_parts = []
    for i, result in enumerate(results):
        if result and result.strip():
            label = "MongoDB/CRM" if i == 0 else "Cached Dossier"
            context_parts.append(f"\n{'='*50}\nSOURCE: {label}\n{'='*50}\n{result}")

    merged = "\n".join(context_parts)
    return {
        **state,
        "retrieved_context": state["retrieved_context"] + [merged],
        "sources_used": state["sources_used"] + sources_log,
        "steps_taken": state["steps_taken"] + ["Internal Retrieval completed"]
    }


# ─── Node 3B: External Retrieval ───────────────────────────────────────────────
async def external_retrieval(state: RAGState) -> Dict[str, Any]:
    """Fetches data from Web Search, APIs, News, Company Website."""
    primary_company = state["current_param"]
    sources_log: List[Dict[str, str]] = []
    steps = []
    
    is_fast = state.get("is_fast_mode", True)
    print(f"[external_retrieval] Starting ingestion for external company: '{primary_company}' (Fast Mode: {is_fast})")
    
    if is_fast:
        tasks = [
            safe_run(live_news_lookup_tool(primary_company), "Live News APIs", "news", sources_log)
        ]
    else:
        tasks = [
            safe_run(get_company_dossier_tool(primary_company), "Web Scraping & APIs", "web", sources_log),
            safe_run(live_news_lookup_tool(primary_company), "Live News APIs", "news", sources_log)
        ]
    
    results = await asyncio.gather(*tasks)
    context_parts = []
    for i, result in enumerate(results):
        if result and result.strip():
            label = "Web/Dossier APIs" if i == 0 else "Live News APIs"
            context_parts.append(f"\n{'='*50}\nSOURCE: {label}\n{'='*50}\n{result}")
            steps.append(f"External Retrieval: {label} succeeded")
        else:
            label = "Web/Dossier APIs" if i == 0 else "Live News APIs"
            steps.append(f"External Retrieval: {label} returned no data")

    # If NO context at all, inject LLM knowledge fallback note
    if not any(r and r.strip() for r in results):
        print(f"[external_retrieval] WARNING: All data sources failed for '{primary_company}'. Using LLM fallback.")
        fallback_ctx = (
            f"\n{'='*50}\nSOURCE: LLM General Knowledge (Fallback)\n{'='*50}\n"
            f"Real-time web scraping could not retrieve data for '{primary_company}'. "
            f"The analysis agent should use its pre-trained knowledge about this company, "
            f"supplemented with our AI product catalog recommendations."
        )
        context_parts.append(fallback_ctx)
        steps.append("External Retrieval: All sources failed — using LLM fallback")

    merged = "\n".join(context_parts)
    return {
        **state,
        "retrieved_context": state["retrieved_context"] + [merged],
        "sources_used": state["sources_used"] + sources_log,
        "steps_taken": state["steps_taken"] + steps
    }


# ─── Node 4: AI Product Matching (NEW — core value prop) ─────────────────────
async def product_matching_node(state: RAGState) -> Dict[str, Any]:
    """
    Fetches our REAL AI products from MongoDB and internal company context.
    This ensures solution_mapping always uses actual products from our catalog.
    """
    primary_company = state["current_param"]
    
    # Get internal company context (CRM, proposals, meetings, case studies)
    internal_ctx = get_internal_company_context(primary_company)
    
    # Determine industry from CRM records or from entity detection
    industry = ""
    if internal_ctx.get("crm_records"):
        industry = internal_ctx["crm_records"][0].get("industry", "")
    
    # Fetch our AI products from MongoDB
    ai_products = get_ai_products_from_mongodb(industry=industry, company_name=primary_company, limit=10)
    
    # Build context summary for the analysis agent
    product_context_parts = []
    
    if ai_products:
        prod_lines = []
        for p in ai_products:
            price = p.get("pricing", 0)
            unit = p.get("unit", "Annual License")
            prod_lines.append(
                f"  - [{p.get('serviceId')}] {p.get('serviceName')} | Category: {p.get('category')} | "
                f"Price: ${price:,}/{unit} | Target: {p.get('targetIndustry', 'Any')} | "
                f"Value: {p.get('businessValue', '')} | Tech: {p.get('technology', '')}"
            )
        product_context_parts.append(
            f"\n{'='*50}\nSOURCE: Our AI Product Catalog (MongoDB)\n{'='*50}\n"
            f"Available AI Solutions:\n" + "\n".join(prod_lines)
        )
    
    if internal_ctx.get("crm_records"):
        crm_lines = [json.dumps(r, indent=2) for r in internal_ctx["crm_records"][:3]]
        product_context_parts.append(
            f"\n{'='*50}\nSOURCE: CRM History for {primary_company}\n{'='*50}\n" +
            "\n---\n".join(crm_lines)
        )
    
    if internal_ctx.get("proposals"):
        prop_lines = [
            f"  - {p.get('proposalTitle','?')} | Service: {p.get('serviceName','?')} | "
            f"Cost: ${p.get('estimatedCost',0):,} | Status: {p.get('proposalStatus','?')}"
            for p in internal_ctx["proposals"][:5]
        ]
        product_context_parts.append(
            f"\n{'='*50}\nSOURCE: Past Proposals for {primary_company}\n{'='*50}\n" +
            "\n".join(prop_lines)
        )
    
    if internal_ctx.get("past_meetings"):
        for m in internal_ctx["past_meetings"][:2]:
            product_context_parts.append(
                f"\n{'='*50}\nSOURCE: Past Meeting — {m.get('date','?')} ({m.get('meetingType','?')})\n{'='*50}\n" +
                f"Agenda: {m.get('agenda','')}\n"
                f"Summary: {m.get('discussionSummary','')}\n"
                f"Client Concerns: {json.dumps(m.get('clientConcerns',[]))}\n"
                f"Next Steps: {json.dumps(m.get('nextSteps',[]))}"
            )
    
    if internal_ctx.get("related_case_studies"):
        cs_lines = [
            f"  - [{cs.get('caseStudyId')}] {cs.get('title','?')} | Client: {cs.get('client','?')} | "
            f"Solution: {cs.get('solution','?')} | Result: {cs.get('results','?')}"
            for cs in internal_ctx["related_case_studies"][:3]
        ]
        product_context_parts.append(
            f"\n{'='*50}\nSOURCE: Relevant Case Studies\n{'='*50}\n" +
            "\n".join(cs_lines)
        )

    merged = "\n".join(product_context_parts)
    
    return {
        **state,
        "ai_products": ai_products,
        "internal_context": internal_ctx,
        "retrieved_context": state["retrieved_context"] + [merged],
        "sources_used": state["sources_used"] + [
            {"name": "MongoDB Product Catalog", "type": "mongodb"},
            {"name": "CRM & Meeting History", "type": "mongodb"}
        ],
        "steps_taken": state["steps_taken"] + [
            f"Product Matching: Found {len(ai_products)} AI products | "
            f"CRM records: {len(internal_ctx.get('crm_records', []))} | "
            f"Proposals: {len(internal_ctx.get('proposals', []))}"
        ]
    }


# ─── Node 5: Agentic RAG Retrieval ────────────────────────────────────────────
async def agentic_rag_retrieval(state: RAGState) -> Dict[str, Any]:
    """Retrieves highly relevant semantic context based on the query."""
    search_query = state["entities"].get("rephrased_query", state["query"])
    sources_log: List[Dict[str, str]] = []
    
    vector_result = await safe_run(vector_search_tool(search_query), "Semantic Vector Search", "vector", sources_log)
    
    if vector_result and vector_result.strip():
        merged = f"\n{'='*50}\nSOURCE: Agentic RAG (Vector Search)\n{'='*50}\n{vector_result}"
    else:
        merged = ""

    return {
        **state,
        "retrieved_context": state["retrieved_context"] + [merged],
        "sources_used": state["sources_used"] + sources_log,
        "steps_taken": state["steps_taken"] + ["Agentic RAG Retrieval completed"]
    }


# ─── Node 6: Analysis Agent (Sales Dashboard) ─────────────────────────────────
async def analysis_agent(state: RAGState) -> Dict[str, Any]:
    """
    Generates structured JSON for the Sales Dashboard.
    Always injects real AI products from MongoDB — never returns empty/null fields.
    """
    context_text = "\n".join(filter(None, state["retrieved_context"]))
    primary_company = state["current_param"] or state["query"]
    is_external = not state.get("is_internal", False)
    ai_products = state.get("ai_products", [])

    # Summarise our product catalog for the prompt
    product_catalog_summary = ""
    if ai_products:
        lines = []
        for p in ai_products[:8]:
            price = p.get("pricing", 0)
            unit = p.get("unit", "Annual License")
            lines.append(
                f'  serviceId="{p.get("serviceId")}" | name="{p.get("serviceName")}" | '
                f'category="{p.get("category")}" | price=${price:,}/{unit} | '
                f'value="{p.get("businessValue","")}" | tech="{p.get("technology","")}"'
            )
        product_catalog_summary = "OUR AVAILABLE AI PRODUCTS (from MongoDB):\n" + "\n".join(lines)

    sources_note = ""
    if state.get("sources_used"):
        src_names = [s.get("name", "?") for s in state["sources_used"]]
        sources_note = f"Data retrieved from: {', '.join(src_names)}."
    elif is_external:
        sources_note = "External company — use retrieved context + pre-trained LLM knowledge."

    prompt = f"""You are the Analysis Agent for a Sales Intelligence Dashboard.
We are an AI Solutions company. Our ONLY offerings are AI and technology products listed below.
Your task: analyse the target company and map OUR products to their specific needs.

Target Company: {primary_company}
Company Type: {"External (new prospect)" if is_external else "Internal (existing CRM relationship)"}
Data Sources: {sources_note}

{product_catalog_summary}

RULES:
1. Return ONLY perfectly valid JSON — no markdown fences.
2. solution_mapping MUST use product names ONLY from our catalog above (use the exact serviceName).
3. "value" in solution_mapping = the product's actual price from catalog (format: "$X,XXX/Annual License").
4. "match" = how well that product fits the company's need (integer 70–98).
5. Be SPECIFIC — no placeholder text, no "TBD", no null values anywhere.
6. overview must be 3-4 sentences about the target company (industry, scale, operations, recent news).
7. needs_prediction: at least 4 specific AI/tech needs grounded in what this company actually does.
8. meeting_prep: all fields must have at least 2-3 real, specific items.
9. strategic_fit: realistic integer 20–95 based on how AI-ready this company appears.
10. If context is thin, use your knowledge about this company — it must never be empty.

REQUIRED JSON SCHEMA:
{{
  "company": "{primary_company}",
  "strategic_fit": <integer 20-95>,
  "overview": "<3-4 sentences: company description, industry, scale, differentiators, recent news>",
  "needs_prediction": [
    "<Specific AI/tech need 1 with business rationale>",
    "<Specific need 2>",
    "<Specific need 3>",
    "<Specific need 4>"
  ],
  "meeting_prep": {{
    "priorities": ["<Business priority 1>", "<Priority 2>", "<Priority 3>"],
    "growth_initiatives": ["<Growth initiative 1>", "<Initiative 2>"],
    "risks": ["<Key risk 1>", "<Risk 2>"],
    "buying_signals": ["<Concrete buying signal 1>", "<Signal 2>"],
    "objections": ["<Realistic objection 1>", "<Objection 2>"],
    "stakeholders": ["<Decision maker role 1>", "<Role 2>", "<Role 3>"]
  }},
  "solution_mapping": [
    {{
      "requirement": "<Their specific need that this product addresses>",
      "solution": "<EXACT serviceName from our catalog>",
      "match": <integer 70-98>,
      "value": "<price from catalog, e.g. '$52,000/Annual License'>"
    }},
    {{
      "requirement": "<Second need>",
      "solution": "<Second product serviceName>",
      "match": <integer 70-98>,
      "value": "<price from catalog>"
    }},
    {{
      "requirement": "<Third need>",
      "solution": "<Third product serviceName>",
      "match": <integer 60-95>,
      "value": "<price from catalog>"
    }}
  ]
}}

RETRIEVED CONTEXT (primary source — supplement with your knowledge if needed):
{context_text[:20000]}
"""
    try:
        response = await ainvoke_with_retry(prompt)
        final_answer = clean_json_response(response.content)
        parsed = json.loads(final_answer)
        
        # ── Guarantee no null/empty fields ──────────────────────────────────
        if not parsed.get("overview") or len(parsed.get("overview", "")) < 20:
            parsed["overview"] = (
                f"{primary_company} is a major enterprise being analyzed for AI-driven transformation opportunities. "
                "They represent a strong prospect for our AI product portfolio based on their scale and industry position."
            )
        if not parsed.get("needs_prediction") or len(parsed["needs_prediction"]) < 2:
            parsed["needs_prediction"] = [
                "AI-powered automation to reduce manual process overhead",
                "Predictive analytics for smarter business decisions",
                "Enterprise knowledge management via Generative AI",
                "Sales intelligence platform to accelerate revenue growth"
            ]
        meeting_prep = parsed.get("meeting_prep", {})
        for field in ["priorities", "growth_initiatives", "risks", "buying_signals", "objections", "stakeholders"]:
            if not meeting_prep.get(field):
                meeting_prep[field] = [f"Awaiting further research on {primary_company}'s {field}"]
        parsed["meeting_prep"] = meeting_prep
        
        if not parsed.get("solution_mapping") or len(parsed["solution_mapping"]) < 1:
            # Build from real products
            fallback = _build_guaranteed_fallback(primary_company, ai_products)
            parsed["solution_mapping"] = fallback["solution_mapping"]
        
        # Ensure solution_mapping values are never empty/null
        for sm in parsed.get("solution_mapping", []):
            if not sm.get("requirement"):
                sm["requirement"] = "Enterprise AI modernization"
            if not sm.get("solution"):
                sm["solution"] = ai_products[0]["serviceName"] if ai_products else "AI Sales Intelligence Platform"
            if not sm.get("match"):
                sm["match"] = 80
            if not sm.get("value") or sm.get("value") in ["TBD", "", None]:
                # Find matching product price
                prod_name = sm.get("solution", "")
                matched = next((p for p in ai_products if p.get("serviceName") == prod_name), None)
                if matched:
                    sm["value"] = f"${matched.get('pricing', 50000):,}/{matched.get('unit', 'Annual License')}"
                else:
                    sm["value"] = "$52,000/Annual License"
        
        final_answer = json.dumps(parsed)
        return {
            **state,
            "final_answer": final_answer,
            "steps_taken": state["steps_taken"] + ["Analysis Agent: Dashboard JSON generated successfully"]
        }
    except Exception as e:
        print(f"[analysis_agent] ERROR: {type(e).__name__}: {e}")
        # Build guaranteed fallback using real products
        fallback = _build_guaranteed_fallback(primary_company, ai_products)
        return {
            **state,
            "final_answer": json.dumps(fallback),
            "steps_taken": state["steps_taken"] + [f"Analysis Agent: Used fallback (LLM error: {e})"]
        }


# ─── Router ───────────────────────────────────────────────────────────────────
def route_company_existence(state: RAGState) -> str:
    if state["is_internal"]:
        return "internal_retrieval"
    else:
        return "external_retrieval"


# ─── Graph Assembly ───────────────────────────────────────────────────────────
def build_rag_graph():
    workflow = StateGraph(RAGState)  # type: ignore

    workflow.add_node("query_understanding", query_understanding)
    workflow.add_node("company_routing_decision", company_routing_decision)
    workflow.add_node("internal_retrieval", internal_retrieval)
    workflow.add_node("external_retrieval", external_retrieval)
    workflow.add_node("product_matching_node", product_matching_node)
    workflow.add_node("agentic_rag_retrieval", agentic_rag_retrieval)
    workflow.add_node("analysis_agent", analysis_agent)

    workflow.set_entry_point("query_understanding")
    workflow.add_edge("query_understanding", "company_routing_decision")
    
    workflow.add_conditional_edges(
        "company_routing_decision",
        route_company_existence,
        {
            "internal_retrieval": "internal_retrieval",
            "external_retrieval": "external_retrieval"
        }
    )
    
    # Both paths merge into product matching, then RAG, then analysis
    workflow.add_edge("internal_retrieval", "product_matching_node")
    workflow.add_edge("external_retrieval", "product_matching_node")
    workflow.add_edge("product_matching_node", "agentic_rag_retrieval")
    workflow.add_edge("agentic_rag_retrieval", "analysis_agent")
    workflow.add_edge("analysis_agent", END)

    return workflow.compile()

# Compile the graph globally to avoid recompiling on every request
rag_app_graph = build_rag_graph()


# ─── Deal Coach Chat ──────────────────────────────────────────────────────────
async def run_deal_coach_chat(message: str, company: str, dashboard_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Provides conversational sales coaching for a specific company.
    Injects real AI products and company intelligence into context.
    """
    # Pull real AI products from MongoDB for context
    ai_products = get_ai_products_from_mongodb(limit=6)
    internal_ctx = get_internal_company_context(company)
    
    product_lines = []
    for p in ai_products[:5]:
        product_lines.append(
            f"  - {p.get('serviceName')} (${p.get('pricing', 0):,}/{p.get('unit', 'yr')}) — {p.get('businessValue', '')}"
        )
    product_catalog = "\n".join(product_lines) if product_lines else "AI Sales Intelligence Platform, Enterprise Knowledge Assistant, Predictive Analytics Engine"
    
    # Build company-specific context from CRM + meetings
    company_context = ""
    if internal_ctx.get("crm_records"):
        crm = internal_ctx["crm_records"][0]
        company_context += (
            f"CRM Status: {crm.get('status', 'Unknown')} | "
            f"Contact: {crm.get('contactName', 'Unknown')} ({crm.get('designation', '')}) | "
            f"Pain Point: {crm.get('painPoint', '')} | "
            f"Next Action: {crm.get('nextAction', '')}\n"
        )
    if internal_ctx.get("past_meetings"):
        m = internal_ctx["past_meetings"][0]
        company_context += (
            f"Last Meeting: {m.get('date', '?')} ({m.get('meetingType', '?')}) | "
            f"Sentiment: {m.get('sentiment', 'Unknown')} | "
            f"Client Concerns: {', '.join(m.get('clientConcerns', []))}\n"
        )
    if internal_ctx.get("proposals"):
        p = internal_ctx["proposals"][0]
        company_context += (
            f"Open Proposal: {p.get('proposalTitle', '?')} | "
            f"Service: {p.get('serviceName', '?')} | "
            f"Value: ${p.get('estimatedCost', 0):,} | "
            f"Status: {p.get('proposalStatus', '?')}\n"
        )
    
    # Include dashboard data summary if available
    dashboard_context = ""
    if dashboard_data:
        dashboard_context = (
            f"Company Overview: {dashboard_data.get('overview', '')}\n"
            f"Strategic Fit Score: {dashboard_data.get('strategic_fit', 'N/A')}%\n"
            f"Predicted Needs: {', '.join(dashboard_data.get('needs_prediction', [])[:3])}\n"
        )

    prompt = f"""You are the SalesIntel AI Deal Coach — an expert enterprise sales copilot for our AI Solutions company.
You help account executives close deals by providing highly strategic, specific, and actionable coaching.

TARGET COMPANY: {company}
{f"COMPANY INTELLIGENCE:{chr(10)}{company_context}" if company_context else ""}
{f"DASHBOARD INTELLIGENCE:{chr(10)}{dashboard_context}" if dashboard_context else ""}

OUR AI PRODUCT PORTFOLIO:
{product_catalog}

AE's Question / Message: {message}

RESPONSE RULES:
- If the AE's message is a simple greeting (e.g., "hi", "hello"), respond naturally and conversationally (e.g., "Hi! How can I help you with the {company} deal today?").
- For substantive questions, provide a highly specific, strategic, and structured response.
- Reference our actual products when making recommendations.
- If we have CRM/meeting history, use those specific details in your answer.
- Keep response concise: 2-3 short paragraphs max.
- Use standard Markdown formatting (like **bold** and newlines). Do NOT use HTML tags like <strong> or <br>.
- For substantive questions, always end with a specific next action the AE should take.
"""
    try:
        response = await ainvoke_with_retry(prompt)
        return response.content
    except Exception as e:
        provider = os.getenv("LLM_PROVIDER", "gemini").upper()
        return (
            f"**[{provider} API Quota Exhausted]**\n\n"
            f"Unable to provide live coaching right now. \n\n"
            f"**Quick Tip for {company}:** Focus on their pain points around "
            f"operational efficiency. Lead with our **AI Sales Intelligence Platform** ($52,000/yr) "
            f"and offer a 30-day POC to reduce friction."
        )


# ─── Entry Point ──────────────────────────────────────────────────────────────
async def run_rag_agent(query: str, fast_mode: bool = True) -> Dict[str, Any]:
    initial_state: RAGState = {
        "query": query,
        "entities": {},
        "is_internal": False,
        "is_fast_mode": fast_mode,
        "steps_taken": [],
        "retrieved_context": [],
        "sources_used": [],
        "next_action": "",
        "current_param": "",
        "final_answer": "",
        "ai_products": [],
        "internal_context": {}
    }
    result = await rag_app_graph.ainvoke(initial_state)
    return result
