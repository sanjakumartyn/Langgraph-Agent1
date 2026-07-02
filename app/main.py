import os
import sys
import asyncio
import json

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from reportlab.lib.pagesizes import letter

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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from app.schemas import CompanyRequest, CompanyResponse, SearchRequest, SearchResponse, SearchResult
from app.graph import run_company_agent
from app.database import (
    get_cached_report,
    save_report,
    search_vector_store,
    get_reports_history,
    clear_reports_history
)

app = FastAPI(
    title="Company Intelligence API",
    description="FastAPI + LangGraph powered company intelligence agent with SQLite database, semantic vector search, and PDF download support.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def generate_chunks_and_embeddings(company_name: str, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Helper to chunk the intelligence report and generate vector embeddings using Gemini."""
    chunks = []
    
    # 1. Summary Chunk
    summary = report.get("company_summary") or ""
    industry = report.get("industry") or ""
    if summary:
        chunks.append(f"Company: {company_name}. Industry: {industry}. Summary: {summary}")
        
    # 2. Products Chunk
    products = report.get("products_or_services") or []
    if products:
        prod_strs = []
        for p in products:
            if isinstance(p, dict):
                cat = p.get("category") or ""
                sub = ", ".join(p.get("subcategories") or [])
                prod_strs.append(f"{cat}: {sub}" if sub else cat)
            else:
                prod_strs.append(str(p))
        chunks.append(f"Company: {company_name}. Products and Services: {', '.join(prod_strs)}")
        
    # 3. Priorities & Pain Points Chunk
    priorities = report.get("business_priorities") or []
    pain_points = report.get("pain_points") or []
    if priorities or pain_points:
        chunks.append(
            f"Company: {company_name}. "
            f"Business Priorities: {', '.join(priorities)}. "
            f"Pain Points and Risks: {', '.join(pain_points)}"
        )
        
    # 4. Tech Signals Chunk
    tech = report.get("technology_signals") or []
    if tech:
        chunks.append(f"Company: {company_name}. Technology Signals and Systems: {', '.join(tech)}")
        
    if not chunks:
        chunks.append(f"Company: {company_name}. Industry: {industry}.")
        
    embeddings_model = get_embeddings()
    
    results = []
    for chunk in chunks:
        # Run embedding generation in thread pool to prevent event loop blocking
        emb = await asyncio.to_thread(embeddings_model.embed_query, chunk)
        results.append({
            "chunk_text": chunk,
            "embedding": emb
        })
        
    return results


def generate_pdf_report(company_name: str, report: Dict[str, Any]) -> bytes:
    """Generate a clean, beautiful PDF report using ReportLab."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    meta_label_style = ParagraphStyle(
        "MetaLabel",
        parent=styles["Normal"],
        fontSize=10,
        leading=12,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#2D3748")
    )
    
    story = []
    
    # Title Header
    story.append(Paragraph(f"Company Intelligence Report: {company_name}", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", body_style))
    story.append(Spacer(1, 10))
    
    # Metadata Table
    conf_score = report.get("confidence_score", 0)
    conf_color = "#38A169" if conf_score >= 70 else ("#DD6B20" if conf_score >= 40 else "#E53E3E")
    
    meta_data = [
        [Paragraph("Website:", meta_label_style), Paragraph(report.get("website") or "N/A", body_style)],
        [Paragraph("Industry:", meta_label_style), Paragraph(report.get("industry") or "N/A", body_style)],
        [Paragraph("Confidence Score:", meta_label_style), Paragraph(f"<font color='{conf_color}'><b>{conf_score}%</b></font>", body_style)],
    ]
    
    meta_table = Table(meta_data, colWidths=[120, 420])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    # Summary Section
    story.append(Paragraph("Executive Summary", h2_style))
    story.append(Paragraph(report.get("company_summary") or "No summary available.", body_style))
    story.append(Spacer(1, 10))
    
    # Products or Services Section
    story.append(Paragraph("Products & Services", h2_style))
    products = report.get("products_or_services") or []
    if products:
        for p in products:
            if isinstance(p, dict):
                cat = p.get("category") or ""
                sub = ", ".join(p.get("subcategories") or [])
                text = f"<b>{cat}</b>: {sub}" if sub else cat
            else:
                text = str(p)
            story.append(Paragraph(f"• {text}", bullet_style))
    else:
        story.append(Paragraph("No product or service information found.", body_style))
    story.append(Spacer(1, 10))
    
    # Business Priorities Section
    story.append(Paragraph("Business Priorities", h2_style))
    priorities = report.get("business_priorities") or []
    if priorities:
        for pr in priorities:
            story.append(Paragraph(f"• {pr}", bullet_style))
    else:
        story.append(Paragraph("No key business priorities identified.", body_style))
    story.append(Spacer(1, 10))
    
    # Pain Points Section
    story.append(Paragraph("Pain Points & Risks", h2_style))
    pain_points = report.get("pain_points") or []
    if pain_points:
        for pp in pain_points:
            story.append(Paragraph(f"• {pp}", bullet_style))
    else:
        story.append(Paragraph("No major pain points or risks identified.", body_style))
    story.append(Spacer(1, 10))
    
    # Technology Signals Section
    story.append(Paragraph("Technology Signals", h2_style))
    tech = report.get("technology_signals") or []
    if tech:
        for t in tech:
            story.append(Paragraph(f"• {t}", bullet_style))
    else:
        story.append(Paragraph("No significant technology signals identified.", body_style))
    story.append(Spacer(1, 10))
    
    # Recent Market & News Section
    story.append(Paragraph("Recent Market & News Developments", h2_style))
    news = report.get("market_news") or []
    if news:
        for n in news:
            if isinstance(n, dict):
                title = n.get("title") or ""
                desc = n.get("description") or ""
                src = n.get("source") or ""
                text = f"<b>{title}</b> ({src})"
                if desc:
                    text += f" - {desc}"
            else:
                text = str(n)
            story.append(Paragraph(f"• {text}", bullet_style))
    else:
        story.append(Paragraph("No recent news articles matched.", body_style))
    story.append(Spacer(1, 10))
    
    # History Section
    story.append(Paragraph("Company History & Milestones", h2_style))
    history = report.get("company_history") or []
    if history:
        for h in history:
            story.append(Paragraph(f"• {h}", bullet_style))
    else:
        story.append(Paragraph("No detailed history milestones found.", body_style))
    story.append(Spacer(1, 10))
    
    # Evidence Section (Page Break)
    evidence = report.get("source_evidence") or []
    if evidence:
        story.append(PageBreak())
        story.append(Paragraph("Source Evidence", h2_style))
        
        ev_data = [[Paragraph("<b>Claim</b>", meta_label_style), Paragraph("<b>Source</b>", meta_label_style)]]
        for ev in evidence:
            claim = ev.get("claim") or ""
            src = ev.get("source") or ""
            url = ev.get("url") or ""
            src_text = f"<a href='{url}' color='#2B6CB0'>{src}</a>" if url else src
            ev_data.append([
                Paragraph(claim, body_style),
                Paragraph(src_text, body_style)
            ])
            
        ev_table = Table(ev_data, colWidths=[380, 160])
        ev_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ]))
        story.append(ev_table)
        
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


@app.get("/")
def health_check():
    return {
        "success": True,
        "message": "Company Intelligence API is running"
    }

import sys
agentic_rag_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agentic_rag")
if agentic_rag_path not in sys.path:
    sys.path.append(agentic_rag_path)
from rag_app.api.routers.dashboard import router as dashboard_router
app.include_router(dashboard_router)


@app.post("/company/intelligence", response_model=CompanyResponse)
async def company_intelligence(payload: CompanyRequest):
    try:
        # Check cache database first unless force_refresh is True
        if not payload.force_refresh:
            cached = get_cached_report(payload.company_name)
            if cached:
                return {
                    "success": True,
                    "data": cached,
                    "errors": ["Loaded from database cache"]
                }

        try:
            # Cache miss or forced reload -> Run via subprocess to avoid Windows event loop conflicts
            import subprocess
            import json
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
                tmp_path = tmp.name
                
            script = f'''
import asyncio
import json
from app.graph import run_company_agent

async def main():
    try:
        res = await run_company_agent({repr(payload.company_name)}, {repr(payload.company_website or '')})
        with open({repr(tmp_path)}, 'w', encoding='utf-8') as f:
            json.dump({{"success": True, "result": res}}, f, default=str)
    except Exception as e:
        import traceback
        with open({repr(tmp_path)}, 'w', encoding='utf-8') as f:
            json.dump({{"success": False, "error": str(e), "traceback": traceback.format_exc()}}, f)

asyncio.run(main())
'''
            def run_sync():
                import sys
                return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
            
            await asyncio.to_thread(run_sync)
            
            with open(tmp_path, 'r', encoding='utf-8') as f:
                output = json.load(f)
            os.remove(tmp_path)
            
            if not output.get("success"):
                return {
                    "success": False,
                    "data": {},
                    "errors": [f"Agent execution failed: {output.get('error')}", output.get("traceback")]
                }
            result = output.get("result", {})
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return {
                "success": False,
                "data": {},
                "errors": [f"Agent wrapper failed: {str(e)}", tb]
            }


        final_result = result.get("final_result", {})
        errors = result.get("errors", [])

        # If we successfully created a report, save it to the DB and index vectors
        if final_result and final_result.get("company_name"):
            comp_name = final_result["company_name"]
            web_url = final_result.get("website") or payload.company_website or ""
            
            try:
                # Generate embeddings & save
                chunks_embeddings = await generate_chunks_and_embeddings(comp_name, final_result)
                save_report(comp_name, web_url, final_result, chunks_embeddings)
            except Exception as db_err:
                errors.append(f"Database save error: {str(db_err)}")

        return {
            "success": True,
            "data": final_result,
            "errors": errors
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/company/report/{company_name}/pdf")
async def download_pdf(company_name: str):
    """Retrieve report from database and generate a PDF response."""
    report = get_cached_report(company_name)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"Company report for '{company_name}' not found. Please generate the intelligence report first."
        )
        
    try:
        # Run PDF generation in thread pool to keep event loop free
        pdf_bytes = await asyncio.to_thread(generate_pdf_report, company_name, report)
        
        headers = {
            'Content-Disposition': f'attachment; filename="{company_name.lower().replace(" ", "_")}_intelligence_report.pdf"'
        }
        return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation error: {str(e)}"
        )


@app.post("/company/search", response_model=SearchResponse)
async def semantic_search(payload: SearchRequest):
    """Embed the query and find semantically matching company report chunks."""
    try:
        # Embed query text
        embeddings_model = get_embeddings()
        query_emb = await asyncio.to_thread(embeddings_model.embed_query, payload.query)
        
        # Query local vector store
        raw_results = search_vector_store(query_emb, limit=payload.limit)
        
        results = [
            SearchResult(
                company_name=r["company_name"],
                chunk_text=r["chunk_text"],
                similarity=round(r["similarity"], 4)
            )
            for r in raw_results
        ]
        
        return {
            "success": True,
            "results": results,
            "errors": []
        }
        
    except Exception as e:
        return {
            "success": False,
            "results": [],
            "errors": [str(e)]
        }


@app.get("/history")
def get_history_endpoint():
    try:
        history = get_reports_history()
        return {
            "success": True,
            "data": history
        }
    except Exception as e:
        return {
            "success": False,
            "data": [],
            "errors": [str(e)]
        }

@app.delete("/history")
def delete_history_endpoint():
    try:
        clear_reports_history()
        return {
            "success": True,
            "message": "History cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/details/{company_name}")
def get_company_details_brief(company_name: str):
    try:
        report = get_cached_report(company_name)
        if report:
            industry = report.get("industry") or "Unknown"
            brief = report.get("company_summary") or "No description available."
            revenue = 15000000
            if "billion" in brief.lower():
                revenue = 1000000000
            elif "million" in brief.lower():
                revenue = 50000000
            return {
                "ticker": "N/A",
                "industry": industry,
                "revenue": revenue,
                "brief": brief
            }
        else:
            return {
                "ticker": "N/A",
                "industry": "Unknown",
                "revenue": 0,
                "brief": "No cached details available. Run an analysis first."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/companydata/company/details")
def get_latest_company_details():
    try:
        import sqlite3
        import json
        from app.database import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT company_name, report_data FROM company_reports ORDER BY created_at DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        if row:
            company_name = row["company_name"]
            try:
                report = json.loads(row["report_data"])
            except Exception:
                report = {}
            
            locations = report.get("locations") or []
            location_str = locations[0] if locations else "Global"
            
            return {
                "success": True,
                "data": {
                    "healthTrend": "Strong Growth",
                    "industry": report.get("industry") or "Technology",
                    "annualRevenue": 18500000,
                    "strategicFit": report.get("strategic_fit_score") or report.get("confidence_score") or 85,
                    "companyName": company_name,
                    "location": location_str,
                    "employees": 1250,
                    "website": report.get("website") or "",
                    "nextMilestone": {
                        "title": "Strategy Update",
                        "date": datetime.now().strftime("%Y-%m-%d")
                    }
                }
            }
            
        return {
            "success": True,
            "data": {
                "healthTrend": "Strong Growth",
                "industry": "Automotive",
                "annualRevenue": 240000000,
                "strategicFit": 92,
                "companyName": "Starlight Automotive",
                "location": "Detroit, USA",
                "employees": 4500,
                "website": "https://starlightautomotive.example.com",
                "nextMilestone": {
                    "title": "Q4 Expansion Review",
                    "date": "2026-10-15"
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    account = body.get("account") or "Unknown"
    message = body.get("message") or ""
    
    report_context = ""
    cached = get_cached_report(account)
    if cached:
        report_context = json.dumps(cached, indent=2)
        
    prompt = f"""
You are an expert AI Deal Coach. You help salespeople win deals, prep for meetings, handle objections, and design sales strategies.

Here is the context about the company:
{report_context}

A salesperson is asking this question:
"{message}"

If the message is a simple greeting (e.g., "hi", "hello"), respond conversationally (e.g., "Hi! How can I help you today?").
Otherwise, provide a professional, strategic, and concise answer. Highlight key steps or insights for the salesperson.
"""
    try:
        from app.nodes import llm
        response = await llm.ainvoke(prompt)
        reply = response.content
    except Exception as e:
        reply = f"Error calling AI Deal Coach: {str(e)}"
        
    return {
        "reply": reply
    }

@app.post("/company-analysis/deal-coach")
async def company_analysis_deal_coach_endpoint(request: Request):
    content_type = request.headers.get("content-type", "")
    
    company_name = "Unknown"
    message_text = ""
    analysis_context = None
    file_contents = []
    
    if "multipart/form-data" in content_type:
        form = await request.form()
        company_name = form.get("company_name") or form.get("company") or "Unknown"
        message_text = form.get("message") or form.get("question") or ""
        context_str = form.get("analysis_context")
        if context_str:
            try:
                analysis_context = json.loads(context_str)
            except Exception:
                analysis_context = context_str
                
        uploaded_files = form.getlist("file")
        for ufile in uploaded_files:
            if ufile.filename:
                content = await ufile.read()
                try:
                    text = content.decode("utf-8", errors="ignore")
                    file_contents.append(f"File: {ufile.filename}\nContent:\n{text}")
                except Exception as e:
                    file_contents.append(f"File: {ufile.filename} (could not decode: {str(e)})")
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        company_name = body.get("company_name") or body.get("company") or "Unknown"
        message_text = body.get("message") or body.get("question") or ""
        analysis_context = body.get("analysis_context")
        
    report_context = ""
    cached = get_cached_report(company_name)
    if cached:
        report_context = json.dumps(cached, indent=2)
    elif analysis_context:
        report_context = json.dumps(analysis_context, indent=2)
        
    prompt = f"""
You are an expert AI Deal Coach. You help salespeople win deals, prep for meetings, handle objections, and design sales strategies.

Here is the context about the company:
{report_context}
"""
    if file_contents:
        prompt += "\nHere is extra information from uploaded files:\n" + "\n\n".join(file_contents)
        
    prompt += f"""
A salesperson is asking this question:
"{message_text}"

If the message is a simple greeting (e.g., "hi", "hello"), respond conversationally (e.g., "Hi! How can I help you today?").
Otherwise, provide a highly professional, strategic, actionable, and structured response.
"""
    
    try:
        from app.nodes import llm
        response = await llm.ainvoke(prompt)
        reply = response.content
    except Exception as e:
        reply = f"Error calling AI Deal Coach: {str(e)}"
        
    return {
        "success": True,
        "data": {
            "answer": reply
        },
        "answer": reply
    }

@app.post("/generate-document")
async def generate_document_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    company = body.get("company") or "Company"
    
    return {
        "status": "success",
        "downloads": {
            "pdf": f"/api/company/report/{company}/pdf",
            "docx": f"/api/company/report/{company}/pdf",
            "xlsx": f"/api/company/report/{company}/pdf"
        }
    }