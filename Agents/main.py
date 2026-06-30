import os
import asyncio
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Response
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
from app.database import get_cached_report, save_report, search_vector_store

app = FastAPI(
    title="Company Intelligence API",
    description="FastAPI + LangGraph powered company intelligence agent with SQLite database, semantic vector search, and PDF download support.",
    version="1.1.0"
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

        # Cache miss or forced reload -> Run async LangGraph execution
        result = await run_company_agent(
            company_name=payload.company_name,
            company_website=payload.company_website
        )

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