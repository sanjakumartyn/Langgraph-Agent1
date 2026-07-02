import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


# ── Brand Colors ──────────────────────────────────────────────────────────────
DARK_NAVY  = colors.HexColor("#06142E")
GOLD       = colors.HexColor("#F5B700")
STEEL_BLUE = colors.HexColor("#1E3A5F")
LIGHT_GRAY = colors.HexColor("#F4F6F9")
MID_GRAY   = colors.HexColor("#6B7280")
WHITE      = colors.white
GREEN      = colors.HexColor("#10B981")
ACCENT     = colors.HexColor("#3B82F6")


def _build_styles():
    styles = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle(
            "TitleStyle",
            fontName="Helvetica-Bold",
            fontSize=26,
            textColor=WHITE,
            spaceAfter=6,
            leading=32,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleStyle",
            fontName="Helvetica",
            fontSize=12,
            textColor=GOLD,
            spaceAfter=4,
        ),
        "section_header": ParagraphStyle(
            "SectionHeader",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=DARK_NAVY,
            spaceBefore=16,
            spaceAfter=8,
            borderPad=4,
        ),
        "body": ParagraphStyle(
            "BodyStyle",
            fontName="Helvetica",
            fontSize=10,
            textColor=colors.HexColor("#374151"),
            leading=15,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "BulletStyle",
            fontName="Helvetica",
            fontSize=10,
            textColor=colors.HexColor("#374151"),
            leading=14,
            leftIndent=16,
            spaceAfter=4,
        ),
        "label": ParagraphStyle(
            "LabelStyle",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=MID_GRAY,
            spaceAfter=2,
        ),
        "value": ParagraphStyle(
            "ValueStyle",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=DARK_NAVY,
            spaceAfter=4,
        ),
        "tag": ParagraphStyle(
            "TagStyle",
            fontName="Helvetica",
            fontSize=9,
            textColor=STEEL_BLUE,
            leading=12,
        ),
    }
    return custom


def generate_company_proposal_pdf(
    company_name: str,
    output_path: str,
    dashboard_data: Optional[Dict[str, Any]] = None,
    ai_products: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Generates a rich, company-specific proposal PDF using real dashboard data
    and our actual AI product catalog.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    s = _build_styles()
    story = []
    today = datetime.now().strftime("%B %d, %Y")

    # Extract data from dashboard_data (with safe fallbacks)
    data = dashboard_data or {}
    overview       = data.get("overview", f"{company_name} is a target enterprise for our AI solutions portfolio.")
    strategic_fit  = data.get("strategic_fit", 80)
    needs          = data.get("needs_prediction", [
        "AI-powered process automation",
        "Predictive analytics for business decisions",
        "Enterprise knowledge management",
        "Sales intelligence optimization",
    ])
    meeting_prep   = data.get("meeting_prep", {})
    solution_map   = data.get("solution_mapping", [])
    
    priorities     = meeting_prep.get("priorities", [])
    growth         = meeting_prep.get("growth_initiatives", [])
    risks          = meeting_prep.get("risks", [])
    buying_signals = meeting_prep.get("buying_signals", [])
    objections     = meeting_prep.get("objections", [])
    stakeholders   = meeting_prep.get("stakeholders", [])

    # ── HEADER BANNER ─────────────────────────────────────────────────────────
    header_data = [[
        Paragraph(f"AI Sales Proposal", s["title"]),
        Paragraph(f"<b>{company_name}</b>", ParagraphStyle(
            "RightTitle", fontName="Helvetica-Bold", fontSize=20, textColor=GOLD,
            alignment=2
        )),
    ]]
    header_table = Table(header_data, colWidths=[3.5 * inch, 3.5 * inch])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_NAVY),
        ("PADDING", (0, 0), (-1, -1), 14),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(header_table)

    # Date + confidential row
    story.append(Spacer(1, 6))
    meta_data = [[
        Paragraph(f"Prepared: {today}", s["label"]),
        Paragraph("CONFIDENTIAL — FOR INTERNAL USE ONLY", ParagraphStyle(
            "Conf", fontName="Helvetica-Oblique", fontSize=8, textColor=MID_GRAY, alignment=2
        )),
    ]]
    meta_table = Table(meta_data, colWidths=[3.5 * inch, 3.5 * inch])
    meta_table.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story.append(meta_table)
    story.append(HRFlowable(width="100%", thickness=2, color=GOLD, spaceAfter=12))

    # ── STRATEGIC FIT SCORE ───────────────────────────────────────────────────
    fit_color = GREEN if strategic_fit >= 75 else GOLD if strategic_fit >= 50 else colors.HexColor("#EF4444")
    score_data = [[
        Paragraph("STRATEGIC FIT SCORE", s["label"]),
        Paragraph(f"<b>{strategic_fit}%</b>", ParagraphStyle(
            "Score", fontName="Helvetica-Bold", fontSize=28, leading=34, textColor=fit_color, alignment=1
        )),
        Paragraph(
            "HIGH POTENTIAL" if strategic_fit >= 75 else "MODERATE" if strategic_fit >= 50 else "EXPLORATORY",
            ParagraphStyle("ScoreLabel", fontName="Helvetica-Bold", fontSize=11,
                          textColor=fit_color, alignment=1)
        ),
    ]]
    score_table = Table(score_data, colWidths=[2.5 * inch, 2 * inch, 2.5 * inch])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("PADDING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 1, GOLD),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 14))

    # ── COMPANY OVERVIEW ──────────────────────────────────────────────────────
    story.append(Paragraph("🏢  Company Overview", s["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY, spaceAfter=6))
    story.append(Paragraph(overview, s["body"]))
    story.append(Spacer(1, 10))

    # ── PREDICTED AI NEEDS ────────────────────────────────────────────────────
    story.append(Paragraph("🤖  Predicted AI & Technology Needs", s["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY, spaceAfter=6))
    for need in (needs or []):
        story.append(Paragraph(f"▸  {need}", s["bullet"]))
    story.append(Spacer(1, 10))

    # ── OUR AI SOLUTION RECOMMENDATIONS ──────────────────────────────────────
    story.append(Paragraph("💡  Recommended AI Solutions", s["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY, spaceAfter=6))

    if solution_map:
        sol_rows = [
            [
                Paragraph("<b>Their Requirement</b>", s["label"]),
                Paragraph("<b>Our AI Solution</b>", s["label"]),
                Paragraph("<b>Match %</b>", s["label"]),
                Paragraph("<b>Estimated Value</b>", s["label"]),
            ]
        ]
        for sm in solution_map:
            match_val = sm.get("match", 80)
            match_color = GREEN if match_val >= 85 else GOLD if match_val >= 70 else MID_GRAY
            sol_rows.append([
                Paragraph(sm.get("requirement", "—"), s["body"]),
                Paragraph(f"<b>{sm.get('solution', '—')}</b>", ParagraphStyle(
                    "SolName", fontName="Helvetica-Bold", fontSize=10, textColor=STEEL_BLUE
                )),
                Paragraph(f"<b>{match_val}%</b>", ParagraphStyle(
                    "Match", fontName="Helvetica-Bold", fontSize=11, textColor=match_color, alignment=1
                )),
                Paragraph(sm.get("value", "—"), s["body"]),
            ])
        sol_table = Table(sol_rows, colWidths=[2.0 * inch, 2.0 * inch, 0.8 * inch, 2.2 * inch])
        sol_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK_NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROUNDEDCORNERS", [4]),
        ]))
        story.append(sol_table)
    else:
        story.append(Paragraph("Solution mapping will be generated after company analysis.", s["body"]))
    story.append(Spacer(1, 10))

    # ── MEETING PREP ──────────────────────────────────────────────────────────
    story.append(Paragraph("📋  Meeting Preparation", s["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY, spaceAfter=6))

    prep_data = [
        ("Business Priorities", priorities),
        ("Growth Initiatives", growth),
        ("Risk Factors", risks),
        ("Buying Signals", buying_signals),
        ("Likely Objections", objections),
        ("Key Stakeholders to Target", stakeholders),
    ]

    for label, items in prep_data:
        if items:
            story.append(Paragraph(f"<b>{label}</b>", s["label"]))
            for item in items:
                story.append(Paragraph(f"• {item}", s["bullet"]))
            story.append(Spacer(1, 6))

    # ── AI PRODUCT CATALOG APPENDIX ───────────────────────────────────────────
    if ai_products:
        story.append(Spacer(1, 8))
        story.append(Paragraph("📦  Our AI Product Portfolio", s["section_header"]))
        story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY, spaceAfter=6))
        story.append(Paragraph(
            "The following products from our catalog are available for deployment:",
            s["body"]
        ))
        prod_rows = [[
            Paragraph("<b>Product</b>", s["label"]),
            Paragraph("<b>Category</b>", s["label"]),
            Paragraph("<b>Business Value</b>", s["label"]),
            Paragraph("<b>Price</b>", s["label"]),
        ]]
        for p in ai_products[:8]:
            price = p.get("pricing", 0)
            unit = p.get("unit", "yr")
            prod_rows.append([
                Paragraph(f"<b>{p.get('serviceName', '?')}</b>", ParagraphStyle(
                    "ProdName", fontName="Helvetica-Bold", fontSize=9, textColor=STEEL_BLUE
                )),
                Paragraph(p.get("category", "?"), s["tag"]),
                Paragraph(p.get("businessValue", "?"), s["body"]),
                Paragraph(f"${price:,}/{unit}", ParagraphStyle(
                    "Price", fontName="Helvetica-Bold", fontSize=9, textColor=GREEN
                )),
            ])
        prod_table = Table(prod_rows, colWidths=[2.1 * inch, 1.5 * inch, 2.3 * inch, 1.1 * inch])
        prod_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), STEEL_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(prod_table)

    story.append(Spacer(1, 16))

    # ── FOOTER ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=GOLD, spaceAfter=8))
    footer_data = [[
        Paragraph(
            "<b>SalesIntel AI Copilot</b> — Powered by Agentic RAG + MongoDB Intelligence",
            ParagraphStyle("Footer", fontName="Helvetica", fontSize=8, textColor=MID_GRAY)
        ),
        Paragraph(
            f"Generated: {today}  |  Confidential",
            ParagraphStyle("FooterRight", fontName="Helvetica", fontSize=8, textColor=MID_GRAY, alignment=2)
        ),
    ]]
    footer_table = Table(footer_data, colWidths=[3.5 * inch, 3.5 * inch])
    footer_table.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 2)]))
    story.append(footer_table)

    doc.build(story)
    print(f"[pdf_generator] PDF generated: {output_path}")
    return output_path
