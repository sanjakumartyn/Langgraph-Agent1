from typing import TypedDict, Optional, Dict, Any, List
from langgraph.graph import StateGraph, END

from app.nodes import (
    guess_company_website,
    scrape_company_website,
    extract_product_menu_data,
    extract_website_summary,
    fetch_wikipedia_data,
    extract_wikipedia_summary,
    fetch_wikidata_data,
    extract_wikidata_summary,
    fetch_news_data,
    extract_news_summary,
    fetch_rss_data,
    extract_rss_summary,
    validate_evidence,
    generate_final_intelligence,
)


class AgentState(TypedDict):
    company_name: str
    company_website: Optional[str]

    website_data: Dict[str, Any]
    product_menu_data: Dict[str, Any]
    wikipedia_data: Dict[str, Any]
    wikidata_data: Dict[str, Any]
    news_data: Dict[str, Any]
    rss_data: Dict[str, Any]

    website_summary: Dict[str, Any]
    wikipedia_summary: Dict[str, Any]
    wikidata_summary: Dict[str, Any]
    news_summary: Dict[str, Any]
    rss_summary: Dict[str, Any]

    validated_evidence: Dict[str, Any]
    final_result: Dict[str, Any]
    errors: List[str]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("guess_company_website", guess_company_website)
    graph.add_node("scrape_company_website", scrape_company_website)
    graph.add_node("extract_product_menu_data", extract_product_menu_data)
    graph.add_node("extract_website_summary", extract_website_summary)

    graph.add_node("fetch_wikipedia_data", fetch_wikipedia_data)
    graph.add_node("extract_wikipedia_summary", extract_wikipedia_summary)

    graph.add_node("fetch_wikidata_data", fetch_wikidata_data)
    graph.add_node("extract_wikidata_summary", extract_wikidata_summary)

    graph.add_node("fetch_news_data", fetch_news_data)
    graph.add_node("extract_news_summary", extract_news_summary)

    graph.add_node("fetch_rss_data", fetch_rss_data)
    graph.add_node("extract_rss_summary", extract_rss_summary)

    graph.add_node("validate_evidence", validate_evidence)
    graph.add_node("generate_final_intelligence", generate_final_intelligence)

    graph.set_entry_point("guess_company_website")

    graph.add_edge("guess_company_website", "scrape_company_website")
    graph.add_edge("scrape_company_website", "extract_product_menu_data")
    graph.add_edge("extract_product_menu_data", "extract_website_summary")

    graph.add_edge("extract_website_summary", "fetch_wikipedia_data")
    graph.add_edge("fetch_wikipedia_data", "extract_wikipedia_summary")

    graph.add_edge("extract_wikipedia_summary", "fetch_wikidata_data")
    graph.add_edge("fetch_wikidata_data", "extract_wikidata_summary")

    graph.add_edge("extract_wikidata_summary", "fetch_news_data")
    graph.add_edge("fetch_news_data", "extract_news_summary")

    graph.add_edge("extract_news_summary", "fetch_rss_data")
    graph.add_edge("fetch_rss_data", "extract_rss_summary")

    graph.add_edge("extract_rss_summary", "validate_evidence")
    graph.add_edge("validate_evidence", "generate_final_intelligence")
    graph.add_edge("generate_final_intelligence", END)

    return graph.compile()


def run_company_agent(company_name: str, company_website: Optional[str] = None):
    app = build_graph()

    initial_state: AgentState = {
        "company_name": company_name,
        "company_website": company_website,

        "website_data": {},
        "product_menu_data": {},
        "wikipedia_data": {},
        "wikidata_data": {},
        "news_data": {},
        "rss_data": {},

        "website_summary": {},
        "wikipedia_summary": {},
        "wikidata_summary": {},
        "news_summary": {},
        "rss_summary": {},

        "validated_evidence": {},
        "final_result": {},
        "errors": [],
    }

    result = app.invoke(initial_state)
    return result