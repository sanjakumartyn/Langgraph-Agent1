from typing import TypedDict, Optional, Dict, Any, List, Annotated
import operator
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


def take_last(left: Any, right: Any) -> Any:
    return right if right is not None else left


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


def merge_lists(left: List[Any], right: List[Any]) -> List[Any]:
    res = []
    seen = set()
    for item in (left or []) + (right or []):
        if item not in seen:
            res.append(item)
            seen.add(item)
    return res


class AgentState(TypedDict):
    company_name: Annotated[str, take_last]
    company_website: Annotated[Optional[str], take_last]

    website_data: Annotated[Dict[str, Any], merge_dicts]
    product_menu_data: Annotated[Dict[str, Any], merge_dicts]
    wikipedia_data: Annotated[Dict[str, Any], merge_dicts]
    wikidata_data: Annotated[Dict[str, Any], merge_dicts]
    news_data: Annotated[Dict[str, Any], merge_dicts]
    rss_data: Annotated[Dict[str, Any], merge_dicts]

    website_summary: Annotated[Dict[str, Any], merge_dicts]
    wikipedia_summary: Annotated[Dict[str, Any], merge_dicts]
    wikidata_summary: Annotated[Dict[str, Any], merge_dicts]
    news_summary: Annotated[Dict[str, Any], merge_dicts]
    rss_summary: Annotated[Dict[str, Any], merge_dicts]

    validated_evidence: Annotated[Dict[str, Any], merge_dicts]
    final_result: Annotated[Dict[str, Any], merge_dicts]
    errors: Annotated[List[str], merge_lists]


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

    # Parallel branching outgoing from guess_company_website
    graph.add_edge("guess_company_website", "scrape_company_website")
    graph.add_edge("guess_company_website", "fetch_wikipedia_data")
    graph.add_edge("guess_company_website", "fetch_wikidata_data")
    graph.add_edge("guess_company_website", "fetch_news_data")
    graph.add_edge("guess_company_website", "fetch_rss_data")

    # Branch 1: Scraping website and summarizing
    graph.add_edge("scrape_company_website", "extract_product_menu_data")
    graph.add_edge("extract_product_menu_data", "extract_website_summary")

    # Branch 2: Wikipedia data extraction
    graph.add_edge("fetch_wikipedia_data", "extract_wikipedia_summary")

    # Branch 3: Wikidata data extraction
    graph.add_edge("fetch_wikidata_data", "extract_wikidata_summary")

    # Branch 4: News data extraction
    graph.add_edge("fetch_news_data", "extract_news_summary")

    # Branch 5: RSS feed matching and extraction
    graph.add_edge("fetch_rss_data", "extract_rss_summary")

    # Merge barrier joining all branches at validate_evidence
    graph.add_edge("extract_website_summary", "validate_evidence")
    graph.add_edge("extract_wikipedia_summary", "validate_evidence")
    graph.add_edge("extract_wikidata_summary", "validate_evidence")
    graph.add_edge("extract_news_summary", "validate_evidence")
    graph.add_edge("extract_rss_summary", "validate_evidence")

    # Final steps
    graph.add_edge("validate_evidence", "generate_final_intelligence")
    graph.add_edge("generate_final_intelligence", END)

    return graph.compile()


async def run_company_agent(company_name: str, company_website: Optional[str] = None):
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

    result = await app.ainvoke(initial_state)
    return result