import os
import json
import time
import asyncio
import httpx
import feedparser
from typing import Dict, Any, List
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler
from playwright.async_api import async_playwright
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

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
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

llm = get_llm()


async def safe_get_async(url: str, params=None, headers=None, timeout=20):
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": str(e)}


def clean_json_response(content: str) -> str:
    content = content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()
    elif content.startswith("```"):
        content = content.replace("```", "").strip()

    return content


async def call_llm_json_async(prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    for attempt in range(4):
        try:
            response = await llm.ainvoke(prompt)
            content = clean_json_response(response.content)
            return json.loads(content)
        except Exception as e:
            if attempt == 3:
                fallback["llm_error"] = str(e)
                return fallback

            await asyncio.sleep(2 ** attempt)


async def guess_company_website(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("company_website"):
        return state

    company = state["company_name"].strip()
    clean = company.lower().replace(" ", "").replace("&", "and")

    return {
        **state,
        "company_website": f"https://www.{clean}.com"
    }


async def fallback_scrape(url: str) -> str:
    """Fallback scraper using standard HTTP requests and BeautifulSoup."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Decompose non-content tags
                for tag in soup(["script", "style", "header", "footer", "nav", "noscript"]):
                    tag.decompose()
                
                # Extract text
                text = soup.get_text(separator="\n")
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                cleaned_text = "\n".join(chunk for chunk in chunks if chunk)
                return cleaned_text[:8000]
            else:
                raise Exception(f"HTTP status {response.status_code}")
    except Exception as e:
        raise Exception(f"Fallback scrape failed: {str(e)}")


async def crawl_url_with_crawler_retry(crawler, url: str) -> Dict[str, Any]:
    """Crawl a URL with up to 3 retries, falling back to a requests/bs4 scraper."""
    for attempt in range(3):
        try:
            result = await crawler.arun(url)
            if result and result.markdown:
                return {
                    "url": url,
                    "success": True,
                    "markdown": result.markdown[:8000]
                }
        except Exception as e:
            if attempt == 2:
                # final attempt failed, run fallback scraper
                try:
                    text = await fallback_scrape(url)
                    return {
                        "url": url,
                        "success": True,
                        "markdown": text
                    }
                except Exception as fb_err:
                    return {
                        "url": url,
                        "success": False,
                        "markdown": "",
                        "error": f"Crawl4AI failed: {str(e)}. Fallback failed: {str(fb_err)}"
                    }
            await asyncio.sleep(2 ** attempt)
            
    # If arun succeeded but markdown was empty, try fallback scraper
    try:
        text = await fallback_scrape(url)
        return {
            "url": url,
            "success": True,
            "markdown": text
        }
    except Exception as fb_err:
        return {
            "url": url,
            "success": False,
            "markdown": "",
            "error": f"Empty crawl results. Fallback failed: {str(fb_err)}"
        }


async def scrape_company_website(state: Dict[str, Any]) -> Dict[str, Any]:
    website = state.get("company_website")
    errors = state["errors"][:]

    if not website:
        errors.append("Company website not available")
        return {
            **state,
            "website_data": {
                "source": "company_website",
                "pages": []
            },
            "errors": errors
        }

    base = website.rstrip("/")

    urls = [
        base,
        f"{base}/about",
    ]

    pages = []

    async with AsyncWebCrawler() as crawler:
        tasks = [crawl_url_with_crawler_retry(crawler, url) for url in urls]
        pages = await asyncio.gather(*tasks)

    for page in pages:
        if not page.get("success"):
            errors.append(
                f"Website scrape failed: {page.get('url')} - {page.get('error')}"
            )

    return {
        **state,
        "website_data": {
            "source": "company_website",
            "pages": pages
        },
        "errors": errors
    }


async def extract_product_menu_data(state: Dict[str, Any]) -> Dict[str, Any]:
    website = state.get("company_website")
    errors = state["errors"][:]

    if not website:
        return {
            **state,
            "product_menu_data": {
                "source": "product_menu_playwright",
                "found": False,
                "data": {}
            }
        }

    try:
        # OPTIMIZED: Playwright menu extraction disabled for speed
        return {
            **state,
            "product_menu_data": {
                "source": "product_menu_playwright",
                "found": False,
                "data": {}
            }
        }
    except Exception as e:
        errors.append(f"Playwright menu extraction failed: {str(e)}")
        return {
            **state,
            "product_menu_data": {
                "source": "product_menu_playwright",
                "found": False,
                "data": {},
            },
            "errors": errors,
        }


async def extract_website_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    pages = state["website_data"].get("pages", [])

    website_text = "\n\n".join(
        f"URL: {page.get('url')}\n{page.get('markdown', '')}"
        for page in pages
        if page.get("success") and page.get("markdown")
    )

    prompt = f"""
You are a website intelligence extraction agent.

Use ONLY company website content and product menu data.

Return ONLY valid JSON.

Schema:
{{
  "website": "",
  "company_summary": "",
  "products_or_services": [],
  "leadership_or_contact_signals": [],
  "locations": [],
  "business_priorities": [],
  "technology_signals": [],
  "sustainability_signals": [],
  "source_evidence": [
    {{
      "claim": "",
      "source": "company_website",
      "url": ""
    }}
  ]
}}

Rules:
- Extract products/services mainly from PRODUCT MENU DATA and website pages.
- Do not use news, Wikipedia, or Wikidata here.
- Do not hallucinate.

COMPANY NAME:
{state["company_name"]}

COMPANY WEBSITE:
{state.get("company_website")}

PRODUCT MENU DATA:
{json.dumps(state.get("product_menu_data", {}), indent=2)[:10000]}

WEBSITE TEXT:
{website_text[:25000]}
"""

    summary = await call_llm_json_async(
        prompt,
        {
            "website": state.get("company_website"),
            "company_summary": "",
            "products_or_services": [],
            "leadership_or_contact_signals": [],
            "locations": [],
            "business_priorities": [],
            "technology_signals": [],
            "sustainability_signals": [],
            "source_evidence": [],
        },
    )

    return {
        **state,
        "website_summary": summary,
    }


async def fetch_wikipedia_data(state: Dict[str, Any]) -> Dict[str, Any]:
    company = state["company_name"]

    status, search_data = await safe_get_async(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": company,
            "format": "json",
        },
    )

    if status != 200 or not search_data.get("query", {}).get("search"):
        return {
            **state,
            "wikipedia_data": {
                "source": "wikipedia",
                "found": False,
                "data": {},
            },
        }

    title = search_data["query"]["search"][0]["title"]

    status, summary = await safe_get_async(
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + title.replace(" ", "_")
    )

    return {
        **state,
        "wikipedia_data": {
            "source": "wikipedia",
            "found": status == 200,
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "data": summary,
        },
    }


async def extract_wikipedia_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are a Wikipedia company profile extraction agent.

Use ONLY Wikipedia data.

Return ONLY valid JSON.

Schema:
{{
  "company_history": [],
  "founded": "",
  "founders": [],
  "headquarters": "",
  "industry": "",
  "key_milestones": [],
  "source_evidence": [
    {{
      "claim": "",
      "source": "wikipedia",
      "url": ""
    }}
  ]
}}

WIKIPEDIA DATA:
{json.dumps(state.get("wikipedia_data", {}), indent=2)[:8000]}
"""

    summary = await call_llm_json_async(
        prompt,
        {
            "company_history": [],
            "founded": "",
            "founders": [],
            "headquarters": "",
            "industry": "",
            "key_milestones": [],
            "source_evidence": [],
        },
    )

    return {
        **state,
        "wikipedia_summary": summary,
    }


async def fetch_wikidata_data(state: Dict[str, Any]) -> Dict[str, Any]:
    company = state["company_name"]

    status, search_data = await safe_get_async(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbsearchentities",
            "search": company,
            "language": "en",
            "format": "json",
            "limit": 1,
        },
    )

    if status != 200 or not search_data.get("search"):
        return {
            **state,
            "wikidata_data": {
                "source": "wikidata",
                "found": False,
                "data": {},
            },
        }

    entity = search_data["search"][0]
    entity_id = entity.get("id")

    status, entity_data = await safe_get_async(
        f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    )

    return {
        **state,
        "wikidata_data": {
            "source": "wikidata",
            "found": status == 200,
            "entity_id": entity_id,
            "label": entity.get("label"),
            "description": entity.get("description"),
            "url": entity.get("concepturi"),
            "raw": entity_data,
        },
    }


async def extract_wikidata_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are a Wikidata structured facts extraction agent.

Use ONLY Wikidata data.

Return ONLY valid JSON.

Schema:
{{
  "structured_facts": {{
    "official_name": "",
    "description": "",
    "industry": "",
    "country": "",
    "headquarters": "",
    "inception": "",
    "official_website": ""
  }},
  "source_evidence": [
    {{
      "claim": "",
      "source": "wikidata",
      "url": ""
    }}
  ]
}}

Rules:
- Extract only facts available from Wikidata.
- If values are encoded and unclear, use entity label/description only.
- Do not hallucinate.

WIKIDATA DATA:
{json.dumps(state.get("wikidata_data", {}), indent=2)[:12000]}
"""

    summary = await call_llm_json_async(
        prompt,
        {
            "structured_facts": {},
            "source_evidence": [],
        },
    )

    return {
        **state,
        "wikidata_summary": summary,
    }


async def fetch_mediastack_news_async(company: str) -> List[Dict[str, Any]]:
    """Fetch news from MediaStack API as an additional/fallback source."""
    api_key = os.getenv("MEDIASTACK_API_KEY")
    if not api_key:
        return []

    status, data = await safe_get_async(
        "http://api.mediastack.com/v1/news",
        params={
            "access_key": api_key,
            "keywords": company,
            "languages": "en",
            "limit": 10
        }
    )

    articles = []
    if status == 200 and data and "data" in data:
        for article in data["data"]:
            articles.append({
                "title": article.get("title"),
                "source": article.get("source"),
                "url": article.get("url"),
                "publishedAt": article.get("published_at"),
                "description": article.get("description")
            })
    return articles


async def fetch_news_data(state: Dict[str, Any]) -> Dict[str, Any]:
    newsapi_key = os.getenv("NEWS_API_KEY")
    company = state["company_name"]
    newsapi_articles = []
    errors = state["errors"][:]

    if newsapi_key:
        status, data = await safe_get_async(
            "https://newsapi.org/v2/everything",
            params={
                "q": f'"{company}"',
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "apiKey": newsapi_key,
                "domains": (
                    "techcrunch.com,"
                    "business-standard.com,"
                    "economictimes.indiatimes.com,"
                    "thehindubusinessline.com,"
                    "reuters.com"
                ),
            },
        )
        if status == 200 and data:
            for article in data.get("articles", []):
                newsapi_articles.append(
                    {
                        "title": article.get("title"),
                        "source": article.get("source", {}).get("name"),
                        "url": article.get("url"),
                        "publishedAt": article.get("publishedAt"),
                        "description": article.get("description"),
                    }
                )
        else:
            errors.append(f"NewsAPI error: code {status}")
    else:
        errors.append("NEWS_API_KEY missing")

    # Integrate MediaStack API
    mediastack_articles = []
    try:
        mediastack_articles = await fetch_mediastack_news_async(company)
    except Exception as e:
        errors.append(f"MediaStack error: {str(e)}")

    # Merge and deduplicate articles by URL
    seen_urls = set()
    merged_articles = []
    for article in (newsapi_articles + mediastack_articles):
        url = article.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            merged_articles.append(article)

    return {
        **state,
        "news_data": {
            "source": "news_feeds",
            "found": bool(merged_articles),
            "articles": merged_articles,
        },
        "errors": errors,
    }


async def extract_news_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are a market news intelligence agent.

Use ONLY NewsAPI data.

Return ONLY valid JSON.

Schema:
{{
  "market_news": [],
  "business_priorities": [],
  "pain_points": [],
  "technology_signals": [],
  "market_signals": [],
  "source_evidence": [
    {{
      "claim": "",
      "source": "newsapi",
      "url": ""
    }}
  ]
}}

Rules:
- Use news only for recent developments, risks, priorities, and market signals.
- Do not use news to define core products unless the article clearly states it.

NEWS DATA:
{json.dumps(state.get("news_data", {}), indent=2)[:12000]}
"""

    summary = await call_llm_json_async(
        prompt,
        {
            "market_news": [],
            "business_priorities": [],
            "pain_points": [],
            "technology_signals": [],
            "market_signals": [],
            "source_evidence": [],
        },
    )

    return {
        **state,
        "news_summary": summary,
    }


async def fetch_rss_data(state: Dict[str, Any]) -> Dict[str, Any]:
    company = state["company_name"].lower()

    rss_feeds = [
        "https://techcrunch.com/feed/",
        "https://www.thehindubusinessline.com/feeder/default.rss",
        "https://www.business-standard.com/rss/latest.rss",
        "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
    ]

    matched_articles = []
    errors = state["errors"][:]

    for feed_url in rss_feeds:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(feed_url)
                if resp.status_code == 200:
                    # Parse using feedparser asynchronously without blocking
                    feed = feedparser.parse(resp.text)
                    for entry in feed.entries[:30]:
                        title = entry.get("title", "")
                        summary = entry.get("summary", "")
                        link = entry.get("link", "")
                        published = entry.get("published", "")

                        content = f"{title} {summary}".lower()

                        if company in content:
                            matched_articles.append(
                                {
                                    "title": title,
                                    "summary": summary[:500],
                                    "url": link,
                                    "published": published,
                                    "feed": feed_url,
                                }
                            )
        except Exception as e:
            errors.append(f"RSS failed: {feed_url} - {str(e)}")

    return {
        **state,
        "rss_data": {
            "source": "rss_feeds",
            "found": bool(matched_articles),
            "articles": matched_articles[:10],
        },
        "errors": errors,
    }


async def extract_rss_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are an RSS news validation agent.

Use ONLY RSS data.

Return ONLY valid JSON.

Schema:
{{
  "rss_articles": [],
  "validated_news_signals": [],
  "source_evidence": [
    {{
      "claim": "",
      "source": "rss",
      "url": ""
    }}
  ]
}}

RSS DATA:
{json.dumps(state.get("rss_data", {}), indent=2)[:10000]}
"""

    summary = await call_llm_json_async(
        prompt,
        {
            "rss_articles": [],
            "validated_news_signals": [],
            "source_evidence": [],
        },
    )

    return {
        **state,
        "rss_summary": summary,
    }


def calculate_confidence_score(state: Dict[str, Any]) -> int:
    score = 0
    # Check website summary
    web_summary = state.get("website_summary", {})
    if web_summary and web_summary.get("company_summary"):
        score += 35
    
    # Check Wikipedia
    wiki_data = state.get("wikipedia_data", {})
    if wiki_data and wiki_data.get("found"):
        score += 25
        
    # Check Wikidata
    wikidata_data = state.get("wikidata_data", {})
    if wikidata_data and wikidata_data.get("found"):
        score += 15
        
    # Check News
    news_data = state.get("news_data", {})
    if news_data and news_data.get("found") and news_data.get("articles"):
        score += 15
        
    # Check RSS
    rss_data = state.get("rss_data", {})
    if rss_data and rss_data.get("found") and rss_data.get("articles"):
        score += 10

    # Adjust for missing data
    validated = state.get("validated_evidence", {})
    if validated:
        missing_data = validated.get("missing_data", [])
        score -= len(missing_data) * 5

    return max(0, min(100, score))


async def validate_evidence(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are an evidence validation and merge agent.

You must analyze every source separately and then merge them.

Return ONLY valid JSON.

Schema:
{{
  "validated_company_profile": {{
    "company_name": "",
    "website": "",
    "industry": "",
    "summary": "",
    "confidence": 0
  }},
  "validated_products_or_services": [],
  "validated_history": [],
  "validated_locations": [],
  "validated_leadership_or_contact": [],
  "validated_business_priorities": [],
  "validated_pain_points": [],
  "validated_technology_signals": [],
  "validated_market_news": [],
  "all_evidence": [
    {{
      "claim": "",
      "source": "",
      "url": ""
    }}
  ],
  "missing_data": []
}}

Rules:
- You MUST use all available summaries.
- Website summary is primary for products/services, leadership, contact, and official website.
- Wikipedia summary is primary for history.
- Wikidata summary is primary for structured facts.
- News summary is primary for recent news, pain points, market signals.
- RSS summary is used for additional news validation.
- Do not hallucinate.

WEBSITE SUMMARY:
{json.dumps(state.get("website_summary", {}), indent=2)}

WIKIPEDIA SUMMARY:
{json.dumps(state.get("wikipedia_summary", {}), indent=2)}

WIKIDATA SUMMARY:
{json.dumps(state.get("wikidata_summary", {}), indent=2)}

NEWS SUMMARY:
{json.dumps(state.get("news_summary", {}), indent=2)}

RSS SUMMARY:
{json.dumps(state.get("rss_summary", {}), indent=2)}
"""

    validated = await call_llm_json_async(
        prompt,
        {
            "validated_company_profile": {},
            "validated_products_or_services": [],
            "validated_history": [],
            "validated_locations": [],
            "validated_leadership_or_contact": [],
            "validated_business_priorities": [],
            "validated_pain_points": [],
            "validated_technology_signals": [],
            "validated_market_news": [],
            "all_evidence": [],
            "missing_data": [],
        },
    )

    return {
        **state,
        "validated_evidence": validated,
    }


async def generate_final_intelligence(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are a final company intelligence report generator.

Use ONLY VALIDATED EVIDENCE.

Return ONLY valid JSON.

JSON schema:
{{
  "company_name": "",
  "website": "",
  "industry": "",
  "company_summary": "",
  "products_or_services": [],
  "leadership_or_contact_signals": [],
  "locations": [],
  "business_priorities": [],
  "pain_points": [],
  "technology_signals": [],
  "market_news": [],
  "company_history": [],
  "source_evidence": [
    {{
      "claim": "",
      "source": "",
      "url": ""
    }}
  ],
  "confidence_score": 0,
  "missing_data": [],
  "strategic_fit_score": 0,
  "ai_needs_prediction": [],
  "meeting_preparation": {{
    "suggested_discussion_points": [],
    "potential_objections": [],
    "relevant_case_studies": [],
    "stakeholders_to_target": []
  }},
  "executive_qbr": {{
    "key_business_priorities": [],
    "growth_initiatives": [],
    "risk_factors": [],
    "buying_signals": []
  }},
  "solution_mapping": [
    {{
      "solution": "",
      "match_percentage": 0,
      "requirement": "",
      "deal_value": ""
    }}
  ],
  "deal_coach": {{
    "recommended_pitch_strategy": ""
  }}
}}

For strategic_fit_score, estimate a score between 1 and 100.
For ai_needs_prediction, predict key requirements or technology needs based on the company's pain points.
For meeting_preparation, suggest discussion points, potential objections, relevant case studies, and list key stakeholders to target.
For executive_qbr, list key business priorities, growth initiatives, risk factors, and buying signals.
For solution_mapping, map matching sales solutions, match percentage, requirements, and estimated deal value.
For deal_coach.recommended_pitch_strategy, provide a short, actionable pitch script/strategy.

VALIDATED EVIDENCE:
{json.dumps(state.get("validated_evidence", {}), indent=2)}
"""

    final_json = await call_llm_json_async(
        prompt,
        {
            "company_name": state["company_name"],
            "website": state.get("company_website"),
            "industry": "",
            "company_summary": "",
            "products_or_services": [],
            "leadership_or_contact_signals": [],
            "locations": [],
            "business_priorities": [],
            "pain_points": [],
            "technology_signals": [],
            "market_news": [],
            "company_history": [],
            "source_evidence": [],
            "confidence_score": 30,
            "missing_data": [],
            "strategic_fit_score": 75,
            "ai_needs_prediction": [],
            "meeting_preparation": {
                "suggested_discussion_points": [],
                "potential_objections": [],
                "relevant_case_studies": [],
                "stakeholders_to_target": []
            },
            "executive_qbr": {
                "key_business_priorities": [],
                "growth_initiatives": [],
                "risk_factors": [],
                "buying_signals": []
            },
            "solution_mapping": [],
            "deal_coach": {
                "recommended_pitch_strategy": ""
            }
        },
    )

    # Compute source confidence score programmatically to ensure precision
    final_json["confidence_score"] = calculate_confidence_score(state)
    if "strategic_fit_score" not in final_json:
        final_json["strategic_fit_score"] = final_json.get("confidence_score", 75)


    return {
        **state,
        "final_result": final_json,
    }