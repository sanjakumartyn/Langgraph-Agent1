import os
import json
import time
import asyncio
import requests
import feedparser
from typing import Dict, Any

from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler
from playwright.async_api import async_playwright
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


def safe_get(url: str, params=None, headers=None, timeout=20):
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
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


def call_llm_json(prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    for attempt in range(4):
        try:
            response = llm.invoke(prompt)
            content = clean_json_response(response.content)
            return json.loads(content)
        except Exception as e:
            if attempt == 3:
                fallback["llm_error"] = str(e)
                return fallback

            time.sleep(2 ** attempt)


def guess_company_website(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("company_website"):
        return state

    company = state["company_name"].strip()
    clean = company.lower().replace(" ", "").replace("&", "and")

    return {
        **state,
        "company_website": f"https://www.{clean}.com"
    }


async def crawl_url_with_crawler(crawler, url: str) -> Dict[str, Any]:
    try:
        result = await crawler.arun(url)

        return {
            "url": url,
            "success": True,
            "markdown": result.markdown[:8000] if result.markdown else ""
        }

    except Exception as e:
        return {
            "url": url,
            "success": False,
            "markdown": "",
            "error": str(e)
        }


async def scrape_company_website_async(state: Dict[str, Any]) -> Dict[str, Any]:
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
        f"{base}/businesses",
        f"{base}/products",
        f"{base}/services",
        f"{base}/solutions",
        f"{base}/leadership",
        f"{base}/careers",
        f"{base}/sustainability",
        f"{base}/contact",
    ]

    pages = []

    async with AsyncWebCrawler() as crawler:
        for url in urls:
            page = await crawl_url_with_crawler(crawler, url)
            pages.append(page)

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


def scrape_company_website(state: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(scrape_company_website_async(state))


async def extract_product_menu_async(state: Dict[str, Any]) -> Dict[str, Any]:
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
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                ],
            )

            page = await browser.new_page()
            await page.goto(website, wait_until="domcontentloaded", timeout=60000)

            try:
                await page.get_by_text("ACCEPT ALL COOKIES").click(timeout=5000)
            except Exception:
                pass

            possible_menu_names = [
                "Products",
                "Businesses",
                "Services",
                "Solutions",
                "What We Do",
            ]

            menu_opened = False

            for menu_name in possible_menu_names:
                try:
                    locator = page.get_by_text(menu_name, exact=False).first
                    await locator.hover(timeout=5000)
                    await page.wait_for_timeout(1500)
                    menu_opened = True
                    break
                except Exception:
                    continue

            body_text = await page.locator("body").inner_text()

            links = await page.locator("a").evaluate_all(
                """els => els.map(a => ({
                    text: a.innerText,
                    href: a.href
                })).filter(x => x.text && x.href)"""
            )

            await browser.close()

        prompt = f"""
You are a product and service menu extraction agent.

Extract ONLY product, service, business segment, solution, and offering-related data.

Return ONLY valid JSON.

Schema:
{{
  "products_or_services": [
    {{
      "category": "",
      "subcategories": [],
      "source": "product_menu_playwright"
    }}
  ],
  "important_product_links": [
    {{
      "title": "",
      "url": ""
    }}
  ]
}}

Rules:
- Do not hallucinate.
- Use only the provided menu text and links.
- For diversified companies, use business segments as products/services.
- Ignore login, privacy, cookies, careers unless relevant as service lines.

MENU OPENED:
{menu_opened}

BODY TEXT:
{body_text[:12000]}

LINKS:
{json.dumps(links[:120], indent=2)}
"""

        product_json = call_llm_json(
            prompt,
            {
                "products_or_services": [],
                "important_product_links": [],
            },
        )

        return {
            **state,
            "product_menu_data": {
                "source": "product_menu_playwright",
                "found": True,
                "data": product_json,
            },
        }

    except Exception as e:
        errors.append(f"Product menu extraction failed: {str(e)}")

        return {
            **state,
            "product_menu_data": {
                "source": "product_menu_playwright",
                "found": False,
                "data": {},
            },
            "errors": errors,
        }


def extract_product_menu_data(state: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(extract_product_menu_async(state))


def extract_website_summary(state: Dict[str, Any]) -> Dict[str, Any]:
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

    summary = call_llm_json(
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


def fetch_wikipedia_data(state: Dict[str, Any]) -> Dict[str, Any]:
    company = state["company_name"]

    status, search_data = safe_get(
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

    status, summary = safe_get(
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


def extract_wikipedia_summary(state: Dict[str, Any]) -> Dict[str, Any]:
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

    summary = call_llm_json(
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


def fetch_wikidata_data(state: Dict[str, Any]) -> Dict[str, Any]:
    company = state["company_name"]

    status, search_data = safe_get(
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

    status, entity_data = safe_get(
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


def extract_wikidata_summary(state: Dict[str, Any]) -> Dict[str, Any]:
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

    summary = call_llm_json(
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


def fetch_news_data(state: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        return {
            **state,
            "news_data": {
                "source": "newsapi",
                "found": False,
                "error": "NEWS_API_KEY missing",
            },
        }

    company = state["company_name"]

    status, data = safe_get(
        "https://newsapi.org/v2/everything",
        params={
            "q": f'"{company}"',
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": api_key,
            "domains": (
                "techcrunch.com,"
                "business-standard.com,"
                "economictimes.indiatimes.com,"
                "thehindubusinessline.com,"
                "reuters.com"
            ),
        },
    )

    articles = []

    for article in data.get("articles", []):
        articles.append(
            {
                "title": article.get("title"),
                "source": article.get("source", {}).get("name"),
                "url": article.get("url"),
                "publishedAt": article.get("publishedAt"),
                "description": article.get("description"),
            }
        )

    return {
        **state,
        "news_data": {
            "source": "newsapi",
            "found": bool(articles),
            "articles": articles,
        },
    }


def extract_news_summary(state: Dict[str, Any]) -> Dict[str, Any]:
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

    summary = call_llm_json(
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


def fetch_rss_data(state: Dict[str, Any]) -> Dict[str, Any]:
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
            feed = feedparser.parse(feed_url)

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


def extract_rss_summary(state: Dict[str, Any]) -> Dict[str, Any]:
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

    summary = call_llm_json(
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


def validate_evidence(state: Dict[str, Any]) -> Dict[str, Any]:
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

    validated = call_llm_json(
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


def generate_final_intelligence(state: Dict[str, Any]) -> Dict[str, Any]:
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
  "missing_data": []
}}

VALIDATED EVIDENCE:
{json.dumps(state.get("validated_evidence", {}), indent=2)}
"""

    final_json = call_llm_json(
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
        },
    )

    return {
        **state,
        "final_result": final_json,
    }