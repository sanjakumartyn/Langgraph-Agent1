import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from .tools import vector_search_tool, get_company_dossier_tool, live_news_lookup_tool

class RAGState(TypedDict):
    query: str
    steps_taken: List[str]
    retrieved_context: List[str]
    next_action: str
    current_param: str
    final_answer: str

# Initialize LLM
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


def clean_json_response(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()
    elif content.startswith("```"):
        content = content.replace("```", "").strip()
    return content


async def planner(state: RAGState) -> Dict[str, Any]:
    prompt = f"""
You are the Planning Node of an Agentic RAG system.
Your job is to analyze the user's query and decide the best FIRST step.

Query: {state["query"]}

Available Actions:
1. "search_vectors": Use this to search across reports for specific concepts or topics (e.g. "cloud priorities", "competitors").
2. "fetch_dossier": Use this if you need the full, complete intelligence data for a specific company (e.g. "Zoho", "Nestle").
3. "live_lookup": Use this if you need recent news/updates or Wikipedia details for a company.
4. "synthesize": Use this ONLY if the query is trivial or can be answered immediately.

Return ONLY valid JSON with keys:
"action": The action string (one of the 4 above)
"parameter": The search query or company name to pass to the tool
"reason": A brief reason for this decision

Example:
{{
  "action": "fetch_dossier",
  "parameter": "Zoho",
  "reason": "Need to look up full profile for Zoho"
}}
"""
    try:
        response = await llm.ainvoke(prompt)
        cleaned = clean_json_response(response.content)
        parsed = json.loads(cleaned)
        return {
            **state,
            "next_action": parsed.get("action", "synthesize"),
            "current_param": parsed.get("parameter", ""),
            "steps_taken": state.get("steps_taken", []) + [f"Planned: {parsed.get('reason')}"]
        }
    except Exception as e:
        return {
            **state,
            "next_action": "synthesize",
            "current_param": "",
            "steps_taken": state.get("steps_taken", []) + [f"Planner error: {str(e)}"]
        }


async def execute_tool(state: RAGState) -> Dict[str, Any]:
    action = state["next_action"]
    param = state["current_param"]
    
    context = ""
    step_desc = f"Executed {action} with parameter '{param}'"
    
    if action == "search_vectors":
        context = await vector_search_tool(param)
    elif action == "fetch_dossier":
        context = await get_company_dossier_tool(param)
    elif action == "live_lookup":
        context = await live_news_lookup_tool(param)
        
    return {
        **state,
        "retrieved_context": state.get("retrieved_context", []) + [context],
        "steps_taken": state.get("steps_taken", []) + [step_desc]
    }


async def critique(state: RAGState) -> Dict[str, Any]:
    # If the tool was synthesize, go directly to synthesis
    if state["next_action"] == "synthesize":
        return state
        
    prompt = f"""
You are the Critique & Self-Correction Node of an Agentic RAG system.
Review the original query, steps taken, and the context retrieved so far. Decide if you have enough information to answer the query, or if you need to fetch more data.

Original Query: {state["query"]}
Steps Taken: {json.dumps(state.get("steps_taken", []), indent=2)}
Retrieved Context so far:
{chr(10).join(state.get("retrieved_context", []))[:15000]}

Decide what to do next. Choose from:
1. "search_vectors" (parameter: search term)
2. "fetch_dossier" (parameter: company name)
3. "live_lookup" (parameter: company name)
4. "synthesize" (parameter: empty string) - Use this if you have sufficient data to answer the query.

Limit: Try to synthesize within 4 steps max to avoid infinite loops.

Return ONLY valid JSON with keys:
"action": The next action string
"parameter": The query/company parameter
"reason": Why you chose this step
"""
    try:
        # Prevent infinite loop by capping steps
        if len(state.get("steps_taken", [])) >= 5:
            return {
                **state,
                "next_action": "synthesize",
                "current_param": ""
            }
            
        response = await llm.ainvoke(prompt)
        cleaned = clean_json_response(response.content)
        parsed = json.loads(cleaned)
        return {
            **state,
            "next_action": parsed.get("action", "synthesize"),
            "current_param": parsed.get("parameter", ""),
            "steps_taken": state.get("steps_taken", []) + [f"Critique: {parsed.get('reason')}"]
        }
    except Exception as e:
        return {
            **state,
            "next_action": "synthesize",
            "current_param": "",
            "steps_taken": state.get("steps_taken", []) + [f"Critique error: {str(e)}"]
        }


async def synthesizer(state: RAGState) -> Dict[str, Any]:
    prompt = f"""
You are the Synthesizer and Output Generator of an Agentic RAG system.
Your job is to answer the user's original query using ONLY the retrieved facts and context. Do not make up facts.

Original Query: {state["query"]}

Retrieved Context:
{chr(10).join(state.get("retrieved_context", []))}

Steps Taken during Research:
{chr(10).join(state.get("steps_taken", []))}

Write a clear, structured, and informative answer in Markdown format. Cite the sources or context where appropriate.
"""
    try:
        response = await llm.ainvoke(prompt)
        return {
            **state,
            "final_answer": response.content
        }
    except Exception as e:
        err_msg = str(e)
        if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
            friendly_msg = (
                "⚠️ **Gemini API Rate Limit Exceeded (429 Resource Exhausted)**:\n\n"
                "Your Google API key has exceeded its free tier rate limits or quota. "
                "Please wait 1–2 minutes before trying again. If you need higher limits, "
                "consider upgrading your Gemini API key to a paid tier in your `.env` file."
            )
            return {
                **state,
                "final_answer": friendly_msg
            }
        return {
            **state,
            "final_answer": f"Error generating final response: {err_msg}"
        }


def router(state: RAGState) -> str:
    """Decide whether to execute tool or go to synthesizer."""
    if state["next_action"] == "synthesize":
        return "synthesizer"
    return "execute_tool"


def build_rag_graph():
    workflow = StateGraph(RAGState)
    
    workflow.add_node("planner", planner)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("critique", critique)
    workflow.add_node("synthesizer", synthesizer)
    
    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "execute_tool")
    workflow.add_edge("execute_tool", "critique")
    
    # Conditional edge after critique
    workflow.add_conditional_edges(
        "critique",
        router,
        {
            "execute_tool": "execute_tool",
            "synthesizer": "synthesizer"
        }
    )
    
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()


async def run_rag_agent(query: str) -> Dict[str, Any]:
    app = build_rag_graph()
    initial_state: RAGState = {
        "query": query,
        "steps_taken": [],
        "retrieved_context": [],
        "next_action": "",
        "current_param": "",
        "final_answer": ""
    }
    result = await app.ainvoke(initial_state)
    return result
