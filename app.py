"""
Last-Mile Delivery Exception Agents - live Streamlit demo.

A LangGraph multi-agent pipeline that triages and resolves last-mile delivery
exceptions: it deduplicates noisy shipment logs, decides a resolution action
against an operational playbook via RAG, escalates to a human when policy
requires it, and drafts a personalized customer notification - with a full
auditable decision trail.

All data (customers, lockers, delivery logs, ground truth, and the operations
playbook) is synthetic, generated for this project.

Requires HF_TOKEN as a Space secret (Settings -> Variables and secrets), a
Hugging Face access token used to authenticate against Hugging Face Inference
Providers (https://router.huggingface.co/v1), which routes chat completion
requests to a hosted open-weight model.
"""
import csv
import json
import os
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Literal, Optional, TypedDict

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, model_validator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Last-Mile Delivery Exception Agents",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
    color: white; padding: 20px 30px; border-radius: 10px; margin-bottom: 20px;
}
.main-header a.eval-link { color: #c8e6c9; text-decoration: underline; }
.main-header a.eval-link:hover { color: #fff; }
.badge-escalated { background:#c62828; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; font-size:.85em; }
.badge-clear     { background:#2e7d32; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; font-size:.85em; }
.trace-box { background:#f5f5f5; color:#1a1a1a; border-left:4px solid #2e7d32; padding:10px 15px; border-radius:4px; margin:6px 0; font-family:monospace; font-size:.85em; }
.message-box { background:#e8f5e9; color:#1a1a1a; border-left:4px solid #1b5e20; padding:14px 18px; border-radius:6px; margin:10px 0; }
div[data-testid="stJson"] { background-color: #1e1e1e !important; }
div[data-testid="stJson"] * { color: #f1f1f1 !important; }
div[data-testid="stDataFrame"] { background-color: #1e1e1e !important; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DELIVERY_LOGS_PATH = os.path.join(DATA_DIR, "delivery_logs.csv")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
CUSTOMERS_DB_PATH = os.path.join(DATA_DIR, "customers.db")
PLAYBOOK_PATH = os.path.join(DATA_DIR, "exception_resolution_playbook.pdf")


def traceable(*args, **kwargs):
    """No-op stand-in for langsmith.traceable - this deployment doesn't ship
    LangSmith credentials, but the decorated node functions are otherwise
    identical to the original notebook implementation."""
    def decorator(fn):
        return fn
    return decorator


# ─── LLM SETUP (Hugging Face Inference Providers, reads HF_TOKEN from Space
# secrets) ──────────────────────────────────────────────────────────────────
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
HF_CHAT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


@st.cache_resource
def get_llms():
    """
    Build the LangChain chat models used by the agents.

    Runs against Hugging Face Inference Providers' OpenAI-compatible endpoint
    (https://router.huggingface.co/v1) rather than OpenAI directly, so every
    Hugging Face account (including free ones) gets a small monthly credit
    pool automatically - no separate billing signup required. Set HF_TOKEN in
    your Hugging Face Space secrets (Settings -> Variables and secrets) to a
    User Access Token from https://huggingface.co/settings/tokens.
    """
    from langchain_openai import ChatOpenAI
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        st.error(
            "No HF_TOKEN found in this Space's environment. Add a Hugging Face "
            "User Access Token under **Settings → Variables and secrets** "
            "(name it `HF_TOKEN`) to run the live pipeline.",
            icon="⚠️",
        )
        st.stop()
    gen = ChatOpenAI(
        model=HF_CHAT_MODEL,
        temperature=0,
        api_key=hf_token,
        base_url=HF_ROUTER_BASE_URL,
    )
    ev = ChatOpenAI(
        model=HF_CHAT_MODEL,
        temperature=0,
        api_key=hf_token,
        base_url=HF_ROUTER_BASE_URL,
    )
    return gen, ev


gen_llm, eval_llm = get_llms()


# ─── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_db_conn():
    conn = sqlite3.connect(CUSTOMERS_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


db_conn = get_db_conn()


@st.cache_data
def load_delivery_logs():
    with open(DELIVERY_LOGS_PATH, "r") as f:
        return list(csv.DictReader(f))


@st.cache_data
def load_ground_truth():
    df = pd.read_csv(GROUND_TRUTH_PATH)
    gt_by_shipment = defaultdict(list)
    for _, row in df.iterrows():
        gt_by_shipment[row["shipment_id"]].append(row.to_dict())
    gt_consolidated = {}
    for sid, rows in gt_by_shipment.items():
        exc_rows = [r for r in rows if r["is_exception"] == "YES"]
        gt_consolidated[sid] = exc_rows[-1] if exc_rows else rows[0]
    return gt_consolidated


@st.cache_resource
def get_playbook_retriever():
    """Load the playbook PDF, split it, embed it, and build a Chroma retriever.
    Cached as a resource since this is expensive (downloads/loads the embedding
    model and re-embeds the playbook) and only needs to happen once per
    Space instance."""
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(PLAYBOOK_PATH)
    playbook_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(playbook_docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


playbook_retriever = get_playbook_retriever()

all_logs = load_delivery_logs()
shipment_groups = defaultdict(list)
for row in all_logs:
    shipment_groups[row["shipment_id"]].append(row)
unique_shipment_ids = list(dict.fromkeys(row["shipment_id"] for row in all_logs))
gt_consolidated = load_ground_truth()


# ─── TOOLS ─────────────────────────────────────────────────────────────────────
@tool
def read_delivery_logs() -> list[dict]:
    """Read all delivery log rows from CSV. Used by preprocessor only."""
    with open(DELIVERY_LOGS_PATH, "r") as f:
        return list(csv.DictReader(f))


@tool
def lookup_customer_profile(customer_id: str, include_pii: bool = False) -> dict:
    """Fetch customer profile from SQLite. PII (name) only included when explicitly requested."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    row = cursor.fetchone()
    if row is None:
        return {}
    profile = dict(row)
    if not include_pii:
        profile.pop("name", None)
    return profile


@tool
def check_locker_availability(zip_code: str, package_size: str) -> list[dict]:
    """Find compatible lockers in the same zip code. Returns eligibility with reasoning."""
    size_hierarchy = {"SMALL": 1, "MEDIUM": 2, "LARGE": 3}
    pkg_level = size_hierarchy.get(package_size, 0)

    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM lockers WHERE zip_code = ?", (zip_code,))
    rows = cursor.fetchall()

    results = []
    for row in rows:
        locker = dict(row)
        locker_max = size_hierarchy.get(locker["max_package_size"], 0)

        if locker_max < pkg_level:
            locker["eligible"] = False
            locker["reason"] = f"Locker max {locker['max_package_size']} < package {package_size}"
        elif locker["capacity_status"] == "FULL":
            locker["eligible"] = False
            locker["reason"] = "Locker is FULL"
        elif locker["capacity_status"] == "LIMITED" and package_size != "SMALL":
            locker["eligible"] = False
            locker["reason"] = "Locker is LIMITED - only SMALL packages accepted"
        else:
            locker["eligible"] = True
            locker["reason"] = "Compatible"

        results.append(locker)

    return results


@tool
def search_playbook(query: str) -> list[dict]:
    """Retrieve relevant playbook sections via vector search. Returns chunks with page metadata."""
    docs = playbook_retriever.invoke(query)
    return [
        {"content": d.page_content, "page": d.metadata.get("page", "?")}
        for d in docs
    ]


@tool
def check_escalation_rules(customer_tier: str, exceptions_last_90d: int,
                            attempt_number: int, package_type: str,
                            status_code: str, status_description: str) -> dict:
    """Deterministic escalation rule engine. Evaluates hard-coded business rules."""
    triggers = []

    if attempt_number >= 3:
        triggers.append("AUTOMATIC: 3rd failed delivery attempt")

    if customer_tier == "VIP" and exceptions_last_90d >= 3:
        triggers.append(f"AUTOMATIC: VIP customer with {exceptions_last_90d} exceptions in 90d (>=3)")

    if status_code == "DAMAGED" and package_type == "PERISHABLE":
        triggers.append("AUTOMATIC: Damaged perishable package")

    if status_code == "WEATHER_DELAY" and package_type == "PERISHABLE":
        hour_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:hr|hour|hours)', status_description.lower())
        if hour_matches:
            hours = float(hour_matches[0])
            if hours > 4:
                triggers.append(f"AUTOMATIC: Perishable with {hours}hr delay (>4hr threshold)")

    fraud_keywords = ["vacant", "demolished", "construction site", "empty lot"]
    if status_code == "ADDRESS_ISSUE" and any(kw in status_description.lower() for kw in fraud_keywords):
        triggers.append("AUTOMATIC: Potential fraud - address is vacant/demolished")

    if customer_tier == "STANDARD" and exceptions_last_90d > 5:
        triggers.append(f"DISCRETIONARY: Standard customer with {exceptions_last_90d} exceptions in 90d (>5)")

    if customer_tier == "PREMIUM" and package_type == "PERISHABLE" and status_code == "WEATHER_DELAY":
        triggers.append("DISCRETIONARY: Premium customer with perishable in weather delay")

    return {
        "has_triggers": len(triggers) > 0,
        "trigger_count": len(triggers),
        "triggers": triggers
    }


# ─── STATE & AGENT VIEWS ───────────────────────────────────────────────────────
class UnifiedAgentState(TypedDict):
    """State object passed through the LangGraph pipeline."""
    raw_rows: list[dict]
    shipment_id: str
    consolidated_event: dict
    customer_profile: dict
    customer_profile_full: dict
    locker_availability: list[dict]
    playbook_context: list[dict]
    escalation_signals: dict
    noise_override: bool
    guardrail_triggered: bool
    resolution_output: dict
    critic_resolution_output: dict
    resolution_revision_count: int
    critic_feedback: str
    communication_output: dict
    critic_communication_output: dict
    next_agent: str
    max_loops: int
    escalated: bool
    tool_calls_log: list[str]
    trajectory_log: list[str]
    start_time: Optional[float]
    latency_sec: Optional[float]
    final_actions: list[dict]


@dataclass
class RouterView:
    """Fields accessible to the Router Agent (preprocessor, orchestrator, finalize)."""
    raw_rows: list[dict]
    shipment_id: str
    consolidated_event: dict
    customer_profile: dict
    customer_profile_full: dict
    locker_availability: list[dict]
    playbook_context: list[dict]
    escalation_signals: dict
    noise_override: bool
    guardrail_triggered: bool
    resolution_output: dict
    critic_resolution_output: dict
    resolution_revision_count: int
    critic_feedback: str
    communication_output: dict
    critic_communication_output: dict
    next_agent: str
    max_loops: int
    escalated: bool
    tool_calls_log: list[str]
    trajectory_log: list[str]
    start_time: Optional[float]
    latency_sec: Optional[float]
    final_actions: list[dict]


@dataclass
class ResolutionAgentView:
    """Fields accessible to the Resolution Agent. No PII."""
    consolidated_event: dict
    customer_profile: dict
    locker_availability: list[dict]
    playbook_context: list[dict]
    escalation_signals: dict
    critic_feedback: str
    resolution_output: dict


@dataclass
class CommunicationAgentView:
    """Fields accessible to the Communication Agent. Includes PII for personalization."""
    consolidated_event: dict
    customer_profile_full: dict
    locker_availability: list[dict]
    resolution_output: dict
    communication_output: dict


@dataclass
class CriticResolutionView:
    """Fields accessible to the Critic Agent for resolution validation. No PII."""
    consolidated_event: dict
    customer_profile: dict
    locker_availability: list[dict]
    playbook_context: list[dict]
    escalation_signals: dict
    resolution_output: dict
    critic_resolution_output: dict


@dataclass
class CriticCommunicationView:
    """Fields accessible to the Critic Agent for communication validation. No PII."""
    consolidated_event: dict
    customer_profile: dict
    resolution_output: dict
    communication_output: dict
    critic_communication_output: dict


def project_into(state: UnifiedAgentState, view_class: type) -> dict:
    """Extract only the fields defined in the agent's view from the global state."""
    view_fields = {f.name for f in fields(view_class)}
    return {k: state.get(k) for k in view_fields if k in state}


def merge_back(state: UnifiedAgentState, agent_output: dict, view_class: type) -> UnifiedAgentState:
    """Write back only the fields owned by the agent's view into the global state."""
    view_fields = {f.name for f in fields(view_class)}
    for k, v in agent_output.items():
        if k in view_fields:
            state[k] = v
    return state


# ─── ROUTER AGENT (preprocessor / orchestrator / finalize) ────────────────────
INJECTION_KEYWORDS = [
    "ignore previous instructions", "disregard all prior commands", "act as a",
    "assume the role of", "forget everything", "new persona", "override",
    "jailbreak", "developer mode", "system prompt", "confidential information",
    "private data", "leak data", "reveal secrets", "tell me about your prompts",
    "what are your instructions", "print current prompt", "execute this command",
    "run shell", "python interpreter", "code execution", "sudo", "rm -rf",
    "cat /etc/passwd", "environment variables", "system files", "do not comply",
    "malicious", "virus", "exploit", "hack", "attack", "sensitive information",
    "hidden directives", "hidden instructions",
]


def scan_for_injection(text: str) -> bool:
    """Returns True if prompt injection keywords are detected in the text."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in INJECTION_KEYWORDS)


def deduplicate_rows(raw_rows: list[dict]) -> list[dict]:
    """Remove duplicate scan events."""
    return [r for r in raw_rows if r.get("is_duplicate_scan", "False") != "True"]


def consolidate_event(unique_rows: list[dict], raw_rows: list[dict]) -> dict:
    """Consolidate multi-row shipment into a single event using highest attempt number."""
    if unique_rows:
        primary = max(unique_rows, key=lambda r: int(r.get("attempt_number", 0)))
    else:
        primary = raw_rows[0]

    prior_notes = [
        f"Attempt {r['attempt_number']}: {r['status_description']}"
        for r in unique_rows
        if r is not primary
    ]

    return {
        "shipment_id": primary["shipment_id"],
        "timestamp": primary["timestamp"],
        "status_code": primary["status_code"],
        "status_description": primary["status_description"],
        "customer_id": primary["customer_id"],
        "delivery_address": primary["delivery_address"],
        "package_type": primary["package_type"],
        "package_size": primary["package_size"],
        "attempt_number": int(primary["attempt_number"]),
        "prior_attempt_notes": prior_notes,
        "total_rows": len(raw_rows),
        "duplicates_removed": len(raw_rows) - len(unique_rows)
    }


def scan_inputs_for_injection(consolidated: dict, raw_rows: list[dict]) -> bool:
    """Scan all free-text fields in delivery data for prompt injection."""
    texts = [consolidated["status_description"]]
    texts.extend(row.get("status_description", "") for row in raw_rows)
    return any(scan_for_injection(text) for text in texts)


def scan_chunks_for_injection(playbook_context: list[dict]) -> bool:
    """Scan retrieved RAG chunks for prompt injection."""
    return any(scan_for_injection(chunk.get("content", "")) for chunk in playbook_context)


def fetch_context(consolidated: dict, tool_log: list[str]) -> dict:
    """Fetch all context via tools: customer profiles, lockers, playbook, escalation rules."""
    customer_id = consolidated["customer_id"]

    customer_profile = lookup_customer_profile.invoke(
        {"customer_id": customer_id, "include_pii": False}
    )
    tool_log.append(f"TOOL: lookup_customer_profile({customer_id}, pii=False)")

    customer_profile_full = lookup_customer_profile.invoke(
        {"customer_id": customer_id, "include_pii": True}
    )
    tool_log.append(f"TOOL: lookup_customer_profile({customer_id}, pii=True)")

    address_parts = consolidated["delivery_address"].split(",")
    zip_code = address_parts[-1].strip() if address_parts else ""
    locker_availability = check_locker_availability.invoke(
        {"zip_code": zip_code, "package_size": consolidated["package_size"]}
    )
    tool_log.append(f"TOOL: check_locker_availability({zip_code}, {consolidated['package_size']})")

    query = f"{consolidated['status_code']} {consolidated['package_type']} {consolidated['status_description'][:100]}"
    playbook_context = search_playbook.invoke({"query": query})
    tool_log.append("TOOL: search_playbook(query)")

    escalation_signals = check_escalation_rules.invoke({
        "customer_tier": customer_profile.get("tier", "STANDARD"),
        "exceptions_last_90d": customer_profile.get("exceptions_last_90d", 0),
        "attempt_number": consolidated["attempt_number"],
        "package_type": consolidated["package_type"],
        "status_code": consolidated["status_code"],
        "status_description": consolidated["status_description"]
    })
    tool_log.append("TOOL: check_escalation_rules(...)")

    return {
        "customer_profile": customer_profile,
        "customer_profile_full": customer_profile_full,
        "locker_availability": locker_availability,
        "playbook_context": playbook_context,
        "escalation_signals": escalation_signals
    }


def check_noise_override(consolidated: dict) -> bool:
    """Flag routine status codes with no anomaly indicators."""
    routine_codes = {"DELIVERED", "IN_TRANSIT", "OUT_FOR_DELIVERY", "SCANNED"}
    if consolidated["status_code"] not in routine_codes:
        return False

    anomaly_indicators = [
        "damage", "wrong", "suspicious", "overdue", "missing",
        "unexpected", "misroute", "lost", "stolen", "abandoned",
        "leak", "crush", "broke", "delay", "late", "fraud"
    ]
    desc = consolidated["status_description"].lower()
    return not any(indicator in desc for indicator in anomaly_indicators)


@traceable(name="preprocessor_node")
def preprocessor_node(state: UnifiedAgentState) -> UnifiedAgentState:
    """Deduplicates, consolidates, assembles context, and runs input guardrails."""
    view = project_into(state, RouterView)
    tool_log = []
    trajectory = []
    start = time.time()

    unique_rows = deduplicate_rows(view["raw_rows"])
    tool_log.append("PREPROCESSOR: Deduplicated rows")
    trajectory.append(f"preprocessor: {len(view['raw_rows'])} raw rows -> {len(unique_rows)} after dedup")

    consolidated = consolidate_event(unique_rows, view["raw_rows"])

    if scan_inputs_for_injection(consolidated, view["raw_rows"]):
        tool_log.append("GUARDRAIL: Injection detected in delivery input")
        trajectory.append("preprocessor: Guardrail triggered - prompt injection detected")
        output = {
            "consolidated_event": consolidated,
            "customer_profile": {}, "customer_profile_full": {},
            "locker_availability": [], "playbook_context": [], "escalation_signals": {},
            "tool_calls_log": tool_log, "trajectory_log": trajectory,
            "resolution_revision_count": 0, "critic_feedback": "",
            "noise_override": False, "guardrail_triggered": True,
            "escalated": True, "start_time": start, "next_agent": "finalize"
        }
        return merge_back(state, output, RouterView)

    noise_override = check_noise_override(consolidated)
    if noise_override:
        tool_log.append("PREPROCESSOR: Noise guardrail - routine status with no anomaly")
        trajectory.append(f"preprocessor: {consolidated['status_code']} flagged as noise by guardrail, skipping tool calls")
        output = {
            "consolidated_event": consolidated,
            "customer_profile": {}, "customer_profile_full": {},
            "locker_availability": [], "playbook_context": [],
            "escalation_signals": {},
            "tool_calls_log": tool_log, "trajectory_log": trajectory,
            "resolution_revision_count": 0, "critic_feedback": "",
            "noise_override": True, "guardrail_triggered": False,
            "escalated": False, "start_time": start, "next_agent": "orchestrator"
        }
        return merge_back(state, output, RouterView)

    context = fetch_context(consolidated, tool_log)

    if scan_chunks_for_injection(context["playbook_context"]):
        tool_log.append("GUARDRAIL: Injection detected in retrieved playbook chunk")
        trajectory.append("preprocessor: Guardrail triggered - injection in RAG chunk")
        context["playbook_context"] = []
        output = {
            "consolidated_event": consolidated,
            **context,
            "tool_calls_log": tool_log, "trajectory_log": trajectory,
            "resolution_revision_count": 0, "critic_feedback": "",
            "noise_override": False, "guardrail_triggered": True,
            "escalated": True, "start_time": start, "next_agent": "finalize"
        }
        return merge_back(state, output, RouterView)

    output = {
        "consolidated_event": consolidated,
        **context,
        "tool_calls_log": tool_log, "trajectory_log": trajectory,
        "resolution_revision_count": 0, "critic_feedback": "",
        "noise_override": noise_override, "guardrail_triggered": False,
        "escalated": False, "start_time": start, "next_agent": "orchestrator"
    }
    return merge_back(state, output, RouterView)


@traceable(name="orchestrator_node")
def orchestrator_node(state: UnifiedAgentState) -> UnifiedAgentState:
    """
    Central router. Determines next_agent based on current state.

    Routing order:
      0. Guardrail triggered                          -> finalize
      1. Noise override from preprocessor             -> finalize (skip LLM)
      2. Resolution not yet run                       -> resolution_agent
      3. Resolution done, critic not yet run          -> critic_resolution
      4. Critic returned REVISE and under loop limit  -> resolution_agent (reset)
      5. Critic returned REVISE and at loop limit     -> force ESCALATE -> communication/finalize
      6. Enforce automatic escalation triggers         (rule engine is authoritative)
      7. Not an exception                             -> finalize
      8. Communication not yet run                    -> communication_agent
      9. Communication done, critic not yet run       -> critic_communication
      10. All done                                    -> finalize
    """
    view = project_into(state, RouterView)

    if view.get("guardrail_triggered"):
        state["resolution_output"] = {
            "is_exception": "YES",
            "resolution": "RESCHEDULE",
            "rationale": "Input flagged by guardrail - prompt injection detected. Defaulting to RESCHEDULE with forced escalation for human review."
        }
        state["escalated"] = True
        state["next_agent"] = "finalize"
        state["trajectory_log"].append("orchestrator: Guardrail triggered, forcing escalation to finalize")
        return state

    if view.get("noise_override") and not view.get("resolution_output"):
        state["resolution_output"] = {
            "is_exception": "NO",
            "resolution": "N/A",
            "rationale": f"Status code {view['consolidated_event']['status_code']} with routine description. No anomaly indicators. Classified as noise by preprocessor guardrail."
        }
        state["trajectory_log"].append("orchestrator: Noise override from preprocessor, skipping to finalize")
        state["next_agent"] = "finalize"
        return state

    if not view.get("resolution_output"):
        state["next_agent"] = "resolution_agent"
        return state

    if not view.get("critic_resolution_output"):
        state["next_agent"] = "critic_resolution"
        return state

    critic_decision = view["critic_resolution_output"].get("decision")

    if critic_decision == "REVISE" and view["resolution_revision_count"] < view["max_loops"]:
        state["resolution_output"] = {}
        state["critic_resolution_output"] = {}
        state["next_agent"] = "resolution_agent"
        state["trajectory_log"].append(
            f"orchestrator: REVISE loop {view['resolution_revision_count']}/{view['max_loops']}"
        )
        return state

    if critic_decision == "REVISE" and view["resolution_revision_count"] >= view["max_loops"]:
        state["escalated"] = True
        state["critic_resolution_output"] = {
            "decision": "ESCALATE",
            "rationale": "Max revision loops reached. Accepting current resolution with escalation."
        }
        if view["resolution_output"].get("is_exception") == "YES":
            state["next_agent"] = "communication_agent"
        else:
            state["next_agent"] = "finalize"
        state["trajectory_log"].append("orchestrator: Max loops reached, forcing ESCALATE")
        return state

    if view.get("escalation_signals", {}).get("has_triggers"):
        automatic = [t for t in view["escalation_signals"].get("triggers", [])
                     if t.startswith("AUTOMATIC")]
        if automatic and view["resolution_output"].get("is_exception") == "YES":
            state["escalated"] = True
            state["trajectory_log"].append(
                f"orchestrator: Forced escalation from rule engine - {automatic}"
            )

    if view["resolution_output"].get("is_exception") == "NO":
        state["next_agent"] = "finalize"
        state["trajectory_log"].append("orchestrator: Not an exception, skipping to finalize")
        return state

    if not view.get("communication_output"):
        state["next_agent"] = "communication_agent"
        return state

    if not view.get("critic_communication_output"):
        state["next_agent"] = "critic_communication"
        return state

    state["next_agent"] = "finalize"
    return state


@traceable(name="finalize_node")
def finalize_node(state: UnifiedAgentState) -> UnifiedAgentState:
    """Packages final results and records end time."""
    view = project_into(state, RouterView)

    if view.get("guardrail_triggered"):
        final = {
            "shipment_id": view["shipment_id"],
            "is_exception": "BLOCKED",
            "resolution": "ESCALATED",
            "escalated": True,
            "tone": "N/A",
            "message": "This shipment was flagged by the input guardrail and requires human review.",
            "revision_count": 0,
            "guardrail_blocked": True
        }
    else:
        final = {
            "shipment_id": view["shipment_id"],
            "is_exception": view.get("resolution_output", {}).get("is_exception", "ERROR"),
            "resolution": view.get("resolution_output", {}).get("resolution", "ERROR"),
            "escalated": view["escalated"],
            "tone": view.get("communication_output", {}).get("tone_label", "N/A"),
            "message": view.get("communication_output", {}).get("communication_message", ""),
            "revision_count": view["resolution_revision_count"],
            "guardrail_blocked": False
        }

    latency = time.time() - view["start_time"] if view.get("start_time") else 0.0

    output = {
        "final_actions": [final],
        "latency_sec": latency,
        "next_agent": "END",
    }

    state["trajectory_log"].append(
        f"finalize: actions={json.dumps(final)}; latency={latency:.3f}s"
    )

    return merge_back(state, output, RouterView)


# ─── RESOLUTION AGENT ──────────────────────────────────────────────────────────
class ResolutionOutput(BaseModel):
    """Resolution Agent output schema."""
    is_exception: Literal["YES", "NO"] = Field(
        description="Whether this delivery event is a real actionable exception"
    )
    resolution: Literal[
        "RESCHEDULE", "REROUTE_TO_LOCKER", "REPLACE", "RETURN_TO_SENDER", "N/A"
    ] = Field(
        description="Resolution action. N/A if is_exception is NO"
    )
    rationale: str = Field(
        description="Step-by-step reasoning for the classification and resolution decision"
    )

    @model_validator(mode="after")
    def validate_consistency(self):
        """Enforce that is_exception and resolution are mutually consistent."""
        if self.is_exception == "YES" and self.resolution == "N/A":
            raise ValueError("resolution cannot be N/A when is_exception is YES")
        if self.is_exception == "NO" and self.resolution != "N/A":
            raise ValueError("resolution must be N/A when is_exception is NO")
        return self


RESOLUTION_AGENT_SYSTEM_PROMPT = """You are an AI assistant designed to analyze delivery events and determine if they constitute an actionable exception. If an exception is identified, you must propose a resolution action and provide a clear rationale for your decision. Your primary goal is to ensure efficient and accurate exception handling, adhering to business rules and customer context.You are provided with:1. DELIVERY EVENT: Details of the shipment event, including status, descriptions, and package information.2. CUSTOMER PROFILE: Redacted customer information (no PII), including tier, and exception history.3. LOCKER AVAILABILITY: Information about available lockers in the delivery area and their eligibility.4. ESCALATION SIGNALS: Output from a deterministic rule engine indicating potential reasons for escalation.5. RELEVANT PLAYBOOK SECTIONS: Specific sections from the exception resolution playbook that are relevant to the current delivery event.Follow these rules:
- Carefully analyze the 'DELIVERY EVENT', 'CUSTOMER PROFILE', 'LOCKER AVAILABILITY', 'ESCALATION SIGNALS', and 'RELEVANT PLAYBOOK SECTIONS'.
- Determine if the event is an exception (YES/NO).
- If it is an exception, choose the most appropriate resolution from: 'RESCHEDULE', 'REROUTE_TO_LOCKER', 'REPLACE', or 'RETURN_TO_SENDER'.
- If it is NOT an exception, the resolution MUST be 'N/A'.
- Always provide a step-by-step rationale for your classification and resolution decision.
- Your output MUST strictly adhere to the `ResolutionOutput` Pydantic schema.

{critic_feedback}"""


def format_playbook_context(playbook: list[dict]) -> str:
    """Format playbook chunks with page references for LLM context."""
    return "\n\n---\n\n".join(
        [f"[Page {c['page']}] {c['content']}" for c in playbook]
    )


@traceable(name="resolution_agent_node")
def resolution_agent_node(state: UnifiedAgentState) -> UnifiedAgentState:
    """Resolution Agent: classifies exception and decides resolution action."""
    view = project_into(state, ResolutionAgentView)

    feedback = view.get("critic_feedback", "")
    feedback_section = ""
    if feedback:
        feedback_section = (
            f"\n\nPREVIOUS ATTEMPT WAS REJECTED. Critic feedback:\n{feedback}\n"
            f"Revise your decision based on this feedback."
        )

    system_prompt = RESOLUTION_AGENT_SYSTEM_PROMPT.format(critic_feedback=feedback_section)
    playbook_text = format_playbook_context(view["playbook_context"])

    user_content = (
        f"DELIVERY EVENT:\n{json.dumps(view['consolidated_event'], indent=2)}\n\n"
        f"CUSTOMER PROFILE (redacted):\n{json.dumps(view['customer_profile'], indent=2)}\n\n"
        f"LOCKER AVAILABILITY:\n{json.dumps(view['locker_availability'], indent=2)}\n\n"
        f"ESCALATION SIGNALS:\n{json.dumps(view['escalation_signals'], indent=2)}\n\n"
        f"RELEVANT PLAYBOOK SECTIONS:\n{playbook_text}"
    )

    structured_llm = gen_llm.with_structured_output(ResolutionOutput)
    max_retries = 3
    result = None

    for attempt in range(max_retries):
        try:
            result = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ])
            break
        except Exception as e:
            if attempt < max_retries - 1:
                state["trajectory_log"].append(
                    f"resolution_agent: Validation failed (attempt {attempt + 1}), retrying - {str(e)[:100]}"
                )
            else:
                result = ResolutionOutput(
                    is_exception="YES",
                    resolution="RESCHEDULE",
                    rationale=(
                        f"Resolution agent failed after {max_retries} attempts. "
                        f"Defaulting to RESCHEDULE with escalation. Last error: {str(e)[:200]}"
                    )
                )
                state["escalated"] = True
                state["trajectory_log"].append(
                    f"resolution_agent: All {max_retries} retries exhausted, "
                    f"defaulting to RESCHEDULE with forced escalation"
                )

    agent_output = {"resolution_output": result.model_dump()}
    state = merge_back(state, agent_output, ResolutionAgentView)

    state["tool_calls_log"].append("AGENT: resolution_agent invoked")
    state["trajectory_log"].append(
        f"resolution_agent: is_exception={result.is_exception}, resolution={result.resolution}"
    )
    state["next_agent"] = "orchestrator"
    return state


# ─── COMMUNICATION AGENT ────────────────────────────────────────────────────────
class CommunicationOutput(BaseModel):
    """Communication Agent output schema."""
    tone_label: Literal["FORMAL", "CASUAL"] = Field(
        description="Tone of the customer message, inferred from customer tier"
    )
    communication_message: str = Field(
        description="The customer-facing notification message"
    )


COMMUNICATION_AGENT_SYSTEM_PROMPT = """You are an AI Communication Agent for a last-mile delivery service.
Your role is to draft concise, empathetic, and personalized customer notifications regarding delivery exceptions.

Rules & Guidelines:
1. Tone: Strictly adhere to the requested tone (FORMAL for VIP/PREMIUM customers, CASUAL for STANDARD customers).
2. Clarity: Clearly state the delivery issue and the proposed resolution (e.g., reschedule, locker reroute, replacement).
3. Personalization: Address the customer by their name and mention the specific package type if relevant.
4. Brevity: Keep the message concise (maximum 3-4 sentences) and avoid blaming the driver or the company.
5. Next Steps: If the package was rerouted to a locker, include the locker location and pickup instructions. If rescheduled, provide the expected outcome.

Use the provided CONTEXT to draft the message."""


def build_communication_context(view: dict) -> tuple[dict, str]:
    """Build the context dict and locker info string for the Communication Agent.
    Only uses fields available in CommunicationAgentView."""
    event = view["consolidated_event"]
    profile = view["customer_profile_full"]
    resolution = view["resolution_output"]
    lockers = view["locker_availability"]

    locker_info = ""
    if resolution.get("resolution") == "REROUTE_TO_LOCKER":
        eligible = [l for l in lockers if l.get("eligible")]
        if eligible:
            locker_info = f"\nLOCKER FOR REROUTE:\n{json.dumps(eligible[0], indent=2)}"

    comm_context = {
        "customer_name": profile.get("name", "Customer"),
        "customer_tier": profile.get("tier"),
        "preferred_channel": profile.get("preferred_channel"),
        "active_credit": profile.get("active_credit", 0),
        "exception_type": event["status_code"],
        "status_description": event["status_description"],
        "package_type": event["package_type"],
        "resolution": resolution.get("resolution"),
        "resolution_rationale": resolution.get("rationale")
    }

    return comm_context, locker_info


@traceable(name="communication_agent_node")
def communication_agent_node(state: UnifiedAgentState) -> UnifiedAgentState:
    """Communication Agent: generates customer notification. Only agent with PII access."""
    view = project_into(state, CommunicationAgentView)
    comm_context, locker_info = build_communication_context(view)
    user_content = f"CONTEXT:\n{json.dumps(comm_context, indent=2)}\n{locker_info}"

    structured_llm = gen_llm.with_structured_output(CommunicationOutput)
    max_retries = 3
    result = None

    for attempt in range(max_retries):
        try:
            result = structured_llm.invoke([
                SystemMessage(content=COMMUNICATION_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=user_content)
            ])
            break
        except Exception as e:
            if attempt < max_retries - 1:
                state["trajectory_log"].append(
                    f"communication_agent: Validation failed (attempt {attempt + 1}), retrying - {str(e)[:100]}"
                )
            else:
                tier = view["customer_profile_full"].get("tier", "STANDARD")
                result = CommunicationOutput(
                    tone_label="FORMAL" if tier in ("VIP", "PREMIUM") else "CASUAL",
                    communication_message="We're aware of an issue with your delivery and are working to resolve it. A team member will follow up shortly."
                )
                state["escalated"] = True
                state["trajectory_log"].append(
                    f"communication_agent: All {max_retries} retries exhausted, "
                    f"defaulting to generic message with forced escalation"
                )

    agent_output = {"communication_output": result.model_dump()}
    state = merge_back(state, agent_output, CommunicationAgentView)

    state["tool_calls_log"].append("AGENT: communication_agent invoked")
    state["trajectory_log"].append(f"communication_agent: tone={result.tone_label}")
    state["next_agent"] = "orchestrator"
    return state


# ─── CRITIC AGENT ──────────────────────────────────────────────────────────────
class CriticResolutionOutput(BaseModel):
    """Critic Agent - resolution validation output."""
    decision: Literal["ACCEPT", "ESCALATE", "REVISE"] = Field(
        description="ACCEPT: valid. ESCALATE: needs supervisor. REVISE: send back to Resolution Agent."
    )
    rationale: str = Field(description="Reasoning for the validation decision")


class CriticCommunicationOutput(BaseModel):
    """Critic Agent - communication validation output."""
    decision: Literal["ACCEPT", "ESCALATE"] = Field(
        description="ACCEPT: message is appropriate. ESCALATE: needs supervisor review."
    )
    rationale: str = Field(description="Reasoning for the validation decision")


CRITIC_RESOLUTION_SYSTEM_PROMPT = """You are a Critic Agent responsible for validating the exception resolution decisions made by the Resolution Agent.
Your goal is to ensure accuracy, policy adherence, and proper escalation handling based on the operations playbook.

Role & Context:
You will be provided with the DELIVERY EVENT, CUSTOMER PROFILE, LOCKER AVAILABILITY, ESCALATION SIGNALS, PLAYBOOK CONTEXT, and the RESOLUTION AGENT OUTPUT.

Rules & Guidelines:
1. Evaluate the Resolution Agent's decision (`is_exception` and `resolution`) against the provided PLAYBOOK CONTEXT and ESCALATION SIGNALS.
2. Decision 'ACCEPT': Choose this if the Resolution Agent's output is completely correct, adheres to all playbook rules, and respects locker constraints (if rerouted).
3. Decision 'ESCALATE': Choose this if the ESCALATION SIGNALS indicate a mandatory escalation (e.g., 3rd attempt, damaged perishables) OR if the playbook explicitly requires human supervisor review.
4. Decision 'REVISE': Choose this if the Resolution Agent made an error that can be fixed (e.g., chose 'RESCHEDULE' instead of 'REROUTE_TO_LOCKER' when eligible lockers exist, or missed a playbook constraint).
5. Always provide a clear, step-by-step `rationale` justifying your decision. If you choose 'REVISE', explicitly state what the Resolution Agent needs to fix.
"""

CRITIC_COMMUNICATION_SYSTEM_PROMPT = """You are a Critic Agent responsible for validating the customer communication messages generated by the Communication Agent.
Your goal is to ensure the message is accurate, empathetic, and uses the correct tone based on the customer's tier.

Role & Context:
You will be provided with the VALIDATION CONTEXT (including customer tier, preferred channel, resolution, and exception type) and the COMMUNICATION AGENT OUTPUT.

Rules & Guidelines:
1. Evaluate the message for Tone: It must be FORMAL for VIP or PREMIUM customers, and CASUAL for STANDARD customers.
2. Evaluate for Clarity & Empathy: The message should clearly explain the resolution and be empathetic without blaming the driver or company.
3. Decision 'ACCEPT': Choose this if the message is completely appropriate, matches the required tone, and accurately reflects the resolution.
4. Decision 'ESCALATE': Choose this if the message is inappropriate, rude, confusing, or fails to use the correct tone. Messages are not revised; problematic messages are escalated for supervisor review.
5. Always provide a clear `rationale` for your decision.
"""


def build_critic_resolution_context(view: dict) -> str:
    """Build the user content string for resolution validation from CriticResolutionView fields."""
    playbook_text = format_playbook_context(view["playbook_context"])
    return (
        f"DELIVERY EVENT:\n{json.dumps(view['consolidated_event'], indent=2)}\n\n"
        f"CUSTOMER PROFILE:\n{json.dumps(view['customer_profile'], indent=2)}\n\n"
        f"LOCKER AVAILABILITY:\n{json.dumps(view['locker_availability'], indent=2)}\n\n"
        f"ESCALATION SIGNALS:\n{json.dumps(view['escalation_signals'], indent=2)}\n\n"
        f"PLAYBOOK CONTEXT:\n{playbook_text}\n\n"
        f"RESOLUTION AGENT OUTPUT:\n{json.dumps(view['resolution_output'], indent=2)}"
    )


def build_critic_communication_context(view: dict) -> str:
    """Build the user content string for communication validation from CriticCommunicationView fields."""
    validation_context = {
        "customer_tier": view["customer_profile"].get("tier"),
        "preferred_channel": view["customer_profile"].get("preferred_channel"),
        "active_credit": view["customer_profile"].get("active_credit", 0),
        "resolution": view["resolution_output"].get("resolution"),
        "exception_type": view["consolidated_event"]["status_code"],
        "package_type": view["consolidated_event"]["package_type"]
    }
    return (
        f"VALIDATION CONTEXT:\n{json.dumps(validation_context, indent=2)}\n\n"
        f"COMMUNICATION AGENT OUTPUT:\n{json.dumps(view['communication_output'], indent=2)}"
    )


@traceable(name="critic_resolution_node")
def critic_resolution_node(state: UnifiedAgentState) -> UnifiedAgentState:
    """Critic Agent: validates resolution decision against playbook and context."""
    view = project_into(state, CriticResolutionView)
    user_content = build_critic_resolution_context(view)

    structured_llm = eval_llm.with_structured_output(CriticResolutionOutput)
    result = structured_llm.invoke([
        SystemMessage(content=CRITIC_RESOLUTION_SYSTEM_PROMPT),
        HumanMessage(content=user_content)
    ])

    agent_output = {"critic_resolution_output": result.model_dump()}
    state = merge_back(state, agent_output, CriticResolutionView)

    if result.decision == "ESCALATE":
        state["escalated"] = True

    if result.decision == "REVISE":
        state["resolution_revision_count"] += 1
        state["critic_feedback"] = result.rationale

    state["tool_calls_log"].append("AGENT: critic_resolution invoked")
    state["trajectory_log"].append(f"critic_resolution: decision={result.decision}")
    state["next_agent"] = "orchestrator"
    return state


@traceable(name="critic_communication_node")
def critic_communication_node(state: UnifiedAgentState) -> UnifiedAgentState:
    """Critic Agent: validates customer communication quality. No PII access."""
    view = project_into(state, CriticCommunicationView)
    user_content = build_critic_communication_context(view)

    structured_llm = eval_llm.with_structured_output(CriticCommunicationOutput)
    result = structured_llm.invoke([
        SystemMessage(content=CRITIC_COMMUNICATION_SYSTEM_PROMPT),
        HumanMessage(content=user_content)
    ])

    agent_output = {"critic_communication_output": result.model_dump()}
    state = merge_back(state, agent_output, CriticCommunicationView)

    if result.decision == "ESCALATE":
        state["escalated"] = True

    state["tool_calls_log"].append("AGENT: critic_communication invoked")
    state["trajectory_log"].append(f"critic_communication: decision={result.decision}")
    state["next_agent"] = "orchestrator"
    return state


# ─── WORKFLOW ──────────────────────────────────────────────────────────────────
@st.cache_resource
def build_workflow():
    workflow = StateGraph(UnifiedAgentState)

    workflow.add_node("preprocessor_node", preprocessor_node)
    workflow.add_node("orchestrator_node", orchestrator_node)
    workflow.add_node("resolution_agent_node", resolution_agent_node)
    workflow.add_node("critic_resolution_node", critic_resolution_node)
    workflow.add_node("communication_agent_node", communication_agent_node)
    workflow.add_node("critic_communication_node", critic_communication_node)
    workflow.add_node("finalize_node", finalize_node)

    workflow.set_entry_point("preprocessor_node")
    workflow.add_edge("preprocessor_node", "orchestrator_node")

    workflow.add_conditional_edges(
        "orchestrator_node",
        lambda state: state["next_agent"],
        {
            "resolution_agent": "resolution_agent_node",
            "critic_resolution": "critic_resolution_node",
            "communication_agent": "communication_agent_node",
            "critic_communication": "critic_communication_node",
            "finalize": "finalize_node",
            "END": END
        }
    )

    workflow.add_edge("resolution_agent_node", "orchestrator_node")
    workflow.add_edge("critic_resolution_node", "orchestrator_node")
    workflow.add_edge("communication_agent_node", "orchestrator_node")
    workflow.add_edge("critic_communication_node", "orchestrator_node")
    workflow.add_edge("finalize_node", END)

    return workflow.compile()


last_mile_app = build_workflow()


# ─── EVALUATION METRICS ────────────────────────────────────────────────────────
def compute_task_completion(gt: dict, pred: dict) -> dict:
    """Compute task completion for a single shipment."""
    res = pred.get("resolution_output", {})
    comm = pred.get("communication_output", {})

    exception_correct = gt.get("is_exception") == res.get("is_exception")

    if gt.get("is_exception") == "YES":
        resolution_correct = gt.get("expected_resolution") == res.get("resolution")
        tone_correct = gt.get("expected_tone") == comm.get("tone_label", "N/A")
    else:
        resolution_correct = res.get("resolution") in ("N/A", None, "")
        tone_correct = True

    return {
        "exception_correct": exception_correct,
        "resolution_correct": resolution_correct,
        "tone_correct": tone_correct,
        "task_complete": exception_correct and resolution_correct and tone_correct
    }


def compute_escalation_accuracy(gt: dict, pred: dict):
    """Check if escalation decision matches ground truth for a single shipment."""
    if gt.get("should_escalate") not in ("YES", "NO"):
        return None
    pred_escalated = "YES" if pred.get("escalated", False) else "NO"
    return gt.get("should_escalate") == pred_escalated


# ─── DEEPEVAL CI TRACE ────────────────────────────────────────────────────────
# Streamlit clicks leave DEEPEVAL_TRACE unset. CI sets it so CallbackHandler
# records the LangGraph trajectory. deepeval is not in Space requirements.txt.


def _deepeval_invoke_config():
    flag = os.environ.get("DEEPEVAL_TRACE", "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return None
    try:
        from deepeval.integrations.langchain import CallbackHandler
    except ImportError:
        return None
    return {
        "callbacks": [
            CallbackHandler(
                name="last-mile",
                tags=["ci", "trajectory"],
                metadata={"graph": "last_mile_app"},
            )
        ]
    }


def run_pipeline(shipment_id: str) -> dict:
    """Execute the live pipeline for a single shipment scenario."""
    rows = shipment_groups[shipment_id]
    gt = gt_consolidated[shipment_id]

    initial_state = {
        "raw_rows": rows,
        "shipment_id": shipment_id,
        "consolidated_event": {},
        "customer_profile": {},
        "customer_profile_full": {},
        "locker_availability": [],
        "playbook_context": [],
        "escalation_signals": {},
        "resolution_output": {},
        "critic_resolution_output": {},
        "resolution_revision_count": 0,
        "critic_feedback": "",
        "communication_output": {},
        "critic_communication_output": {},
        "next_agent": "resolution_agent",
        "max_loops": 2,
        "escalated": False,
        "tool_calls_log": [],
        "trajectory_log": [],
        "start_time": None,
        "latency_sec": None,
        "final_actions": [],
        "noise_override": False,
        "guardrail_triggered": False
    }

    config = _deepeval_invoke_config()
    result = (
        last_mile_app.invoke(initial_state, config=config)
        if config
        else last_mile_app.invoke(initial_state)
    )
    task = compute_task_completion(gt, result)
    esc_acc = compute_escalation_accuracy(gt, result)
    citations = sorted({f"Page {c['page']}" for c in result.get("playbook_context", [])})

    return {"state": result, "gt": gt, "task_completion": task,
            "escalation_correct": esc_acc, "citations": citations}


# ─── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🚚 Last-Mile Delivery Exception Agents</h1>
  <p>A live LangGraph multi-agent pipeline for triaging and resolving last-mile delivery exceptions
    · <a class="eval-link" href="https://huggingface.co/spaces/alexoviedo999/last-mile-deepeval-results" target="_blank" rel="noopener noreferrer">CI DeepEval results</a></p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.write(
        "This is a **live** run of the actual multi-agent pipeline: a deterministic "
        "preprocessor and rule engine, a Resolution Agent, a Communication Agent, and "
        "two Critic agents, all orchestrated with LangGraph and backed by an open-"
        "weight model served via Hugging Face Inference Providers, making real-time "
        "decisions."
    )
    st.write(
        "Pick one of the 10 curated shipment scenarios below and run the pipeline to "
        "see the full decision trail: tool calls, agent reasoning, escalation triggers, "
        "the resolution decision, and the generated customer message."
    )
    st.markdown(
        "[CI DeepEval results](https://huggingface.co/spaces/alexoviedo999/last-mile-deepeval-results) "
        "— gold equality and trajectory scores. That Space does not run this graph."
    )
    st.divider()
    st.caption(
        "All customer records, delivery logs, and the operations playbook are "
        "synthetic data generated for this project."
    )
    st.caption("Built as part of a postgraduate program in AI Agents for Business "
               "Applications at UT Austin's McCombs School of Business.")

SCENARIO_LABELS = {
    "SHP-001": "SHP-001 — Successful delivery (noise filtering)",
    "SHP-002": "SHP-002 — VIP, multi-attempt with duplicate scan",
    "SHP-003": "SHP-003 — Address not found (standard customer)",
    "SHP-004": "SHP-004 — Damaged fragile package (VIP escalation)",
    "SHP-005": "SHP-005 — Third failed attempt (mandatory escalation)",
    "SHP-006": "SHP-006 — Refused delivery (return to sender)",
    "SHP-007": "SHP-007 — Perishable weather delay (VIP replace)",
    "SHP-008": "SHP-008 — Discretionary escalation (high exception history)",
    "SHP-009": "SHP-009 — Severely damaged perishable (premium replace)",
    "SHP-010": "SHP-010 — Routine depot scan (noise filtering)",
}

selected = st.selectbox(
    "Choose a shipment scenario",
    options=unique_shipment_ids,
    format_func=lambda sid: SCENARIO_LABELS.get(sid, sid),
)

with st.expander("View raw delivery log rows for this shipment"):
    st.dataframe(pd.DataFrame(shipment_groups[selected]), use_container_width=True)

run_clicked = st.button("▶ Run Pipeline", type="primary")

if not os.environ.get("HF_TOKEN"):
    st.warning(
        "No HF_TOKEN found in this Space's environment. Add a Hugging Face User "
        "Access Token under **Settings → Variables and secrets** to run the live "
        "pipeline.",
        icon="⚠️",
    )

if run_clicked:
    if not os.environ.get("HF_TOKEN"):
        st.error("Cannot run the pipeline without an HF_TOKEN secret configured on this Space.")
    else:
        with st.spinner(f"Running the live multi-agent pipeline for {selected}..."):
            try:
                result = run_pipeline(selected)
            except Exception as e:
                st.error(f"Pipeline run failed: {e}")
                result = None

        if result:
            state = result["state"]
            gt = result["gt"]
            res = state.get("resolution_output", {})
            comm = state.get("communication_output", {})
            tc = result["task_completion"]

            st.subheader("Decision")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Exception?", res.get("is_exception", "N/A"))
            c2.metric("Resolution", res.get("resolution", "N/A"))
            c3.metric("Tone", comm.get("tone_label", "N/A"))
            escalated_badge = "🚨 Escalated" if state.get("escalated") else "✅ Clear"
            c4.metric("Escalation", escalated_badge)

            if state.get("guardrail_triggered"):
                st.error("⚠️ Input guardrail triggered — this shipment's data was flagged as a "
                         "potential prompt injection and was blocked before reaching any LLM.")

            if comm.get("communication_message"):
                st.subheader("Generated Customer Message")
                st.markdown(
                    f'<div class="message-box">{comm["communication_message"]}</div>',
                    unsafe_allow_html=True,
                )

            st.subheader("Agent Trajectory")
            for entry in state.get("trajectory_log", []):
                st.markdown(f'<div class="trace-box">{entry}</div>', unsafe_allow_html=True)

            with st.expander("Ground truth comparison"):
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    st.markdown("**System Prediction**")
                    st.json({
                        "is_exception": res.get("is_exception"),
                        "resolution": res.get("resolution"),
                        "escalated": state.get("escalated"),
                        "tone": comm.get("tone_label"),
                    })
                with gcol2:
                    st.markdown("**Ground Truth**")
                    st.json({
                        "is_exception": gt.get("is_exception"),
                        "expected_resolution": gt.get("expected_resolution"),
                        "should_escalate": gt.get("should_escalate"),
                        "expected_tone": gt.get("expected_tone"),
                    })
                st.write(f"Ground truth reasoning: _{gt.get('ground_truth_reasoning', '')}_")
                esc_str = "N/A" if result["escalation_correct"] is None else (
                    "PASS" if result["escalation_correct"] else "FAIL")
                st.write(
                    f"**Task complete:** {'PASS' if tc['task_complete'] else 'FAIL'}  |  "
                    f"**Escalation accuracy:** {esc_str}"
                )

            with st.expander("Tool calls log"):
                for entry in state.get("tool_calls_log", []):
                    st.text(entry)

            if result["citations"]:
                st.caption(f"Playbook pages referenced: {', '.join(result['citations'])}")

            st.caption(f"Latency: {state.get('latency_sec', 0):.2f}s  |  "
                       f"Revisions: {state.get('resolution_revision_count', 0)}")

st.divider()
st.markdown("""
<div style="text-align:center;color:#888;font-size:.8em;padding:10px">
Last-Mile Delivery Exception Agents | LangGraph | Streamlit | RAG over an operations playbook
<br>⚠️ Demonstration system with synthetic data only.
</div>
""", unsafe_allow_html=True)
