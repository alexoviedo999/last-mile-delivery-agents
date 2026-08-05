# Last-Mile Delivery Exception Agents

An AI-powered multi-agent system that automates the triage and resolution of last-mile delivery exceptions (failed attempts, address issues, damaged packages, weather delays) for a mid-sized retailer's logistics operation.

## Business Context

Roughly 10% of last-mile deliveries encounter exceptions, and most retailers still handle them manually: reading driver notes, cross-referencing customer profiles, deciding on a resolution, and drafting a customer notification, all under time pressure. This project builds a proof-of-concept multi-agent pipeline that automates that workflow end-to-end while preserving human oversight where policy requires it.

## What it does

Given a raw delivery status log, including noisy, duplicated, multi-row shipment events, the system:

- deduplicates and consolidates events into a single shipment record
- classifies whether the event is an actionable exception or routine noise
- decides the appropriate resolution (reschedule, reroute to locker, replace, or return to sender) by reasoning over an operational playbook, customer tier, and package constraints
- escalates to a human supervisor when policy requires it, using a deterministic rule engine that cannot be overridden by LLM judgment
- generates a personalized customer notification calibrated to tone, channel, and customer tier
- produces an auditable decision trail with step-by-step rationale and evaluation metrics across five dimensions: task completion, escalation accuracy, tool call accuracy, reasoning coherence, and latency

  ## Architecture

  Built as a LangGraph state machine with a Router/Preprocessor agent, a Resolution Agent, a Communication Agent, and Critic agents that validate each output before it is finalized, with a capped revision loop. A deterministic escalation rule engine sits outside the LLM path entirely for auditability, and a preprocessing guardrail scans all inputs and retrieved RAG chunks for prompt injection before anything reaches an LLM.

  Retrieval-augmented generation over an internal exception-resolution playbook (PDF) grounds every resolution decision in actual policy rather than model judgment alone.

  ## Tech stack

  LangGraph, LangChain, OpenAI (gpt-4o-mini), ChromaDB, HuggingFace sentence-transformers embeddings, pandas, SQLite.

  ## Data

  All data (customer records, delivery logs, locker inventory, ground-truth labels) is synthetic, generated for this project.

  Built as part of a postgraduate program in AI Agents for Business Applications at UT Austin's McCombs School of Business.
