"""
Flask API Server for Neural Introspection and Confabulation Detection

This module provides a RESTful API interface for running confabulation detection
experiments through a web interface. It wraps the auditor.py core functionality
with HTTP endpoints for ease of use.

Endpoints:
    GET  /              - Serve index.html web interface
    GET  /models        - List available models and default selection
    POST /run_audit     - Run single trace with confabulation detection
    POST /run_eval      - Run evaluation suite with metrics export

Configuration:
    EVAL_EXPORT_DIR environment variable sets the JSON export location

Usage:
    export EVAL_EXPORT_DIR="./eval_logs"
    export HACKATHON_TEAM_ID="your_team_id"
    export HACKATHON_API_TOKEN="your_token"
    flask --app app run

The server preloads the SentenceTransformer embedding model at startup for
efficient confabulation score computation.

Author: Research Team
Date: 2025
License: MIT
"""

import os
from flask import Flask, jsonify, request
import auditor

# Import configuration from auditor module
DEFAULT_MODEL_ID = auditor.DEFAULT_MODEL_ID
MODEL_CATALOG = auditor.MODEL_CATALOG
EVAL_EXPORT_DIR = os.environ.get("EVAL_EXPORT_DIR")

# Initialize Flask app with static file serving for index.html
app = Flask(__name__, static_folder=".", static_url_path="")

# Preload embedding model once
try:
    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    auditor.load_embedding_model()
    print("Embedding model loaded successfully.")
except Exception as exc:
    print(f"CRITICAL ERROR: Could not load embedding model. {exc}")
    print("Please ensure 'sentence-transformers' is installed: pip install sentence-transformers")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/run_audit", methods=["POST"])
def run_audit():
    """
    Execute a single agentic reasoning trace with confabulation detection.
    
    This endpoint runs a complete audit graph for a given user prompt, computing
    confabulation scores and introspection probes at each step.
    
    Request Body (JSON):
        {
            "prompt": "Research the capital of France",  // Required
            "model": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"  // Optional
        }
    
    Response (JSON):
        List of graph nodes, each containing:
            - step: Step number (1-indexed)
            - state: Current reasoning state (PLAN, TOOL_CALL, FINAL_ANSWER, etc.)
            - thought: Model's internal reasoning
            - action: Tool call or action description
            - confabulation_score: Semantic similarity score [0.0, 1.0]
            - audit_reason: Human-readable explanation of alignment
            - introspection_alignment: Self-reported ALIGNED/MISALIGNED/UNSURE
            - introspection_confidence: Confidence in self-report [0.0, 1.0]
            - introspection_explanation: Model's justification
            - latency_ms: Step execution time
    
    Example:
        curl -X POST http://localhost:5000/run_audit \\
             -H "Content-Type: application/json" \\
             -d '{"prompt": "What is the weather in Paris?"}'
    
    Returns:
        200: JSON trace of reasoning steps
        400: Missing prompt in request
        500: Execution error (API failure, network issues, etc.)
    """
    data = request.json or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    model_id = data.get("model") or DEFAULT_MODEL_ID
    try:
        trace = auditor.run_full_audit_graph(prompt, model_id=model_id)
        return jsonify(trace)
    except Exception as exc:
        print(f"Error in /run_audit: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"default": DEFAULT_MODEL_ID, "catalog": MODEL_CATALOG})


@app.route("/run_eval", methods=["POST"])
def run_eval():
    data = request.json or {}
    model_id = data.get("model") or DEFAULT_MODEL_ID
    scenario_ids = data.get("scenarios")
    count = data.get("count")
    export_path = data.get("export_path") or EVAL_EXPORT_DIR
    try:
        max_scenarios = int(count) if count is not None and str(count).strip().isdigit() else None
    except (ValueError, TypeError):
        max_scenarios = None

    try:
        result = auditor.run_eval_suite(
            model_id=model_id,
            scenario_ids=scenario_ids,
            max_scenarios=max_scenarios,
            export_path=export_path,
        )
        return jsonify(result)
    except Exception as exc:
        print(f"Error in /run_eval: {exc}")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
