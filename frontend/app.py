"""SatQuery-AI Streamlit frontend.

UI Flow
-------
1.  User uploads one or more raster files (.tif/.tiff/.png/.jpg).
2.  User types a natural-language question in the chat input.
3.  On submit, the files and query are POSTed to the orchestrator /query endpoint.
4.  The assistant answer is appended to the chat history.
5.  An expandable "Execution Trace" section shows the per-node audit log.
"""

from __future__ import annotations

import os
import json
import time
from typing import Any

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ORCHESTRATOR_URL: str = os.getenv(
    "ORCHESTRATOR_URL", "http://localhost:8000"
).rstrip("/")

_QUERY_ENDPOINT: str = f"{ORCHESTRATOR_URL}/query"
_HEALTH_ENDPOINT: str = f"{ORCHESTRATOR_URL}/health"

_ALLOWED_TYPES: list[str] = ["tif", "tiff", "png", "jpg", "jpeg"]

_REQUEST_TIMEOUT: int = int(os.getenv("FRONTEND_REQUEST_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SatQuery-AI",
    page_icon="🛰️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state["chat_history"]: list[dict[str, str]] = []

if "last_trace" not in st.session_state:
    st.session_state["last_trace"]: list[dict] = []

# ---------------------------------------------------------------------------
# Sidebar — file upload & settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛰️ SatQuery-AI")
    st.caption("Satellite Imagery Question Answering")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload raster image(s)",
        type=_ALLOWED_TYPES,
        accept_multiple_files=True,
        help="GeoTIFF, PNG, or JPEG files. Multi-image upload enables "
             "cross-modal and bi-temporal analysis.",
    )

    st.divider()

    # Orchestrator health indicator
    with st.spinner("Checking orchestrator…"):
        try:
            health_resp = requests.get(_HEALTH_ENDPOINT, timeout=5)
            if health_resp.ok:
                st.success("Orchestrator: online", icon="✅")
            else:
                st.warning("Orchestrator: degraded", icon="⚠️")
        except requests.exceptions.ConnectionError:
            st.error("Orchestrator: unreachable", icon="🔴")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state["chat_history"] = []
        st.session_state["last_trace"] = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main panel — chat
# ---------------------------------------------------------------------------

st.header("Ask a question about your satellite image(s)")

# Render existing chat history.
for turn in st.session_state["chat_history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# Chat input widget — returns the submitted string (or None).
user_query: str | None = st.chat_input(
    placeholder="e.g. What land-cover classes are visible in this scene?",
    disabled=not uploaded_files,
)

if not uploaded_files:
    st.info("👈  Upload at least one raster image to start chatting.", icon="ℹ️")

# ---------------------------------------------------------------------------
# Handle a new query submission
# ---------------------------------------------------------------------------

if user_query and uploaded_files:
    # Append user message to history and render it immediately.
    st.session_state["chat_history"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Call the orchestrator.
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Analysing…")

        try:
            t0 = time.perf_counter()
            response = requests.post(
                _QUERY_ENDPOINT,
                data={"query": user_query},
                files=[
                    ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                    for f in uploaded_files
                ],
                timeout=_REQUEST_TIMEOUT,
            )
            elapsed = time.perf_counter() - t0

            if response.ok:
                payload: dict[str, Any] = response.json()
                messages: list[str] = payload.get("messages", [])
                trace: list[dict] = payload.get("execution_trace", [])
                input_config: str = payload.get("input_config", "unknown")

                # The assistant answer is the last message (everything after
                # the original user query).
                assistant_messages = messages[1:] if len(messages) > 1 else messages
                answer_text = "\n\n".join(assistant_messages) if assistant_messages else "(no answer)"

                placeholder.markdown(answer_text)
                st.caption(
                    f"Input config: **{input_config}** · {elapsed:.1f}s · "
                    f"request_id: `{payload.get('request_id', 'n/a')}`"
                )

                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": answer_text}
                )
                st.session_state["last_trace"] = trace

            else:
                # Parse and surface the FastAPI error detail.
                try:
                    err_detail = response.json().get("detail", response.text)
                except Exception:
                    err_detail = response.text

                error_msg = f"❌ Error {response.status_code}: {err_detail}"
                placeholder.error(error_msg)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": error_msg}
                )

        except requests.exceptions.ConnectionError:
            msg = "❌ Cannot reach the orchestrator. Is it running?"
            placeholder.error(msg)
            st.session_state["chat_history"].append({"role": "assistant", "content": msg})

        except requests.exceptions.Timeout:
            msg = f"❌ Request timed out after {_REQUEST_TIMEOUT}s."
            placeholder.error(msg)
            st.session_state["chat_history"].append({"role": "assistant", "content": msg})

# ---------------------------------------------------------------------------
# Execution trace expander (persists across re-renders)
# ---------------------------------------------------------------------------

if st.session_state["last_trace"]:
    with st.expander("🔍 Execution Trace", expanded=False):
        for i, entry in enumerate(st.session_state["last_trace"], start=1):
            node_name: str = entry.get("node", f"node_{i}")
            result: Any = entry.get("result", {})

            st.subheader(f"Step {i} — `{node_name}`")
            st.json(result)
