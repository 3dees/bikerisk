import json
import os
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Feedback Viewer", page_icon="📝", layout="wide")

st.title("📝 Feedback Viewer")

DEFAULT_FILE = "feedback.jsonl"

# File selector
colA, colB = st.columns([3, 1])
with colA:
    feedback_path = st.text_input("Feedback file (JSONL)", value=DEFAULT_FILE, help="Path to feedback.jsonl")
with colB:
    if st.button("🔄 Reload", use_container_width=True):
        st.rerun()

records: List[Dict[str, Any]] = []
if os.path.exists(feedback_path):
    with open(feedback_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed line
                continue
else:
    st.info("No feedback file found yet. Submit some feedback from the main app's sidebar.")

if not records:
    st.stop()

# Normalize into DataFrame
_df = pd.json_normalize(records)
# Ensure columns exist
for col in [
    "timestamp", "type", "page", "text", "email", "rating", "session_id"
]:
    if col not in _df.columns:
        _df[col] = None

# Sidebar filters
st.sidebar.header("Filters")
ftype = st.sidebar.multiselect("Type", sorted([t for t in _df["type"].dropna().unique()]))
page = st.sidebar.multiselect("Page", sorted([t for t in _df["page"].dropna().unique()]))

f = _df
if ftype:
    f = f[f["type"].isin(ftype)]
if page:
    f = f[f["page"].isin(page)]

# Summary metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total feedback", len(_df))
with c2:
    st.metric("After filters", len(f))
with c3:
    try:
        avg_rating = f["rating"].dropna().astype(float).mean()
        st.metric("Avg rating", f"{avg_rating:.2f}" if not pd.isna(avg_rating) else "N/A")
    except Exception:
        st.metric("Avg rating", "N/A")
with c4:
    st.metric("Unique sessions", f["session_id"].nunique())

st.divider()

# Show table
st.subheader("Feedback entries")
st.dataframe(
    f.sort_values(by=["timestamp"], ascending=False),
    use_container_width=True,
    hide_index=True,
)

# Export controls
st.subheader("Export")
export_name = st.text_input("Export CSV filename", value="feedback_export.csv")
if st.button("💾 Export to CSV", type="primary"):
    try:
        f.to_csv(export_name, index=False)
        st.success(f"Exported to {export_name}")
        with open(export_name, "rb") as fh:
            st.download_button(
                label="⬇️ Download CSV",
                data=fh,
                file_name=export_name,
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Failed to export: {e}")
