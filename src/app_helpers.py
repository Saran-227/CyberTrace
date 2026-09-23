"""Application session state and helper utilities."""

from typing import Any, Dict
import streamlit as st

def get_or_create_session_state(key: str, default_value: Any) -> Any:
    """Retrieve or initialize key in st.session_state."""
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]
