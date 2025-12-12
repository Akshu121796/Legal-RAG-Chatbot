import streamlit as st
from pathlib import Path
import base64
import mimetypes

CSS_FILE = Path("ui/styles.css")
CARD_HTML_FILE = Path("ui/card.html")


def load_css():
    if CSS_FILE.exists():
        css = CSS_FILE.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _local_path_to_data_uri(path_str: str) -> str:
    """If the path points to a local file, return a data URI, else return original."""
    if not path_str:
        return ""

    p = Path(path_str)
    if not p.exists():
        # Try relative to workspace root
        alt = Path(".") / path_str
        if alt.exists():
            p = alt
        else:
            return path_str

    mime, _ = mimetypes.guess_type(p.as_posix())
    if not mime:
        mime = "application/octet-stream"

    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def render_card(item):
    if not CARD_HTML_FILE.exists():
        st.write(item.get("title", ""))
        return

    html = CARD_HTML_FILE.read_text(encoding="utf-8")

    # If the item references a local image file, encode it as a data URI
    img_val = item.get("image", "")
    if isinstance(img_val, str) and img_val:
        data_uri = _local_path_to_data_uri(img_val)
        item = dict(item)  # shallow copy so we don't mutate caller data
        item["image"] = data_uri

    for key, value in item.items():
        html = html.replace(f"{{{{{key}}}}}", str(value))
    st.markdown(html, unsafe_allow_html=True)


def render_section_title(text):
    st.markdown(f"<div class='pp-section-title'>{text}</div>", unsafe_allow_html=True)
