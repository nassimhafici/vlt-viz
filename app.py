import streamlit as st
import base64
FONT        = "'Helvetica Neue', Helvetica, Arial, sans-serif"
ACCENT      = "#6366f1"
ACCENT_SOFT = "#eef2ff"
ACCENT_MID  = "#a5b4fc"

st.set_page_config(
    page_title="VLT",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


_SVG = f"""<svg width="160" height="40" viewBox="0 0 160 40" fill="none"
     xmlns="http://www.w3.org/2000/svg">
  <rect width="36" height="36" x="2" y="2" rx="8" fill="#111827"/>
  <polyline points="8,28 14,20 19,23 26,12 32,15"
            stroke="white" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <circle cx="32" cy="18" r="2.2" fill="{ACCENT_MID}"/>
  <text x="48" y="22"
        font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="18" font-weight="700" letter-spacing="3"
        fill="#111827">VLT</text>
  <text x="49" y="32"
        font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="8" font-weight="550" letter-spacing="0.9"
        fill="#9ca3af">MARKET INTELLIGENCE</text>
</svg>"""

_data_uri = "data:image/svg+xml;base64," + base64.b64encode(_SVG.encode()).decode()

# Doit être appelé AVANT st.navigation() pour s'afficher au-dessus
st.logo(_data_uri, size="large")
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"], .stApp {
  background-color: #f9fafb;
  color: #111827;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

*, *::before, *::after { box-sizing: border-box; }

.block-container {
  padding: 32px 40px 48px 40px !important;
  max-width: 100% !important;
  background: #f9fafb;
}

/* ───────────────── Sidebar ───────────────── */

[data-testid="stSidebar"] {
  background-color: #ffffff !important;
  border-right: 1px solid #e5e7eb !important;
}

/* ── Collapse button (open sidebar) ── */
[data-testid="stSidebarCollapseButton"] {
  position: absolute !important;
  top: 8px !important;
  right: -14px !important;
  z-index: 1000 !important;
}

/* ── Expand button (collapsed sidebar) ── */
[data-testid="stSidebarCollapsedControl"] {
  position: fixed !important;
  top: 12px !important;
  left: 12px !important;
  z-index: 1000 !important;
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
}

[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 6px !important;
  color: #6b7280 !important;
  padding: 4px 6px !important;
  min-height: 28px !important;
  height: 28px !important;
  width: 28px !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
  cursor: pointer !important;
}

[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover {
  background: #f3f4f6 !important;
  border-color: #2563eb !important;
  color: #2563eb !important;
}

/* ───────────────── Headings ───────────────── */

h1 {
  font-size: 10px !important;
  font-weight: 600 !important;
  color: #9ca3af !important;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  border-bottom: 1px solid #e5e7eb;
  padding-bottom: 16px;
  margin-bottom: 28px !important;
  margin-top: 0 !important;
}

h2, h3 {
  font-size: 10px !important;
  font-weight: 600 !important;
  color: #9ca3af !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 36px !important;
  margin-bottom: 12px !important;
}

/* ───────────────── Tabs ───────────────── */

[data-testid="stTabs"] button {
  font-size: 11px !important;
  font-weight: 500;
  letter-spacing: 0.05em;
  color: #6b7280 !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  padding: 8px 18px !important;
}

[data-testid="stTabs"] button:hover {
  color: #111827 !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
  color: #2563eb !important;
  border-bottom-color: #2563eb !important;
  font-weight: 600;
}

[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid #e5e7eb !important;
}

/* ───────────────── Buttons ───────────────── */

[data-testid="stSidebar"] .stButton button {
  background: #ffffff !important;
  border: 1px solid #d1d5db !important;
  border-radius: 6px !important;
  color: #374151 !important;
  font-size: 11px !important;
  font-weight: 500;
  padding: 6px 16px !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

[data-testid="stSidebar"] .stButton button:hover {
  border-color: #2563eb !important;
  color: #2563eb !important;
  background: #eff6ff !important;
}

/* ───────────────── Inputs ───────────────── */

.stSelectbox > div > div,
.stMultiSelect > div > div {
  background-color: #ffffff !important;
  border: 1px solid #d1d5db !important;
  border-radius: 6px !important;
  font-size: 13px !important;
  color: #111827 !important;
  min-height: 36px !important;
}

.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
  border-color: #2563eb !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* ───────────────── Dataframes ───────────────── */

.stDataFrame {
  border: 1px solid #e5e7eb !important;
  border-radius: 8px !important;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ───────────────── Charts ───────────────── */

.stPlotlyChart {
  background: #ffffff;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  overflow: hidden;
}

/* ───────────────── Scrollbar ───────────────── */

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #f9fafb; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #9ca3af; }

/* ───────────────── Misc ───────────────── */

#MainMenu, footer { visibility: hidden; }

[data-testid="stDecoration"] { display: none; }
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("pages/1_market_overview.py", title="OVERVIEW"),
    st.Page("pages/2_asset_explorer.py",  title="EXPLORER"),
    st.Page("pages/3_compare.py",         title="COMPARE"),
    st.Page("pages/4_rates.py",           title="RATES"),
    st.Page("pages/5_shortvol.py",        title="SHORTVOL"),
    st.Page("pages/6_residual_momentum.py", title="RES. MOMENTUM"),
agent     st.Page("pages/7_🧠agent.py", title="AGENT"),
])

pg.run()