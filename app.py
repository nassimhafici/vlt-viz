import streamlit as st

st.set_page_config(
    page_title="VLT",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"

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

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background-color: #ffffff !important;
  border-right: 1px solid #e5e7eb !important;
}

/* ── Headings ── */
h1 {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
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
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  color: #9ca3af !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 36px !important;
  margin-bottom: 12px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
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
  background: transparent !important;
  gap: 0;
}

hr {
  border: none;
  border-top: 1px solid #e5e7eb !important;
  margin: 24px 0 !important;
}

[data-testid="stMetric"] { display: none !important; }

/* ── Form labels ── */
.stSelectbox label, .stMultiSelect label, .stRadio label,
.stSlider label, .stDateInput label, .stCheckbox label {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
  font-size: 10px !important;
  font-weight: 600;
  color: #6b7280 !important;
  text-transform: uppercase;
  letter-spacing: 0.07em;
}

/* ── Select boxes ── */
.stSelectbox > div > div, .stMultiSelect > div > div {
  background-color: #ffffff !important;
  border: 1px solid #d1d5db !important;
  border-radius: 6px !important;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
  font-size: 13px !important;
  color: #111827 !important;
  min-height: 36px !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
  border-color: #2563eb !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
[data-baseweb="option"] {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
  font-size: 12px !important;
  color: #374151 !important;
  background-color: transparent !important;
  padding: 7px 14px !important;
}
[data-baseweb="option"]:hover {
  background-color: #eff6ff !important;
  color: #2563eb !important;
}
[data-testid="stSelectboxVirtualDropdown"], [data-baseweb="popover"] {
  background-color: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 8px !important;
  box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important;
}

/* ── Radio ── */
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
  font-size: 12px !important;
  color: #374151 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] div[role="slider"] {
  background-color: #2563eb !important;
  border-color: #2563eb !important;
}
[data-testid="stSlider"] p {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
  font-size: 10px !important;
  color: #6b7280 !important;
}

/* ── Generic buttons (sidebar only, not period pills) ── */
[data-testid="stSidebar"] .stButton button,
.stDownloadButton button {
  background: #ffffff !important;
  border: 1px solid #d1d5db !important;
  border-radius: 6px !important;
  color: #374151 !important;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
  font-size: 11px !important;
  font-weight: 500;
  padding: 6px 16px !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
  height: auto !important;
  transition: all 0.15s;
}
[data-testid="stSidebar"] .stButton button:hover,
.stDownloadButton button:hover {
  border-color: #2563eb !important;
  color: #2563eb !important;
  background: #eff6ff !important;
  box-shadow: none !important;
}

/* ── Dataframe ── */
.stDataFrame {
  border: 1px solid #e5e7eb !important;
  border-radius: 8px !important;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── Alerts ── */
[data-testid="stAlert"] {
  background-color: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 8px !important;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  font-size: 13px;
  color: #374151 !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── Multiselect tags ── */
[data-baseweb="tag"] {
  background-color: #eff6ff !important;
  border-radius: 4px !important;
  border: 1px solid #bfdbfe !important;
}
[data-baseweb="tag"] span {
  color: #2563eb !important;
  font-size: 11px !important;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #f9fafb; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #9ca3af; }

/* ── Plotly chart wrapper ── */
.stPlotlyChart {
  background: #ffffff;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  overflow: hidden;
}

/* ── Date input ── */
.stDateInput input {
  background: #ffffff !important;
  border: 1px solid #d1d5db !important;
  border-radius: 6px !important;
  color: #111827 !important;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
  font-size: 12px !important;
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"], [data-testid="stToolbar"] { display: none; }

/* ── Period pills — all real st.button, active gets ● prefix ── */
div[data-testid="stHorizontalBlock"] .stButton button {
  width: 100% !important;
  height: 28px !important;
  min-height: 28px !important;
  padding: 0 6px !important;
  border-radius: 5px !important;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  border: 1px solid #e5e7eb !important;
  background: #ffffff !important;
  color: #6b7280 !important;
  box-shadow: none !important;
  transition: all 0.12s !important;
  cursor: pointer !important;
  line-height: 1 !important;
}
div[data-testid="stHorizontalBlock"] .stButton button:hover {
  border-color: #2563eb !important;
  color: #2563eb !important;
  background: #eff6ff !important;
  box-shadow: none !important;
}
div[data-testid="stHorizontalBlock"] .stButton button:focus {
  box-shadow: none !important; outline: none !important;
}
/* Active pill — button whose text starts with ● */
div[data-testid="stHorizontalBlock"] .stButton button:has(div[data-testid="stMarkdownContainer"] p:first-child) {
  /* fallback — overridden below */
}
div[data-testid="stHorizontalBlock"] .stButton button[kind="secondary"] p,
div[data-testid="stHorizontalBlock"] .stButton button p {
  margin: 0 !important; padding: 0 !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
}
/* Active: we inject data-active via a wrapper div */
div.pill-active-wrap .stButton button {
  background: #2563eb !important;
  border-color: #2563eb !important;
  color: #ffffff !important;
  font-weight: 700 !important;
}
div.pill-active-wrap .stButton button:hover {
  background: #1d4ed8 !important;
  border-color: #1d4ed8 !important;
  color: #ffffff !important;
}

"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("pages/1_market_overview.py", title="OVERVIEW"),
    st.Page("pages/2_asset_explorer.py",  title="EXPLORER"),
    st.Page("pages/3_compare.py",         title="COMPARE"),
    st.Page("pages/4_rates.py",           title="RATES"),
    st.Page("pages/5_shortvol.py",        title="SHORTVOL"),
])
pg.run()