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
  padding-top: 24px;
  box-shadow: 1px 0 0 #e5e7eb;
}
[data-testid="stSidebar"] section { padding: 0 18px; }

[data-testid="stSidebarNav"] { padding: 0 0 12px 0; }
[data-testid="stSidebarNav"] a {
  display: block;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  font-size: 11.5px;
  font-weight: 500;
  color: #6b7280 !important;
  letter-spacing: 0.02em;
  padding: 8px 16px;
  border-left: 2px solid transparent;
  text-decoration: none !important;
  transition: color 0.15s;
  border-radius: 0 4px 4px 0;
  margin-bottom: 2px;
}
[data-testid="stSidebarNav"] a:hover {
  color: #111827 !important;
  background: #f3f4f6;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
  color: #2563eb !important;
  border-left-color: #2563eb;
  background: #eff6ff;
  font-weight: 600;
}

/* ── Sidebar toggle — always visible ── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  pointer-events: auto !important;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] button {
  opacity: 1 !important;
  visibility: visible !important;
  pointer-events: auto !important;
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

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"], [data-testid="stToolbar"] { display: none; }
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("pages/1_market_overview.py", title="Overview"),
    st.Page("pages/2_asset_explorer.py",  title="Explorer"),
    st.Page("pages/3_heatmap.py",         title="Heatmap"),
    st.Page("pages/4_compare.py",         title="Compare"),
    st.Page("pages/5_rates.py",           title="Rates"),
    st.Page("pages/6_indices.py",         title="Indices & Vol"),
    st.Page("pages/7_short_vol.py",       title="Short Vol"),
])
pg.run()