"""
Agent — Financial data assistant
Comparaisons, calculs, et insights sur les données de marché.

Architecture : tool-calling avec Claude API
  - tool `query_db`  → exécute du SQL en lecture seule sur PostgreSQL
  - tool `compute`   → exécute du code pandas sur les DataFrames résultants
"""

import json
import re
import textwrap
import traceback
from datetime import date

import anthropic
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.db import get_engine, load_assets
from core.formatting import (
    BLUE, BORDER, BG, BG2, BG3, FONT,
    GREEN, GRAY, RED, TEXT, TEXT_DIM, TEXT_MID, YELLOW,
)

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

st.title("agent (Beta)")

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 2000
MAX_TOOL_ROUNDS = 6   # guard against infinite loops

# ── DB schema injecté dans le system prompt ──────────────────────
DB_SCHEMA = """
Tables disponibles (PostgreSQL, read-only) :

fact_prices (symbol TEXT, datetime TEXT 'YYYY-MM-DD', open REAL, high REAL, low REAL, close REAL, volume REAL)
fact_returns (symbol TEXT, date TEXT 'YYYY-MM-DD', r1d REAL, r1w REAL, r1m REAL, r3m REAL, rytd REAL, r1y REAL)
  -- r1d=1 jour, r1w=1 semaine, r1m=1 mois, r3m=3 mois, rytd=YTD, r1y=1 an
  -- Fixed Income : valeurs en bps (diff absolue × 100). Autres : décimal (0.05 = 5%)
ref_assets (symbol TEXT PK, name TEXT, asset_class TEXT, category TEXT,
            asset_type TEXT, currency TEXT, exchange TEXT, country_cd TEXT,
            sector_cd INT, data_source TEXT, is_diff BOOL, is_active INT)
  -- asset_class : 'Equity', 'Fixed Income', 'Commodity', 'Volatility'
  -- is_diff=TRUE → les returns sont en bps (Fixed Income / rates)

Exemples de symboles : SPY, QQQ, VTI, GLD, ^VIX, ^VIX3M, SVIX, VXX, SVXY
Indices VIX : ^VIX (30j), ^VIX3M (3m), ^VIX9D (9j), ^VIX6M (6m)
Rates FRED  : DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS10, DGS30, SOFR

Règles SQL :
- Toujours filtrer is_active=1 sur ref_assets
- Pour les comparaisons de performance, utiliser fact_prices (close) et calculer en Python
- Pour des snapshots rapides de rendements, utiliser fact_returns
- Utiliser NOW() ou CURRENT_DATE pour les dates relatives
- Les dates sont en TEXT format ISO ('YYYY-MM-DD'), ORDER BY datetime ASC
"""

SYSTEM_PROMPT = f"""Tu es un assistant financier quantitatif intégré dans un dashboard de marché.
Tu réponds en français, de manière concise et directe.
Tu as accès à une base de données PostgreSQL de séries temporelles financières.

{DB_SCHEMA}

OUTILS DISPONIBLES :
1. query_db(sql) : exécute une requête SQL SELECT (read-only). Retourne un JSON avec les résultats.
2. compute(code) : exécute du code Python/pandas. Les DataFrames des requêtes précédentes sont disponibles
   dans la variable `dfs` (liste de DataFrames dans l'ordre des appels query_db).
   Retourne le résultat sous forme de texte ou JSON.

INSTRUCTIONS :
- Pour une comparaison de performance : requête fact_prices avec les deux symboles, puis calcule en Python
- Formate les pourcentages avec 2 décimales et le signe (ex: +3.42%)
- Pour les graphiques : retourne un JSON avec clé "chart" contenant {{type, data: [{{symbol, dates, values}}]}}
- Si tu as besoin de tracer, structure le JSON chart comme :
  {{"chart": {{"type": "line", "title": "...", "y_label": "...", "data": [{{"symbol": "SPY", "dates": [...], "values": [...]}}]}}}}
- Sois factuel, pas de bruit. Pas de markdown excessif.
- Date du jour : {date.today().isoformat()}
"""

# ══════════════════════════════════════════════════════════════════
# TOOLS — définitions pour l'API Claude
# ══════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "query_db",
        "description": "Exécute une requête SQL SELECT sur la base de données PostgreSQL. Retourne les résultats en JSON.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "Requête SQL SELECT à exécuter. Doit être en lecture seule (pas de INSERT/UPDATE/DELETE)."
                }
            },
            "required": ["sql"]
        }
    },
    {
        "name": "compute",
        "description": "Exécute du code Python/pandas. `dfs` est une liste des DataFrames retournés par query_db dans l'ordre. Doit retourner un résultat dans la variable `result` (str, dict, ou DataFrame).",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Code Python à exécuter. Utilise pandas/numpy. Assigne le résultat final à `result`."
                }
            },
            "required": ["code"]
        }
    }
]

# ══════════════════════════════════════════════════════════════════
# TOOL EXECUTION
# ══════════════════════════════════════════════════════════════════

def _is_safe_sql(sql: str) -> bool:
    """Bloque toute requête non-SELECT."""
    clean = sql.strip().upper()
    forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE", "GRANT")
    return clean.startswith("SELECT") and not any(w in clean for w in forbidden)


def run_query_db(sql: str) -> tuple[str, pd.DataFrame | None]:
    """Exécute le SQL et retourne (json_result, dataframe)."""
    if not _is_safe_sql(sql):
        return json.dumps({"error": "Seules les requêtes SELECT sont autorisées."}), None
    try:
        with get_engine().connect() as conn:
            df = pd.read_sql(sql, conn)
        if df.empty:
            return json.dumps({"rows": [], "count": 0}), df
        # Limit rows retournés à l'agent pour ne pas exploser le contexte
        preview = df.head(500)
        result = {
            "count": len(df),
            "columns": list(preview.columns),
            "rows": preview.to_dict(orient="records"),
        }
        return json.dumps(result, default=str), df
    except Exception as e:
        return json.dumps({"error": str(e)}), None


def run_compute(code: str, dfs: list[pd.DataFrame]) -> str:
    """Exécute le code Python dans un contexte contrôlé."""
    import numpy as np  # noqa: F401 — disponible dans exec

    local_ns = {
        "pd": pd,
        "np": np,
        "dfs": dfs,
        "result": None,
    }
    try:
        exec(textwrap.dedent(code), local_ns)  # noqa: S102
        result = local_ns.get("result")
        if result is None:
            return json.dumps({"warning": "La variable `result` n'a pas été assignée."})
        if isinstance(result, pd.DataFrame):
            return json.dumps({
                "count": len(result),
                "columns": list(result.columns),
                "rows": result.head(200).to_dict(orient="records"),
            }, default=str)
        if isinstance(result, dict):
            return json.dumps(result, default=str)
        return str(result)
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


# ══════════════════════════════════════════════════════════════════
# CHART RENDERING
# ══════════════════════════════════════════════════════════════════

_CHART_COLORS = [BLUE, YELLOW, GREEN, RED, "#7c3aed", "#0891b2"]


def _try_render_chart(text: str):
    """Cherche un JSON chart dans la réponse et le render avec Plotly."""
    match = re.search(r'\{.*"chart".*\}', text, re.DOTALL)
    if not match:
        return

    try:
        payload = json.loads(match.group())
        chart = payload.get("chart", {})
        data = chart.get("data", [])
        if not data:
            return

        fig = go.Figure()
        for i, serie in enumerate(data):
            color = _CHART_COLORS[i % len(_CHART_COLORS)]
            fig.add_trace(go.Scatter(
                x=serie.get("dates", []),
                y=serie.get("values", []),
                name=serie.get("symbol", f"Serie {i+1}"),
                line=dict(color=color, width=1.8),
                mode="lines",
            ))

        fig.update_layout(
            title=dict(
                text=chart.get("title", ""),
                font=dict(family=FONT, size=12, color=TEXT_DIM),
                x=0,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=BG2,
            font=dict(family=FONT, size=11, color=TEXT),
            hovermode="x unified",
            margin=dict(l=48, r=16, t=36, b=44),
            height=320,
            legend=dict(
                bgcolor="rgba(0,0,0,0)", borderwidth=0,
                font=dict(size=9, color=TEXT_DIM),
                orientation="h", x=0, y=1.08,
            ),
            xaxis=dict(
                showgrid=False, linecolor=BORDER,
                tickfont=dict(size=9, color=TEXT_DIM),
                rangebreaks=[dict(bounds=["sat", "mon"])],
            ),
            yaxis=dict(
                title=chart.get("y_label", ""),
                showgrid=True, gridcolor="#f3f4f6",
                tickfont=dict(size=9, color=TEXT_DIM),
                tickformat=".1%",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass  # Si le JSON est malformé on ignore silencieusement


# ══════════════════════════════════════════════════════════════════
# AGENT LOOP
# ══════════════════════════════════════════════════════════════════

def run_agent(user_message: str, status_container):
    """
    Boucle principale tool-calling.
    Retourne le texte final de la réponse.
    """
    client = anthropic.Anthropic()

    messages = [{"role": "user", "content": user_message}]
    dfs_accumulated: list[pd.DataFrame] = []

    for round_n in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Ajoute la réponse de l'assistant dans l'historique
        messages.append({"role": "assistant", "content": response.content})

        # Si stop_reason == end_turn → réponse finale
        if response.stop_reason == "end_turn":
            final_text = " ".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            return final_text

        # Si pas d'outil appelé → on sort
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            final_text = " ".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            return final_text

        # Exécution des outils
        tool_results = []
        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_input = tool_use.input

            if tool_name == "query_db":
                sql = tool_input.get("sql", "")
                status_container.markdown(
                    f"🔍 SQL : <code>{sql[:120]}{'…' if len(sql) > 120 else ''}",
                    unsafe_allow_html=True,
                )
                result_str, df = run_query_db(sql)
                if df is not None:
                    dfs_accumulated.append(df)

            elif tool_name == "compute":
                code = tool_input.get("code", "")
                status_container.markdown(
                    f"⚙️ Calcul Python…",
                    unsafe_allow_html=True,
                )
                result_str = run_compute(code, dfs_accumulated)
            else:
                result_str = json.dumps({"error": f"Outil inconnu : {tool_name}"})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result_str,
            })

        # Injecte les résultats dans le fil de messages
        messages.append({"role": "user", "content": tool_results})

    return "⚠️ Nombre maximum de rounds atteint sans réponse finale."


# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════

# ── Initialisation session state ─────────────────────────────────
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []   # [{role, content}]

# ── CSS ──────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.agent-user {{
    background:{BG3};
    border-radius:8px 8px 2px 8px;
    padding:10px 14px;
    margin:8px 0 4px auto;
    max-width:80%;
    font-family:{FONT};
    font-size:13px;
    color:{TEXT};
    width:fit-content;
    margin-left:auto;
}}
.agent-assistant {{
    background:{BG2};
    border:1px solid {BORDER};
    border-radius:2px 8px 8px 8px;
    padding:12px 16px;
    margin:4px 0 8px 0;
    max-width:90%;
    font-family:{FONT};
    font-size:13px;
    color:{TEXT};
    line-height:1.6;
}}
.agent-thinking {{
    font-family:{FONT};
    font-size:11px;
    color:{TEXT_DIM};
    padding:6px 0;
    min-height:20px;
}}
</style>
""", unsafe_allow_html=True)

# ── Suggestions rapides ───────────────────────────────────────────
SUGGESTIONS = [
    "Compare la performance SPY vs QQQ sur 3 mois et affiche un graphique",
    "Compare la volatilité mensuelle moyenne de l'or vs. Nasdaq  sur 1 an",
    "Combien de fois le VIX a depassé 30 sur l'historique et pourquoi ?",
]

if not st.session_state.agent_history:
    st.markdown(
        f"<p style='font-family:{FONT};font-size:12px;color:{TEXT_DIM};"
        f"margin:0 0 12px 0'>Quelques questions pour commencer :</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(SUGGESTIONS))
    for col, sug in zip(cols, SUGGESTIONS):
        if col.button(sug, use_container_width=True, key=f"sug_{sug[:20]}"):
            st.session_state._pending_question = sug
            st.rerun()

# ── Historique du chat ────────────────────────────────────────────
chat_area = st.container()
with chat_area:
    for msg in st.session_state.agent_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='agent-user'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            # Texte de la réponse (sans le JSON chart s'il est présent)
            display_text = re.sub(r'\{[^{}]*"chart"[^{}]*\}', "", msg["content"], flags=re.DOTALL).strip()
            st.markdown(
                f"<div class='agent-assistant'>{display_text}</div>",
                unsafe_allow_html=True,
            )
            # Chart éventuel
            _try_render_chart(msg["content"])

# ── Zone de saisie ────────────────────────────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

with st.form("agent_form", clear_on_submit=True):
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        user_input = st.text_input(
            "Question",
            placeholder="ex: Compare SPY vs QQQ sur 6 mois…",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("→", use_container_width=True)

# Gestion de la suggestion cliquée
if hasattr(st.session_state, "_pending_question"):
    user_input = st.session_state._pending_question
    submitted = True
    del st.session_state._pending_question

# ── Traitement de la question ─────────────────────────────────────
if submitted and user_input.strip():
    question = user_input.strip()

    # Ajoute dans l'historique
    st.session_state.agent_history.append({"role": "user", "content": question})

    # Affiche la question immédiatement
    with chat_area:
        st.markdown(
            f"<div class='agent-user'>{question}</div>",
            unsafe_allow_html=True,
        )

    # Zone de status pendant le traitement
    with chat_area:
        status_box = st.empty()
        status_box.markdown(
            f"<div class='agent-thinking'>Analyse en cours…</div>",
            unsafe_allow_html=True,
        )

    try:
        answer = run_agent(question, status_box)
    except Exception as e:
        answer = f"Erreur inattendue : {e}"

    # Efface le status
    status_box.empty()

    # Ajoute la réponse dans l'historique
    st.session_state.agent_history.append({"role": "assistant", "content": answer})

    # Affiche la réponse finale
    with chat_area:
        display_text = re.sub(r'\{[^{}]*"chart"[^{}]*\}', "", answer, flags=re.DOTALL).strip()
        st.markdown(
            f"<div class='agent-assistant'>{display_text}</div>",
            unsafe_allow_html=True,
        )
        _try_render_chart(answer)

    st.rerun()

# ── Bouton reset ──────────────────────────────────────────────────
if st.session_state.agent_history:
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    if st.button("Effacer la conversation", key="clear_chat"):
        st.session_state.agent_history = []
        st.rerun()