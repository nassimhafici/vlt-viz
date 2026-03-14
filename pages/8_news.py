"""
VLT — pages/8_news.py

Pipeline :
  1. Fetch RSS (7 sources) → ~105 articles bruts
  2. Déduplication + nettoyage + filtre âge < 6h
  3. claude-haiku-4-5 : filtre et score les articles selon pertinence institutionnelle
  4. Affichage top articles triés par score avec bandeau déroulant

Cache : 30 min fixe partagé entre toutes les sessions (non-paramétrisable).
"""

import os
import streamlit as st
import feedparser
import anthropic
import yfinance as yf
import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import re
import hashlib

# ── Constantes ────────────────────────────────────────────────────────────────

REFRESH_MINUTES       = 30
MAX_ARTICLES_TO_HAIKU = 80
TOP_N_DISPLAY         = 30
MAX_AGE_HOURS         = 6

MONTREAL_TZ = ZoneInfo("America/Montreal")

RSS_FEEDS = {
    "Reuters Markets":   "https://feeds.reuters.com/reuters/companyNews",
    "WSJ Markets":       "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "Yahoo Finance":     "https://finance.yahoo.com/news/rssindex",
    "MarketWatch":       "https://feeds.marketwatch.com/marketwatch/topstories/",
    "FT Markets":        "https://www.ft.com/rss/home/uk",
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
    "Seeking Alpha":     "https://seekingalpha.com/market_currents.xml",
}

# Symboles yfinance → (label, format prix)
TICKER_DEFS = [
    ("^GSPC",    "S&P 500",  "{:.0f}"),
    ("^IXIC",    "NASDAQ",   "{:.0f}"),
    ("^DJI",     "DOW",      "{:.0f}"),
    ("^VIX",     "VIX",      "{:.2f}"),
    ("CL=F",     "WTI",      "${:.2f}"),
    ("BZ=F",     "BRENT",    "${:.2f}"),
    ("GC=F",     "GOLD",     "${:.0f}"),
    ("^TNX",     "10Y UST",  "{:.2f}%"),
    ("EURUSD=X", "EUR/USD",  "{:.4f}"),
    ("USDCAD=X", "USD/CAD",  "{:.4f}"),
    ("USDJPY=X", "USD/JPY",  "{:.2f}"),
    ("USDCHF=X", "USD/CHF",  "{:.4f}"),
]

HAIKU_SYSTEM = """You are a senior financial analyst at a major institutional asset manager 
with a total portfolio mandate (equities, rates, volatility, macro).

Your job: filter a list of raw news articles and return ONLY those that matter 
to a professional investor managing a global multi-asset portfolio.
Articles about volatility, macro and momentum matter the most alongside with big world news.

KEEP articles about:
- Macroeconomic data and surprises (CPI, PCE, GDP, jobs, PMI, trade balance)
- Central bank decisions, signals, speeches (Fed, ECB, BoE, BoC, BoJ)
- Interest rates, yield curve dynamics, bond market flows
- Equity market structure: index moves, sector rotations, risk-on/off
- Volatility regime: VIX, options market, vol surfaces, tail risk
- Geopolitical events with direct market impact (wars, sanctions, supply shocks)
- Commodity prices with macro significance (oil, gold, copper)
- Major earnings from mega-cap or systemically important firms (>$100B market cap)
- Credit markets, spreads, financial conditions index
- Currency dynamics with macro relevance (DXY, USD, EUR, JPY, CNY, CHF, CAD) and score higher news for CAD and CHF
- Structural market themes: AI capex cycle, energy transition, deglobalization

DISCARD articles about:
- Individual small/mid-cap stocks (<$50B market cap) unless systemic impact
- Retail investor sentiment, meme stocks, crypto (unless macro contagion)
- Company-specific operational news without market-wide implications
- Human interest, lifestyle, personal finance for retail investors
- Redundant articles covering the same event as a higher-quality piece already kept
- Articles older than 12 hours (age_min > 720) unless highly significant
- Analyst price target changes on individual stocks
- Earnings from companies not systemically important

SCORING (1-10):
10 — Fed decision, major geopolitical shock, systemic market event
8-9 — Significant macro data surprise, major central bank signal
6-7 — Relevant market color, notable sector move, important earnings
<5  — Useful context, secondary macro data

Return a JSON array. Each element:
{
  "id": "<article id>",
  "score": <int 1-10>,
  "rationale": "<one sentence why this matters institutionally>",
  "theme": "<one of: RATES | EQUITIES | VOLATILITY | MACRO | GEOPOLITICS | COMMODITIES | FX | CREDIT>"
}

Only include articles with score >= 5. Return ONLY valid JSON, no markdown, no preamble."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    for ent, rep in [("&amp;","&"),("&lt;","<"),("&gt;",">"),
                     ("&nbsp;"," "),("&#39;","'"),("&quot;",'"')]:
        text = text.replace(ent, rep)
    return re.sub(r"\s+", " ", text).strip()


def parse_date(entry) -> datetime:
    for field in ("published_parsed", "updated_parsed"):
        t = getattr(entry, field, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc).astimezone(MONTREAL_TZ)
            except Exception:
                pass
    return datetime.now(MONTREAL_TZ)


def article_id(entry) -> str:
    key = getattr(entry, "link", "") or getattr(entry, "title", "") or str(time.time())
    return hashlib.md5(key.encode()).hexdigest()[:8]


def age_label(minutes: int) -> str:
    if minutes < 1:    return "À l'instant"
    if minutes < 60:   return f"{minutes}m"
    if minutes < 1440: return f"{minutes // 60}h{minutes % 60:02d}"
    return f"{minutes // 1440}j"


def theme_color(theme: str) -> tuple[str, str]:
    palette = {
        "RATES":       ("#bcd5f5", "#1d4ed8"),
        "EQUITIES":    ("#f0fdf4", "#15803d"),
        "VOLATILITY":  ("#fefce8", "#a16207"),
        "MACRO":       ("#faf5ff", "#7e22ce"),
        "GEOPOLITICS": ("#fff1f2", "#be123c"),
        "COMMODITIES": ("#fff7ed", "#c2410c"),
        "FX":          ("#f0fdfa", "#0f766e"),
        "CREDIT":      ("#f8fafc", "#475569"),
    }
    return palette.get(theme, ("#f3f4f6", "#374151"))


def score_color(score: int) -> str:
    if score >= 9: return "#dc2626"
    if score >= 7: return "#d97706"
    if score >= 5: return "#2563eb"
    return "#9ca3af"


# ── Fetch + Score (cache partagé) ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_and_score_all(_ttl_bucket: int) -> dict:
    """
    Cache partagé entre toutes les sessions.
    Un seul appel Haiku + yfinance par fenêtre de REFRESH_MINUTES minutes.
    Retourne {"articles": [...], "tickers": [...]}.
    """
    # ── 0. Prix live via yfinance ─────────────────────────────────────────
    tickers_out = []
    try:
        symbols = [t[0] for t in TICKER_DEFS]
        data    = yf.download(symbols, period="2d", interval="1d",
                              progress=False, auto_adjust=True)["Close"]
        for sym, label, fmt in TICKER_DEFS:
            try:
                col   = data[sym].dropna()
                price = float(col.iloc[-1])
                prev  = float(col.iloc[-2]) if len(col) >= 2 else price
                chg   = (price - prev) / prev * 100
                tickers_out.append({
                    "label":  label,
                    "value":  fmt.format(price),
                    "change": f"{abs(chg):.2f}%",
                    "up":     chg >= 0,
                })
            except Exception:
                pass
    except Exception:
        pass  # Fallback : ruban vide, pas bloquant

    # ── 1. Fetch RSS (toutes sources) ────────────────────────────────────
    raw  = []
    seen = set()

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                aid = article_id(entry)
                if aid in seen:
                    continue
                seen.add(aid)
                title   = clean_html(getattr(entry, "title", ""))
                summary = clean_html(getattr(entry, "summary", "") or getattr(entry, "description", ""))
                link    = getattr(entry, "link", "#")
                date    = parse_date(entry)
                age_min = int((datetime.now(MONTREAL_TZ) - date).total_seconds() / 60)
                if age_min > MAX_AGE_HOURS * 60:
                    continue
                if title:
                    raw.append({
                        "id":      aid,
                        "source":  source,
                        "title":   title,
                        "summary": summary[:300],
                        "link":    link,
                        "date":    date,
                        "age_min": age_min,
                    })
        except Exception:
            pass

    raw.sort(key=lambda a: a["age_min"])
    raw = raw[:MAX_ARTICLES_TO_HAIKU]

    if not raw:
        return {"articles": [], "tickers": tickers_out}

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"articles": [{**a, "score": 5, "rationale": "", "theme": "MACRO"} for a in raw],
                "tickers":  tickers_out}

    client = anthropic.Anthropic(api_key=api_key)
    payload = [
        {"id": a["id"], "title": a["title"],
         "summary": a["summary"][:150], "source": a["source"], "age_min": a["age_min"]}
        for a in raw
    ]

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=HAIKU_SYSTEM,
            messages=[{"role": "user", "content":
                f"Filter and score these {len(payload)} articles:\n\n"
                f"{json.dumps(payload, ensure_ascii=False)}"}],
        )
        text = response.content[0].text.strip()
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        scored_map = {s["id"]: s for s in json.loads(text)}
    except Exception:
        scored_map = {}

    enriched = []
    for a in raw:
        meta = scored_map.get(a["id"])
        if meta and meta.get("score", 0) >= 5:
            enriched.append({**a,
                "score":     meta["score"],
                "rationale": meta.get("rationale", ""),
                "theme":     meta.get("theme", "MACRO"),
            })
        elif not scored_map:
            enriched.append({**a, "score": 5, "rationale": "", "theme": "MACRO"})

    return {"articles": sorted(enriched, key=lambda a: a["score"], reverse=True),
            "tickers":  tickers_out}


# ── CSS ───────────────────────────────────────────────────────────────────────

PAGE_CSS = """
<style>
/* ── Marquee ── */
.vlt-marquee-wrap {
    overflow: hidden; white-space: nowrap;
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;
    padding: 8px 0; margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.vlt-marquee-inner { display: inline-block; animation: vlt-scroll 45s linear infinite; }
@keyframes vlt-scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }
.vlt-marquee-item { display: inline-block; margin-right: 28px; font-size: 11px;
                    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 500; }
.vlt-tick-label { color: #9ca3af; margin-right: 3px; }
.vlt-tick-val   { color: #111827; font-weight: 600; margin-right: 3px; }
.vlt-tick-up    { color: #16a34a; font-weight: 600; }
.vlt-tick-dn    { color: #dc2626; font-weight: 600; }
.vlt-tick-sep   { color: #d1d5db; margin: 0 14px; }

/* ── Stats row — scrollable on mobile ── */
.vlt-stats {
    display: flex; background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 8px; overflow-x: auto; margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    -webkit-overflow-scrolling: touch;
}
.vlt-stat-cell { flex: 0 0 auto; padding: 10px 16px; border-right: 1px solid #e5e7eb; min-width: 80px; }
.vlt-stat-cell:last-child { border-right: none; }
.vlt-stat-label { font-size: 9px; font-weight: 600; color: #9ca3af;
                  letter-spacing: 0.1em; text-transform: uppercase;
                  margin-bottom: 3px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
.vlt-stat-val { font-size: 14px; font-weight: 600; color: #111827;
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }

/* ── Card standard ── */
.vlt-card {
    background: #ffffff; border: 1px solid #e5e7eb; border-left: 3px solid #e5e7eb;
    border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.15s;
}
.vlt-card:hover { box-shadow: 0 2px 8px rgba(37,99,235,0.08); }

/* ── Card critique score 9-10 ── */
.vlt-card-critical {
    background: #fffbeb;
    border: 1.5px solid #fbbf24;
    border-left: 5px solid #dc2626;
    border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;
    box-shadow: 0 2px 12px rgba(220,38,38,0.10);
    transition: box-shadow 0.15s;
}
.vlt-card-critical:hover { box-shadow: 0 4px 18px rgba(220,38,38,0.18); }

.vlt-critical-banner {
    font-size: 9px; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: #dc2626;
    margin-bottom: 8px;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    display: flex; align-items: center; gap: 5px;
}

/* ── Card meta — wraps cleanly on mobile ── */
.vlt-card-meta {
    display: flex; align-items: flex-start; gap: 6px;
    justify-content: space-between; margin-bottom: 6px; flex-wrap: wrap;
}
.vlt-card-left { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }

.vlt-card-source {
    font-size: 9px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #2563eb; background: #eff6ff;
    padding: 2px 7px; border-radius: 4px; white-space: nowrap;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
.vlt-theme-badge {
    font-size: 9px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 2px 7px; border-radius: 4px;
    white-space: nowrap;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
.vlt-score {
    font-size: 10px; font-weight: 700; padding: 2px 6px;
    border-radius: 4px; background: #f3f4f6; white-space: nowrap;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
.vlt-card-age {
    font-size: 10px; color: #9ca3af; white-space: nowrap; margin-top: 2px;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

/* ── Title ── */
.vlt-card-title {
    font-size: 13px; font-weight: 600; color: #111827;
    line-height: 1.5; margin-bottom: 4px;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
.vlt-card-critical .vlt-card-title { font-size: 14px; }
.vlt-card-title a { color: #111827; text-decoration: none; }
.vlt-card-title a:hover { color: #2563eb; }

.vlt-card-rationale {
    font-size: 11px; color: #6b7280; line-height: 1.5;
    font-style: italic; margin-top: 5px;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
.vlt-card-critical .vlt-card-rationale { color: #92400e; }

/* ── Mobile ── */
@media (max-width: 640px) {
    .vlt-marquee-item { font-size: 10px; margin-right: 18px; }
    .vlt-card, .vlt-card-critical { padding: 11px 12px; }
    .vlt-card-title { font-size: 13px; }
    .vlt-card-critical .vlt-card-title { font-size: 13px; }
    .vlt-stat-val { font-size: 13px; }
    .vlt-stat-cell { padding: 9px 12px; }
}
</style>
"""

def marquee_html(tickers: list[dict]) -> str:
    if not tickers:
        return ""
    def item(t):
        cls   = "vlt-tick-up" if t["up"] else "vlt-tick-dn"
        arrow = "▲" if t["up"] else "▼"
        return (f'<span class="vlt-marquee-item">'
                f'<span class="vlt-tick-label">{t["label"]}</span>'
                f'<span class="vlt-tick-val">{t["value"]}</span>'
                f'<span class="{cls}">{arrow} {t["change"]}</span>'
                f'</span><span class="vlt-tick-sep">|</span>')
    inner = "".join(item(t) for t in tickers * 2)
    return f'<div class="vlt-marquee-wrap"><div class="vlt-marquee-inner">{inner}</div></div>'


def card_html(a: dict) -> str:
    bg, fg   = theme_color(a["theme"])
    sc       = a["score"]
    sc_col   = score_color(sc)
    critical = sc >= 9
    rationale = (f'<div class="vlt-card-rationale">↳ {a["rationale"]}</div>'
                 if a["rationale"] else "")

    if critical:
        return f"""
        <div class="vlt-card-critical">
          <div class="vlt-critical-banner">⚡ TOP NEWS &nbsp;·&nbsp; {sc}/10</div>
          <div class="vlt-card-meta">
            <div class="vlt-card-left">
              <span class="vlt-card-source">{a['source']}</span>
              <span class="vlt-theme-badge" style="background:{bg};color:{fg}">{a['theme']}</span>
            </div>
            <span class="vlt-card-age">{age_label(a['age_min'])} · {a['date'].strftime('%H:%M')}</span>
          </div>
          <div class="vlt-card-title"><a href="{a['link']}" target="_blank">{a['title']}</a></div>
          {rationale}
        </div>"""

    return f"""
    <div class="vlt-card" style="border-left-color:{sc_col}">
      <div class="vlt-card-meta">
        <div class="vlt-card-left">
          <span class="vlt-card-source">{a['source']}</span>
          <span class="vlt-theme-badge" style="background:{bg};color:{fg}">{a['theme']}</span>
          <span class="vlt-score" style="color:{sc_col}">{sc}/10</span>
        </div>
        <span class="vlt-card-age">{age_label(a['age_min'])} · {a['date'].strftime('%H:%M')}</span>
      </div>
      <div class="vlt-card-title"><a href="{a['link']}" target="_blank">{a['title']}</a></div>
      {rationale}
    </div>"""


# ── Render ────────────────────────────────────────────────────────────────────

def render():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.title("NEWS FEED")

    with st.sidebar:
        if st.button("↺ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    now_mtl      = datetime.now(MONTREAL_TZ)
    market_hours = 8 <= now_mtl.hour < 17

    # Auto-refresh uniquement entre 8h et 17h HNE.
    # Hors heures : ttl_bucket figé à la dernière valeur de 17h → cache stable,
    # aucun appel RSS/Haiku jusqu'au prochain refresh manuel.
    if market_hours:
        ttl_bucket = int(now_mtl.timestamp() // (REFRESH_MINUTES * 60))
    else:
        # Bucket figé = dernier bucket de la journée (17h00)
        close = now_mtl.replace(hour=17, minute=0, second=0, microsecond=0)
        ttl_bucket = int(close.timestamp() // (REFRESH_MINUTES * 60))

    with st.spinner("Chargement…"):
        result       = fetch_and_score_all(ttl_bucket)
        all_enriched = result["articles"]
        tickers      = result["tickers"]

    if not all_enriched:
        st.info("Aucun article récupéré. Vérifie ta connexion.")
        return

    enriched = all_enriched[:TOP_N_DISPLAY]

    st.markdown(marquee_html(tickers), unsafe_allow_html=True)

    # ── Stats bar ─────────────────────────────────────────────────────────
    theme_counts = {}
    for a in enriched:
        theme_counts[a["theme"]] = theme_counts.get(a["theme"], 0) + 1
    top_theme    = max(theme_counts, key=theme_counts.get) if theme_counts else "—"
    avg_score    = round(sum(a["score"] for a in enriched) / len(enriched), 1) if enriched else 0
    n_critical   = sum(1 for a in enriched if a["score"] >= 9)
    next_ref_min = REFRESH_MINUTES - (int(now_mtl.timestamp()) % (REFRESH_MINUTES * 60)) // 60
    refresh_label = f"~{next_ref_min}m" if market_hours else "Manuel"

    st.markdown(f"""
    <div class="vlt-stats">
      <div class="vlt-stat-cell">
        <div class="vlt-stat-label">Articles</div>
        <div class="vlt-stat-val">{len(enriched)}</div>
      </div>
      <div class="vlt-stat-cell">
        <div class="vlt-stat-label">⚡ Top news</div>
        <div class="vlt-stat-val" style="color:#dc2626">{n_critical}</div>
      </div>
      <div class="vlt-stat-cell">
        <div class="vlt-stat-label">Score moy.</div>
        <div class="vlt-stat-val" style="color:{score_color(int(avg_score))}">{avg_score}</div>
      </div>
      <div class="vlt-stat-cell">
        <div class="vlt-stat-label">Thème</div>
        <div class="vlt-stat-val" style="font-size:12px">{top_theme}</div>
      </div>
      <div class="vlt-stat-cell">
        <div class="vlt-stat-label">Refresh</div>
        <div class="vlt-stat-val">{refresh_label}</div>
      </div>
      <div class="vlt-stat-cell">
        <div class="vlt-stat-label">HNE</div>
        <div class="vlt-stat-val">{now_mtl.strftime("%H:%M")}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    for a in enriched:
        st.markdown(card_html(a), unsafe_allow_html=True)

    if market_hours:
        st.markdown(
            f'<meta http-equiv="refresh" content="{REFRESH_MINUTES * 60}">',
            unsafe_allow_html=True,
        )
    st.caption(
        f"Filtre : claude-haiku-4-5 · Refresh : {REFRESH_MINUTES}min · "
        f"MAJ : {now_mtl.strftime('%H:%M:%S')} HNE"
    )


render()