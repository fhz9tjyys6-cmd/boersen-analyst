import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import math
from datetime import datetime
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO


# ============================================================
# KONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Börsen Analyst",
    page_icon="📈",
    layout="wide"
)

CAPITAL = 50_000

st.title("📈 Börsen Analyst")
st.caption(
    "Analyse- und Ranking-System für das Börsen-Planspiel · "
    "Modellbudget: 50.000 €"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Einstellungen")

profile = st.sidebar.selectbox(
    "Risikoprofil",
    [
        "Ausgewogen",
        "Defensiv",
        "Chancenorientiert"
    ]
)

max_position_percent = st.sidebar.slider(
    "Maximaler Anteil pro Aktie",
    min_value=5,
    max_value=30,
    value=20,
    step=5
)

reserve_percent = st.sidebar.slider(
    "Mindest-Cashreserve",
    min_value=0,
    max_value=40,
    value=10,
    step=5
)

max_stocks = st.sidebar.slider(
    "Maximale Anzahl Aktien im Depot",
    min_value=3,
    max_value=15,
    value=8
)

st.sidebar.divider()

st.sidebar.subheader("📂 Aktienuniversum")

uploaded_file = st.sidebar.file_uploader(
    "planspiel_universe.csv hochladen",
    type=["csv"]
)

manual_ticker = st.sidebar.text_input(
    "Oder einzelne Aktie analysieren",
    "SAP.DE"
)

st.sidebar.caption(
    "Die CSV sollte mindestens eine Spalte "
    "`ticker` enthalten."
)


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_stock_data(ticker, period="1y"):
    """
    Holt historische Kursdaten.
    Cache: 15 Minuten.
    """

    try:
        data = yf.Ticker(ticker).history(
            period=period,
            auto_adjust=True
        )

        if data is None or data.empty:
            return None

        data = data.dropna(subset=["Close"])

        if len(data) < 60:
            return None

        return data

    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def get_batch_data(tickers):
    """
    Lädt viele Aktien möglichst effizient gemeinsam.
    """

    if not tickers:
        return {}

    try:
        data = yf.download(
            tickers=list(tickers),
            period="1y",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True
        )

        result = {}

        # Mehrere Ticker -> MultiIndex
        if isinstance(data.columns, pd.MultiIndex):

            available = set(
                data.columns.get_level_values(0)
            )

            for ticker in tickers:

                if ticker not in available:
                    continue

                try:
                    ticker_data = data[ticker].copy()

                    if "Close" not in ticker_data.columns:
                        continue

                    ticker_data = ticker_data.dropna(
                        subset=["Close"]
                    )

                    if len(ticker_data) >= 60:
                        result[ticker] = ticker_data

                except Exception:
                    continue

        else:
            # Nur ein Ticker
            if "Close" in data.columns:
                data = data.dropna(
                    subset=["Close"]
                )

                if len(data) >= 60:
                    result[tickers[0]] = data

        return result

    except Exception:
        return {}


def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_indicators(data):

    close = data["Close"].dropna()

    if len(close) < 60:
        return None

    current = float(close.iloc[-1])

    sma20 = float(
        close.rolling(20).mean().iloc[-1]
    )

    sma50 = float(
        close.rolling(50).mean().iloc[-1]
    )

    sma200 = (
        float(close.rolling(200).mean().iloc[-1])
        if len(close) >= 200
        else np.nan
    )

    rsi_series = calculate_rsi(close)

    rsi = float(rsi_series.iloc[-1])

    # Momentum
    momentum_1m = (
        current / float(close.iloc[-22]) - 1
    ) * 100 if len(close) >= 22 else np.nan

    momentum_3m = (
        current / float(close.iloc[-66]) - 1
    ) * 100 if len(close) >= 66 else np.nan

    momentum_6m = (
        current / float(close.iloc[-132]) - 1
    ) * 100 if len(close) >= 132 else np.nan

    momentum_1y = (
        current / float(close.iloc[-252]) - 1
    ) * 100 if len(close) >= 252 else np.nan

    returns = close.pct_change().dropna()

    volatility = (
        float(returns.std()) *
        math.sqrt(252) *
        100
    )

    peak = close.cummax()

    drawdown = (
        close / peak - 1
    ) * 100

    max_drawdown = float(drawdown.min())

    # Abstand zu SMA
    distance_sma50 = (
        current / sma50 - 1
    ) * 100 if sma50 else 0

    if not np.isnan(sma200):
        distance_sma200 = (
            current / sma200 - 1
        ) * 100
    else:
        distance_sma200 = 0

    return {
        "price": current,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "momentum_1m": momentum_1m,
        "momentum_3m": momentum_3m,
        "momentum_6m": momentum_6m,
        "momentum_1y": momentum_1y,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "distance_sma50": distance_sma50,
        "distance_sma200": distance_sma200
    }


def estimated_probability(data, days=30):

    close = data["Close"].dropna()

    log_returns = np.log(
        close / close.shift(1)
    ).dropna()

    if len(log_returns) < 50:
        return 50.0, 50.0, 0.0

    mean = float(log_returns.mean())
    std = float(log_returns.std())

    if std <= 0:
        return 50.0, 50.0, 0.0

    # Modellannahme:
    # tägliche Logrenditen ~ Normalverteilung

    z = (
        math.sqrt(days) *
        mean /
        std
    )

    probability_up = (
        0.5 *
        (
            1 +
            math.erf(
                z / math.sqrt(2)
            )
        )
    ) * 100

    probability_up = max(
        1,
        min(99, probability_up)
    )

    probability_down = (
        100 - probability_up
    )

    expected_return = (
        math.exp(
            mean * days +
            0.5 * std**2 * days
        ) - 1
    ) * 100

    return (
        probability_up,
        probability_down,
        expected_return
    )


def calculate_score(
    indicators,
    probability_up,
    profile
):

    score = 50.0

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    price = indicators["price"]
    sma20 = indicators["sma20"]
    sma50 = indicators["sma50"]
    sma200 = indicators["sma200"]

    if price > sma20:
        score += 5

    if price > sma50:
        score += 7

    if not np.isnan(sma200):

        if price > sma200:
            score += 10
        else:
            score -= 7

    # Golden-Cross-artiger Trend
    if sma20 > sma50:
        score += 5

    if not np.isnan(sma200):

        if sma50 > sma200:
            score += 5

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    m1 = indicators["momentum_1m"]
    m3 = indicators["momentum_3m"]
    m6 = indicators["momentum_6m"]

    if not np.isnan(m1):

        if m1 > 0:
            score += 3
        elif m1 < -10:
            score -= 4

    if not np.isnan(m3):

        if m3 > 5:
            score += 5

        elif m3 < -10:
            score -= 5

    if not np.isnan(m6):

        if m6 > 10:
            score += 5

        elif m6 < -15:
            score -= 5

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = indicators["rsi"]

    if 45 <= rsi <= 65:
        score += 5

    elif 65 < rsi <= 70:
        score += 2

    elif rsi > 70:
        score -= 7

    elif 30 <= rsi < 40:
        score += 2

    elif rsi < 30:
        score += 5

    # --------------------------------------------------------
    # MONTE-CARLO / WAHRSCHEINLICHKEITSMODELL
    # --------------------------------------------------------

    score += (
        probability_up - 50
    ) * 0.45

    # --------------------------------------------------------
    # RISIKO
    # --------------------------------------------------------

    volatility = indicators["volatility"]

    if profile == "Defensiv":

        if volatility < 20:
            score += 7

        elif volatility > 40:
            score -= 12

        elif volatility > 30:
            score -= 7

    elif profile == "Chancenorientiert":

        if volatility > 20 and volatility < 55:
            score += 3

    else:

        if volatility > 60:
            score -= 8

    # Drawdown
    drawdown = indicators["max_drawdown"]

    if drawdown < -50:
        score -= 8

    elif drawdown < -35:
        score -= 5

    elif drawdown > -20:
        score += 2

    return float(
        max(
            0,
            min(
                100,
                score
            )
        )
    )


def decision_from_score(score):

    if score >= 75:
        return "🟢 STARK"

    if score >= 65:
        return "🟢 INTERESSANT"

    if score >= 55:
        return "🟡 BEOBACHTEN"

    if score >= 45:
        return "🟠 SCHWACH"

    return "🔴 MEIDEN"


def risk_label(volatility):

    if volatility < 20:
        return "Niedrig"

    if volatility < 30:
        return "Mittel"

    if volatility < 45:
        return "Erhöht"

    return "Hoch"


def reason_for_stock(row):

    reasons = []

    if row["Score"] >= 70:
        reasons.append("hoher Gesamtscore")

    if row["Preis > SMA50"]:
        reasons.append("positiver Trend")

    if row["Preis > SMA200"]:
        reasons.append("über langfristigem Trend")

    if row["RSI"] < 30:
        reasons.append("RSI sehr niedrig")

    elif 45 <= row["RSI"] <= 65:
        reasons.append("RSI im neutralen Bereich")

    if row["Momentum 3M"] > 5:
        reasons.append("positives Momentum")

    if row["Volatilität"] < 25:
        reasons.append("vergleichsweise geringes Risiko")

    if not reasons:
        reasons.append("gemischtes Signalbild")

    return ", ".join(reasons[:3])


# ============================================================
# CSV
# ============================================================

@st.cache_data
def read_universe(file_bytes):

    try:

        df = pd.read_csv(
            BytesIO(file_bytes)
        )

    except Exception:

        try:

            df = pd.read_csv(
                BytesIO(file_bytes),
                sep=";"
            )

        except Exception:

            return None

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    return df


def prepare_universe(df):

    if df is None:
        return None

    # mögliche Schreibweisen
    ticker_candidates = [
        "ticker",
        "symbol",
        "isin_ticker",
        "aktien_ticker"
    ]

    ticker_column = None

    for column in ticker_candidates:

        if column in df.columns:
            ticker_column = column
            break

    if ticker_column is None:
        return None

    result = df.copy()

    result["ticker"] = (
        result[ticker_column]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    result = result[
        (result["ticker"] != "") &
        (result["ticker"] != "NAN")
    ]

    result = result.drop_duplicates(
        subset=["ticker"]
    )

    return result


# ============================================================
# AUTOMATISCHER SCANNER
# ============================================================

def analyse_ticker(
    ticker,
    data,
    profile
):

    try:

        indicators = calculate_indicators(
            data
        )

        if indicators is None:
            return None

        (
            probability_up,
            probability_down,
            expected_return
        ) = estimated_probability(
            data
        )

        score = calculate_score(
            indicators,
            probability_up,
            profile
        )

        return {
            "Ticker": ticker,
            "Kurs": indicators["price"],
            "Score": round(score, 1),
            "Chance 30T": round(
                probability_up,
                1
            ),
            "Risiko 30T": round(
                probability_down,
                1
            ),
            "Erwartete Rendite": round(
                expected_return,
                1
            ),
            "RSI": round(
                indicators["rsi"],
                1
            ),
            "Momentum 3M": round(
                indicators["momentum_3m"],
                1
            ),
            "Volatilität": round(
                indicators["volatility"],
                1
            ),
            "Drawdown": round(
                indicators["max_drawdown"],
                1
            ),
            "Preis > SMA50": (
                indicators["price"] >
                indicators["sma50"]
            ),
            "Preis > SMA200": (
                False
                if np.isnan(
                    indicators["sma200"]
                )
                else
                indicators["price"] >
                indicators["sma200"]
            )
        }

    except Exception:
        return None


def scan_universe(
    universe,
    profile,
    limit
):

    tickers = (
        universe["ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .drop_duplicates()
        .tolist()
    )

    tickers = tickers[:limit]

    data_dict = get_batch_data(
        tuple(tickers)
    )

    results = []

    progress = st.progress(
        0,
        text="Analysiere Aktien..."
    )

    total = len(tickers)

    for index, ticker in enumerate(tickers):

        data = data_dict.get(ticker)

        if data is not None:

            result = analyse_ticker(
                ticker,
                data,
                profile
            )

            if result is not None:
                results.append(result)

        progress.progress(
            min(
                (index + 1) / max(total, 1),
                1.0
            ),
            text=f"{index + 1}/{total}: {ticker}"
        )

    progress.empty()

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        "Score",
        ascending=False
    ).reset_index(drop=True)

    result_df["Rang"] = (
        result_df.index + 1
    )

    result_df["Entscheidung"] = (
        result_df["Score"]
        .apply(decision_from_score)
    )

    result_df["Risiko"] = (
        result_df["Volatilität"]
        .apply(risk_label)
    )

    result_df["Grund"] = (
        result_df.apply(
            reason_for_stock,
            axis=1
        )
    )

    return result_df


# ============================================================
# DEPOT-ALLOKATION
# ============================================================

def create_allocation(
    ranking,
    capital=50_000,
    max_position_percent=20,
    reserve_percent=10,
    max_stocks=8
):

    if ranking.empty:
        return pd.DataFrame()

    eligible = ranking[
        ranking["Score"] >= 55
    ].copy()

    if eligible.empty:
        return pd.DataFrame()

    eligible = eligible.head(
        max_stocks
    ).copy()

    # Kapital, das tatsächlich investiert werden darf
    investable_capital = (
        capital *
        (1 - reserve_percent / 100)
    )

    max_position = (
        capital *
        max_position_percent /
        100
    )

    # Scores relativ zueinander
    weights = (
        eligible["Score"] -
        50
    ).clip(lower=1)

    eligible["Gewicht"] = (
        weights /
        weights.sum()
    )

    eligible["Zielbetrag"] = (
        eligible["Gewicht"] *
        investable_capital
    )

    eligible["Zielbetrag"] = (
        eligible["Zielbetrag"]
        .clip(
            upper=max_position
        )
    )

    # Nach dem Cap kann Kapital übrig bleiben.
    # Dieses wird nicht künstlich irgendwo hineingesteckt.
    eligible["Stück"] = (
        eligible["Zielbetrag"] /
        eligible["Kurs"]
    ).apply(
        lambda x: max(
            0,
            math.floor(x)
        )
    )

    eligible["Investition"] = (
        eligible["Stück"] *
        eligible["Kurs"]
    )

    eligible["Investition"] = (
        eligible["Investition"]
        .round(2)
    )

    eligible["Anteil"] = (
        eligible["Investition"] /
        capital *
        100
    ).round(1)

    eligible["Zielbetrag"] = (
        eligible["Zielbetrag"]
        .round(2)
    )

    total_invested = (
        eligible["Investition"]
        .sum()
    )

    cash = (
        capital -
        total_invested
    )

    eligible["Restbudget"] = round(
        cash,
        2
    )

    return eligible


# ============================================================
# NEWS
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_news(ticker):

    query = quote_plus(
        f'"{ticker}" Aktie OR Unternehmen'
    )

    url = (
        "https://news.google.com/rss/search?"
        f"q={query}&hl=de&gl=DE&ceid=DE:de"
    )

    try:
        feed = feedparser.parse(
            url
        )
    except Exception:
        return []

    important_sources = [
        "Handelsblatt",
        "Reuters",
        "Börsen-Zeitung",
        "FAZ",
        "Frankfurter Allgemeine",
        "Manager Magazin",
        "Tagesschau",
        "WirtschaftsWoche"
    ]

    news = []

    for entry in feed.entries[:20]:

        title = entry.get(
            "title",
            ""
        )

        link = entry.get(
            "link",
            ""
        )

        published = entry.get(
            "published",
            ""
        )

        source = ""

        if hasattr(
            entry,
            "source"
        ):

            source = entry.source.get(
                "title",
                ""
            )

        combined = (
            title + " " + source
        ).lower()

        important = any(
            source_name.lower()
            in combined
            for source_name
            in important_sources
        )

        news.append({
            "title": title,
            "source": source,
            "date": published,
            "link": link,
            "important": important
        })

    return news


# ============================================================
# EINZELANALYSE
# ============================================================

def show_single_analysis(
    ticker,
    profile
):

    ticker = ticker.strip().upper()

    if not ticker:
        return

    with st.spinner(
        f"Analysiere {ticker}..."
    ):

        data = get_stock_data(
            ticker,
            "2y"
        )

    if data is None:

        st.error(
            f"Für {ticker} konnten keine "
            "ausreichenden Kursdaten geladen werden."
        )

        return

    indicators = calculate_indicators(
        data
    )

    (
        probability_up,
        probability_down,
        expected_return
    ) = estimated_probability(
        data
    )

    score = calculate_score(
        indicators,
        probability_up,
        profile
    )

    decision = decision_from_score(
        score
    )

    # --------------------------------------------------------
    # ÜBERSICHT
    # --------------------------------------------------------

    st.header(
        f"Analyse: {ticker}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Aktueller Kurs",
        f"{indicators['price']:.2f}"
    )

    c2.metric(
        "Modell-Chance ↑",
        f"{probability_up:.1f}%"
    )

    c3.metric(
        "Modell-Risiko ↓",
        f"{probability_down:.1f}%"
    )

    c4.metric(
        "Modell-Rendite 30T",
        f"{expected_return:+.1f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # EMPFEHLUNG
    # --------------------------------------------------------

    st.subheader(
        "🎯 Modellbewertung"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Score",
        f"{score:.0f}/100"
    )

    c2.metric(
        "Bewertung",
        decision
    )

    c3.metric(
        "Risiko",
        risk_label(
            indicators["volatility"]
        )
    )

    # --------------------------------------------------------
    # TECHNIK
    # --------------------------------------------------------

    st.subheader(
        "📊 Technische Analyse"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "RSI",
        f"{indicators['rsi']:.1f}"
    )

    c2.metric(
        "SMA 20",
        f"{indicators['sma20']:.2f}"
    )

    c3.metric(
        "SMA 50",
        f"{indicators['sma50']:.2f}"
    )

    c4.metric(
        "SMA 200",
        (
            "–"
            if np.isnan(
                indicators["sma200"]
            )
            else
            f"{indicators['sma200']:.2f}"
        )
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "📈 Performance"
    )

    performance = {}

    close = data["Close"]

    if len(close) >= 22:
        performance["1 Monat"] = (
            close.iloc[-1] /
            close.iloc[-22] - 1
        ) * 100

    if len(close) >= 66:
        performance["3 Monate"] = (
            close.iloc[-1] /
            close.iloc[-66] - 1
        ) * 100

    if len(close) >= 132:
        performance["6 Monate"] = (
            close.iloc[-1] /
            close.iloc[-132] - 1
        ) * 100

    if len(close) >= 252:
        performance["1 Jahr"] = (
            close.iloc[-1] /
            close.iloc[-252] - 1
        ) * 100

    cols = st.columns(
        len(performance)
    )

    for col, (
        period,
        value
    ) in zip(
        cols,
        performance.items()
    ):

        col.metric(
            period,
            f"{value:+.1f}%"
        )

    # --------------------------------------------------------
    # RISIKO
    # --------------------------------------------------------

    st.subheader(
        "⚠️ Risiko"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Volatilität p.a.",
        f"{indicators['volatility']:.1f}%"
    )

    c2.metric(
        "Max. historischer Drawdown",
        f"{indicators['max_drawdown']:.1f}%"
    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.subheader(
        "📉 Kurschart"
    )

    chart = data[
        ["Close"]
    ].copy()

    chart["SMA 20"] = (
        chart["Close"]
        .rolling(20)
        .mean()
    )

    chart["SMA 50"] = (
        chart["Close"]
        .rolling(50)
        .mean()
    )

    chart["SMA 200"] = (
        chart["Close"]
        .rolling(200)
        .mean()
    )

    st.line_chart(
        chart
    )

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    st.subheader(
        "📰 Aktuelle Nachrichten"
    )

    news = get_news(
        ticker
    )

    if not news:

        st.info(
            "Keine Nachrichten gefunden."
        )

    else:

        for article in news[:10]:

            if article["important"]:

                st.markdown(
                    f"### ⭐ {article['title']}"
                )

            else:

                st.markdown(
                    f"### {article['title']}"
                )

            if article["source"]:

                st.caption(
                    f"{article['source']} · "
                    f"{article['date']}"
                )

            if article["link"]:

                st.markdown(
                    f"[Artikel öffnen]({article['link']})"
                )

            st.divider()

    st.caption(
        "Die Aufwärts-/Abwärtswerte sind Modellschätzungen "
        "aus historischen Renditen. Sie sind keine "
        "verlässlich kalibrierten Wahrscheinlichkeiten."
    )


# ============================================================
# HAUPTAPP
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🤖 Automatischer Scanner",
        "🔎 Einzelanalyse",
        "ℹ️ Über das Modell"
    ]
)


# ============================================================
# TAB 1 — SCANNER
# ============================================================

with tab1:

    st.header(
        "🤖 Automatischer Aktien-Scanner"
    )

    st.write(
        f"Das Modell analysiert das Aktienuniversum "
        f"und versucht daraus eine sinnvolle Verteilung "
        f"des Modellkapitals von **{CAPITAL:,.0f} €** abzuleiten."
        .replace(",", ".")
    )

    if uploaded_file is None:

        st.warning(
            "Bitte links eure `planspiel_universe.csv` "
            "hochladen."
        )

        st.info(
            "Danach kann das Programm die Aktien aus "
            "eurem Planspiel automatisch bewerten."
        )

    else:

        file_bytes = uploaded_file.getvalue()

        universe = read_universe(
            file_bytes
        )

        universe = prepare_universe(
            universe
        )

        if universe is None:

            st.error(
                "Die CSV konnte nicht erkannt werden. "
                "Sie benötigt mindestens eine Spalte "
                "`ticker`."
            )

        else:

            st.success(
                f"{len(universe)} Aktien im Universum erkannt."
            )

            st.caption(
                "Für einen ersten Scan werden standardmäßig "
                "bis zu 150 Ticker untersucht. Dadurch bleibt "
                "die Anwendung trotz vieler Aktien praktikabel."
            )

            scan_limit = st.slider(
                "Anzahl zu analysierender Aktien",
                min_value=20,
                max_value=min(
                    300,
                    len(universe)
                ),
                value=min(
                    150,
                    len(universe)
                ),
                step=10
            )

            scan_button = st.button(
                "🚀 Aktienuniversum analysieren",
                type="primary",
                use_container_width=True
            )

            if scan_button:

                ranking = scan_universe(
                    universe,
                    profile,
                    scan_limit
                )

                if ranking.empty:

                    st.error(
                        "Es konnten keine Aktien "
                        "ausgewertet werden."
                    )

                else:

                    st.session_state[
                        "ranking"
                    ] = ranking

                    st.session_state[
                        "universe"
                    ] = universe

                    st.success(
                        f"{len(ranking)} Aktien erfolgreich analysiert."
                    )

            if "ranking" in st.session_state:

                ranking = st.session_state[
                    "ranking"
                ]

                st.divider()

                # ------------------------------------------------
                # TOP AKTIEN
                # ------------------------------------------------

                st.subheader(
                    "🏆 Aktuelles Ranking"
                )

                display_columns = [
                    "Rang",
                    "Ticker",
                    "Score",
                    "Entscheidung",
                    "Chance 30T",
                    "Erwartete Rendite",
                    "RSI",
                    "Momentum 3M",
                    "Volatilität",
                    "Risiko",
                    "Grund"
                ]

                st.dataframe(
                    ranking[
                        display_columns
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # TOP 5
                # ------------------------------------------------

                st.subheader(
                    "🥇 Top-Kandidaten"
                )

                top5 = ranking.head(5)

                cols = st.columns(
                    len(top5)
                )

                for col, (_, row) in zip(
                    cols,
                    top5.iterrows()
                ):

                    with col:

                        st.metric(
                            f"#{int(row['Rang'])} "
                            f"{row['Ticker']}",
                            f"{row['Score']:.0f}/100"
                        )

                        st.caption(
                            row["Entscheidung"]
                        )

                        st.write(
                            f"Chance: "
                            f"**{row['Chance 30T']:.1f}%**"
                        )

                        st.write(
                            f"Risiko: "
                            f"**{row['Risiko']}**"
                        )

                # ------------------------------------------------
                # ALLOKATION
                # ------------------------------------------------

                st.divider()

                st.header(
                    "💰 Automatische 50.000-€-Aufteilung"
                )

                allocation = create_allocation(
                    ranking,
                    CAPITAL,
                    max_position_percent,
                    reserve_percent,
                    max_stocks
                )

                if allocation.empty:

                    st.warning(
                        "Das Modell findet momentan "
                        "keine ausreichend starken "
                        "Kandidaten für eine automatische "
                        "Aufteilung."
                    )

                else:

                    total_invested = (
                        allocation[
                            "Investition"
                        ].sum()
                    )

                    cash = (
                        CAPITAL -
                        total_invested
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Investiert",
                        f"{total_invested:,.0f} €"
                        .replace(",", ".")
                    )

                    c2.metric(
                        "Cash",
                        f"{cash:,.0f} €"
                        .replace(",", ".")
                    )

                    c3.metric(
                        "Anzahl Positionen",
                        str(len(allocation))
                    )

                    allocation_columns = [
                        "Ticker",
                        "Score",
                        "Kurs",
                        "Stück",
                        "Investition",
                        "Anteil",
                        "Chance 30T",
                        "Volatilität",
                        "Risiko",
                        "Grund"
                    ]

                    st.dataframe(
                        allocation[
                            allocation_columns
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader(
                        "📌 Vorgeschlagene Positionen"
                    )

                    for _, row in allocation.iterrows():

                        st.write(
                            f"**{row['Ticker']}** — "
                            f"{int(row['Stück'])} Stück · "
                            f"{row['Investition']:,.0f} € · "
                            f"Score {row['Score']:.0f}"
                            .replace(",", ".")
                        )

                    st.caption(
                        "Die Aufteilung ist eine regelbasierte "
                        "Simulation für das Schul-Börsenplanspiel. "
                        "Sie ist keine Empfehlung für echtes Geld."
                    )

                    # ------------------------------------------------
                    # DOWNLOAD
                    # ------------------------------------------------

                    csv = ranking.to_csv(
                        index=False
                    ).encode(
                        "utf-8-sig"
                    )

                    st.download_button(
                        "⬇️ Ranking als CSV herunterladen",
                        data=csv,
                        file_name=(
                            "boersen_ranking.csv"
                        ),
                        mime="text/csv"
                    )


# ============================================================
# TAB 2 — EINZELANALYSE
# ============================================================

with tab2:

    st.header(
        "🔎 Einzelanalyse"
    )

    ticker = st.text_input(
        "Aktien-Ticker",
        value=manual_ticker,
        key="single_ticker"
    )

    analyse_button = st.button(
        "🔎 Aktie analysieren",
        type="primary"
    )

    if analyse_button:

        show_single_analysis(
            ticker,
            profile
        )

    else:

        st.info(
            "Ticker eingeben und Analyse starten."
        )

        st.markdown(
            """
**Beispiele**

- SAP → `SAP.DE`
- Siemens → `SIE.DE`
- Allianz → `ALV.DE`
- Adidas → `ADS.DE`
- BMW → `BMW.DE`
- Mercedes-Benz → `MBG.DE`
- Deutsche Bank → `DBK.DE`
- Nvidia → `NVDA`
- Apple → `AAPL`
- Microsoft → `MSFT`
"""
        )


# ============================================================
# TAB 3 — MODELL
# ============================================================

with tab3:

    st.header(
        "ℹ️ Wie funktioniert das Modell?"
    )

    st.markdown(
        """
### 1. Trend

Das Modell untersucht unter anderem:

- SMA 20
- SMA 50
- SMA 200
- Verhältnis der gleitenden Durchschnitte

Damit wird geprüft, ob sich die Aktie eher in einem
positiven oder negativen Trend befindet.

### 2. Momentum

Berücksichtigt werden unter anderem:

- 1 Monat
- 3 Monate
- 6 Monate
- 1 Jahr

Stark positives Momentum erhöht den Score,
stark negatives Momentum senkt ihn.

### 3. RSI

Der RSI wird genutzt, um ungewöhnlich starke
kurzfristige Bewegungen zu erkennen.

Ein sehr hoher RSI wird nicht automatisch als gut bewertet,
weil die Aktie kurzfristig überhitzt sein könnte.

### 4. Risiko

Berücksichtigt werden:

- historische Volatilität
- maximaler Drawdown

Das Risikoprofil verändert die Gewichtung.

### 5. Aufwärtswahrscheinlichkeit

Die angezeigte Wahrscheinlichkeit basiert auf
historischen Renditen und einem vereinfachten
statistischen Modell.

**Sie ist ausdrücklich keine echte, kalibrierte
Wahrscheinlichkeit für die Zukunft.**

### 6. Gesamt-Score

Aus diesen Faktoren entsteht ein Score von:

**0 = sehr schwach**

bis

**100 = sehr stark**

### 7. Depotaufteilung

Das Modell verwendet:

**50.000 €**

und berücksichtigt dabei:

- maximalen Anteil pro Aktie
- Cashreserve
- maximale Anzahl Positionen
- Score
- Risikoprofil

Es versucht also nicht einfach, die komplette Summe
in die Aktie mit dem höchsten Score zu stecken.

### Wichtig

Das Ganze ist für euer **Schul-Börsenplanspiel** gedacht.
Historische Daten können zukünftige Kursbewegungen
nicht zuverlässig vorhersagen.
"""
    )

st.divider()

st.caption(
    f"Letzte Berechnung: "
    f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
)
