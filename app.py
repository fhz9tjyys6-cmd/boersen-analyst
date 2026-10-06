import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
from datetime import datetime
from urllib.parse import quote_plus

st.set_page_config(
    page_title="Börsen Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Börsen Analyst")
st.caption("Analyse für das Börsen-Planspiel")

# --------------------------------------------------
# EINSTELLUNGEN
# --------------------------------------------------

st.sidebar.header("Einstellungen")

profile = st.sidebar.selectbox(
    "Risikoprofil",
    ["Ausgewogen", "Defensiv", "Chancenorientiert"]
)

ticker_input = st.sidebar.text_input(
    "Aktien-Ticker",
    "SAP.DE"
)

analyse_button = st.sidebar.button(
    "🔎 Aktie analysieren",
    type="primary"
)

# --------------------------------------------------
# HILFSFUNKTIONEN
# --------------------------------------------------

def get_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="2y", auto_adjust=True)

        if data.empty:
            return None

        return data

    except Exception:
        return None


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def analyse_technical(data):

    close = data["Close"]

    current = float(close.iloc[-1])

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])

    rsi = float(calculate_rsi(close).iloc[-1])

    return {
        "price": current,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi
    }


def calculate_performance(data):

    close = data["Close"]

    current = close.iloc[-1]

    result = {}

    if len(close) >= 22:
        result["1 Monat"] = (current / close.iloc[-22] - 1) * 100

    if len(close) >= 66:
        result["3 Monate"] = (current / close.iloc[-66] - 1) * 100

    if len(close) >= 132:
        result["6 Monate"] = (current / close.iloc[-132] - 1) * 100

    if len(close) >= 252:
        result["1 Jahr"] = (current / close.iloc[-252] - 1) * 100

    return result


def calculate_risk(data):

    returns = data["Close"].pct_change().dropna()

    volatility = returns.std() * np.sqrt(252) * 100

    peak = data["Close"].cummax()
    drawdown = (data["Close"] / peak - 1) * 100
    max_drawdown = drawdown.min()

    return volatility, max_drawdown


def monte_carlo(data, simulations=5000, days=30):

    close = data["Close"].dropna()

    log_returns = np.log(close / close.shift(1)).dropna()

    mean = log_returns.mean()
    std = log_returns.std()

    last_price = float(close.iloc[-1])

    random_returns = np.random.normal(
        mean,
        std,
        (days, simulations)
    )

    paths = last_price * np.exp(
        np.cumsum(random_returns, axis=0)
    )

    final_prices = paths[-1]

    probability_up = np.mean(final_prices > last_price) * 100
    probability_down = 100 - probability_up

    expected_return = (
        np.mean(final_prices) / last_price - 1
    ) * 100

    return probability_up, probability_down, expected_return


def get_news(ticker):

    query = quote_plus(
        f"{ticker} Aktie OR Unternehmen"
    )

    url = (
        "https://news.google.com/rss/search?"
        f"q={query}&hl=de&gl=DE&ceid=DE:de"
    )

    feed = feedparser.parse(url)

    news = []

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

    for entry in feed.entries[:20]:

        title = entry.get("title", "")
        link = entry.get("link", "")
        published = entry.get("published", "")

        source = ""

        if hasattr(entry, "source"):
            source = entry.source.get("title", "")

        news.append({
            "title": title,
            "source": source,
            "date": published,
            "link": link,
            "important": any(
                x.lower() in (title + source).lower()
                for x in important_sources
            )
        })

    return news


def recommendation(
    probability_up,
    rsi,
    price,
    sma20,
    sma50,
    sma200
):

    score = 50

    # Monte-Carlo
    score += (probability_up - 50) * 0.7

    # Trend
    if price > sma20:
        score += 5

    if price > sma50:
        score += 7

    if price > sma200:
        score += 10

    # RSI
    if rsi > 70:
        score -= 10

    elif rsi < 30:
        score += 8

    score = max(0, min(100, score))

    if score >= 72:
        decision = "🟢 KAUFEN"

    elif score >= 58:
        decision = "🟢 KLEIN KAUFEN"

    elif score >= 42:
        decision = "🟡 ABWARTEN"

    else:
        decision = "🔴 NICHT KAUFEN"

    return score, decision


# --------------------------------------------------
# START
# --------------------------------------------------

if analyse_button:

    ticker = ticker_input.strip().upper()

    with st.spinner("Analysiere Aktie..."):

        data = get_data(ticker)

    if data is None:

        st.error(
            "Keine Kursdaten gefunden. "
            "Überprüfe den Ticker."
        )

        st.stop()

    technical = analyse_technical(data)

    performance = calculate_performance(data)

    volatility, max_drawdown = calculate_risk(data)

    probability_up, probability_down, expected_return = (
        monte_carlo(data)
    )

    score, decision = recommendation(
        probability_up,
        technical["rsi"],
        technical["price"],
        technical["sma20"],
        technical["sma50"],
        technical["sma200"]
    )

    # --------------------------------------------------
    # ÜBERSICHT
    # --------------------------------------------------

    st.header(f"Analyse: {ticker}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Aktueller Kurs",
        f"{technical['price']:.2f}"
    )

    col2.metric(
        "Chance ↑",
        f"{probability_up:.1f}%"
    )

    col3.metric(
        "Risiko ↓",
        f"{probability_down:.1f}%"
    )

    col4.metric(
        "Erwartete 30-Tage-Rendite",
        f"{expected_return:+.1f}%"
    )

    st.divider()

    # --------------------------------------------------
    # EMPFEHLUNG
    # --------------------------------------------------

    st.subheader("🎯 Handlungsempfehlung")

    st.metric(
        "Modell-Score",
        f"{score:.0f}/100"
    )

    st.markdown(
        f"## {decision}"
    )

    if profile == "Defensiv":

        if volatility > 35:
            st.warning(
                "Für ein defensives Depot ist die "
                "aktuelle Volatilität relativ hoch."
            )

    elif profile == "Chancenorientiert":

        if probability_up > 60:
            st.success(
                "Das Modell sieht aktuell ein "
                "überdurchschnittliches Aufwärtspotenzial."
            )

    # --------------------------------------------------
    # TECHNISCHE ANALYSE
    # --------------------------------------------------

    st.subheader("📊 Technische Analyse")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "RSI",
        f"{technical['rsi']:.1f}"
    )

    c2.metric(
        "SMA 20",
        f"{technical['sma20']:.2f}"
    )

    c3.metric(
        "SMA 50",
        f"{technical['sma50']:.2f}"
    )

    c4.metric(
        "SMA 200",
        f"{technical['sma200']:.2f}"
    )

    if technical["rsi"] > 70:

        st.warning(
            "RSI über 70: Die Aktie könnte kurzfristig "
            "überkauft sein."
        )

    elif technical["rsi"] < 30:

        st.info(
            "RSI unter 30: Die Aktie könnte "
            "kurzfristig überverkauft sein."
        )

    # --------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------

    st.subheader("📈 Kursentwicklung")

    perf_cols = st.columns(len(performance))

    for column, (period, value) in zip(
        perf_cols,
        performance.items()
    ):

        column.metric(
            period,
            f"{value:+.1f}%"
        )

    # --------------------------------------------------
    # RISIKO
    # --------------------------------------------------

    st.subheader("⚠️ Risiko")

    c1, c2 = st.columns(2)

    c1.metric(
        "Jährliche Volatilität",
        f"{volatility:.1f}%"
    )

    c2.metric(
        "Max. historischer Drawdown",
        f"{max_drawdown:.1f}%"
    )

    # --------------------------------------------------
    # CHART
    # --------------------------------------------------

    st.subheader("📉 Kurschart")

    chart_data = data[["Close"]].copy()

    chart_data["SMA 20"] = (
        chart_data["Close"].rolling(20).mean()
    )

    chart_data["SMA 50"] = (
        chart_data["Close"].rolling(50).mean()
    )

    chart_data["SMA 200"] = (
        chart_data["Close"].rolling(200).mean()
    )

    st.line_chart(chart_data)

    # --------------------------------------------------
    # NEWS
    # --------------------------------------------------

    st.subheader("📰 Aktuelle Nachrichten")

    news = get_news(ticker)

    if not news:

        st.info("Keine Nachrichten gefunden.")

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

    # --------------------------------------------------
    # HINWEIS
    # --------------------------------------------------

    st.caption(
        "Wichtig: Die Prozentwerte sind Modellberechnungen "
        "auf Basis historischer Kursdaten und keine "
        "Garantie für zukünftige Kursbewegungen."
    )

else:

    st.info(
        "👈 Gib links einen Aktien-Ticker ein "
        "und klicke auf „Aktie analysieren“."
    )

    st.markdown("""
### Beispiele

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
""")
