"""실시간 모니터링 대시보드 (Streamlit).

실행:
  streamlit run dashboard/app.py

접속:
  http://localhost:8501
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from tracking.trade_log import TradeLogger
from live.sector_instruments import SECTOR_ORDER


# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="Alpha Signal Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 데이터 로드 ──────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_performance(days: int = 90):
    logger = TradeLogger()
    return logger.get_performance_df(days=days)

@st.cache_data(ttl=60)
def load_trades(days: int = 30):
    logger = TradeLogger()
    return logger.get_trades_df(days=days)

@st.cache_data(ttl=300)
def load_summary():
    logger = TradeLogger()
    return logger.get_summary()

@st.cache_data(ttl=300)
def load_signal_accuracy(days: int = 30):
    logger = TradeLogger()
    return logger.get_signal_accuracy(days=days)


# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정")
    period = st.selectbox("조회 기간", [30, 60, 90, 180], index=1)
    auto_refresh = st.checkbox("자동 새로고침 (60초)", value=True)
    st.divider()
    st.markdown("**실행 명령어**")
    st.code("python scheduler/daily_runner.py --step signal", language="bash")
    st.code("python scheduler/daily_runner.py --step order", language="bash")
    st.code("python scheduler/retrain_runner.py --mode weekly", language="bash")
    st.divider()
    st.caption(f"마지막 갱신: {datetime.now():%H:%M:%S}")

if auto_refresh:
    st.empty()  # 자동 새로고침 트리거


# ── 헤더 ─────────────────────────────────────────────────────
st.title("📈 Alpha Signal Discovery Engine")
st.caption("한국투자증권 연동 | 실시간 모니터링 대시보드")


# ── 요약 지표 카드 ────────────────────────────────────────────
summary = load_summary()
perf_df = load_performance(days=period)

col1, col2, col3, col4, col5 = st.columns(5)

if perf_df.empty:
    st.info("운용 데이터가 없습니다. 먼저 daily_runner를 실행하세요.")
else:
    latest = perf_df.iloc[-1]
    total_return = (
        perf_df["portfolio_value"].iloc[-1] /
        perf_df["portfolio_value"].iloc[0] - 1
        if len(perf_df) > 1 else 0
    )

    with col1:
        st.metric(
            "포트폴리오",
            f"{latest['portfolio_value']:,.0f}원",
            f"{latest['daily_return']:+.2%} (오늘)",
        )
    with col2:
        st.metric(
            f"누적 수익 ({period}일)",
            f"{total_return:+.2%}",
            delta_color="normal",
        )
    with col3:
        sharpe = latest.get("sharpe_30d", 0) or 0
        st.metric("30일 Sharpe", f"{sharpe:.2f}")
    with col4:
        mdd = perf_df["mdd_cumul"].min()
        st.metric("최대낙폭", f"{mdd:.2%}", delta_color="inverse")
    with col5:
        win_rate = (perf_df["daily_return"] > 0).mean()
        st.metric("승률", f"{win_rate:.1%}")


st.divider()


# ── 수익 곡선 ─────────────────────────────────────────────────
if not perf_df.empty:
    st.subheader("📊 수익 곡선")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf_df["date"],
        y=perf_df["portfolio_value"],
        name="포트폴리오",
        line=dict(color="#00D4FF", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.1)",
    ))

    if "benchmark_return" in perf_df.columns and perf_df["benchmark_return"].notna().any():
        initial = perf_df["portfolio_value"].iloc[0]
        bench_cumul = (1 + perf_df["benchmark_return"].fillna(0)).cumprod() * initial
        fig.add_trace(go.Scatter(
            x=perf_df["date"],
            y=bench_cumul,
            name="Equal-weight 기준선",
            line=dict(color="#FF6B6B", width=1.5, dash="dash"),
        ))

    fig.update_layout(
        height=400,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.02),
        yaxis_title="포트폴리오 가치 (원)",
        xaxis_title="날짜",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 섹터 배분 + 신호 정확도 ───────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🎯 현재 섹터 배분")
    # 가장 최근 신호 로드
    try:
        logger = TradeLogger()
        with logger._conn() as conn:
            import sqlite3
            latest_signals = pd.read_sql(
                """SELECT sector, weight FROM model_signals
                   WHERE date = (SELECT MAX(date) FROM model_signals)""",
                conn,
            )

        if not latest_signals.empty:
            fig_pie = px.pie(
                latest_signals,
                names="sector",
                values="weight",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_pie.update_layout(
                height=350,
                template="plotly_dark",
                margin=dict(l=0, r=0, t=20, b=0),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("신호 데이터 없음")
    except Exception:
        st.info("신호 데이터 로딩 중...")

with col_right:
    st.subheader("🎯 섹터별 신호 정확도")
    accuracy = load_signal_accuracy(days=30)
    dir_acc  = accuracy.get("directional_accuracy")

    if dir_acc is not None:
        # 게이지 차트
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=dir_acc * 100,
            title={"text": "방향 예측 정확도 (30일)"},
            delta={"reference": 50, "suffix": "%"},
            gauge={
                "axis": {"range": [40, 70]},
                "bar": {"color": "#00D4FF"},
                "steps": [
                    {"range": [40, 50], "color": "#FF4444"},
                    {"range": [50, 55], "color": "#FFA500"},
                    {"range": [55, 70], "color": "#00FF88"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "value": 52,
                },
            },
            number={"suffix": "%", "font": {"size": 30}},
        ))
        fig_gauge.update_layout(
            height=300,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(f"샘플 수: {accuracy.get('n_samples', 0)}개")
    else:
        st.info("정확도 데이터 축적 중...")


st.divider()


# ── 일별 수익률 히트맵 ────────────────────────────────────────
if not perf_df.empty and len(perf_df) >= 5:
    st.subheader("📅 일별 수익률")

    daily_ret_pct = perf_df["daily_return"] * 100

    fig_bar = go.Figure(go.Bar(
        x=perf_df["date"],
        y=daily_ret_pct,
        marker_color=["#00FF88" if r > 0 else "#FF4444" for r in daily_ret_pct],
        name="일수익률 (%)",
    ))
    fig_bar.update_layout(
        height=250,
        template="plotly_dark",
        yaxis_title="수익률 (%)",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── 최근 거래 내역 ────────────────────────────────────────────
st.subheader("📋 최근 거래 내역")
trades_df = load_trades(days=30)

if not trades_df.empty:
    display_df = trades_df[
        ["timestamp", "ticker", "side", "qty", "price", "amount", "sector"]
    ].head(20)
    display_df["side"] = display_df["side"].map({"buy": "🟢 매수", "sell": "🔴 매도"})
    display_df["amount"] = display_df["amount"].apply(lambda x: f"{x:,.0f}원")
    display_df.columns = ["시간", "티커", "구분", "수량", "가격", "금액", "섹터"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("거래 내역이 없습니다.")


# ── 재학습 이력 ───────────────────────────────────────────────
st.subheader("🔄 재학습 이력")
try:
    logger = TradeLogger()
    with logger._conn() as conn:
        retrain_df = pd.read_sql(
            "SELECT timestamp, trigger, duration_sec, dir_acc, val_loss "
            "FROM retrain_log ORDER BY timestamp DESC LIMIT 10",
            conn,
        )

    if not retrain_df.empty:
        retrain_df["소요시간"] = retrain_df["duration_sec"].apply(
            lambda s: f"{s//60}분 {s%60}초" if s else "N/A"
        )
        retrain_df["정확도"] = retrain_df["dir_acc"].apply(
            lambda x: f"{x:.1%}" if x else "N/A"
        )
        retrain_df = retrain_df.rename(columns={
            "timestamp": "일시", "trigger": "유형",
            "val_loss": "Val Loss",
        })
        st.dataframe(
            retrain_df[["일시", "유형", "소요시간", "정확도", "Val Loss"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("재학습 이력 없음")
except Exception:
    st.info("재학습 이력 로딩 중...")


# ── 자동 새로고침 ─────────────────────────────────────────────
if auto_refresh:
    import time
    time.sleep(60)
    st.rerun()
