"""대체 데이터 피처 (옵션 IV 프록시 + 공시 NLP 감성).

직접 옵션 데이터 접근이 없으므로 OHLCV 기반 프록시를 사용:
- Implied Volatility proxy: Parkinson / Garman-Klass 역사적 변동성
- Put-Call ratio proxy: 하락 변동성 비대칭 (downside vol / upside vol)
- NLP 감성: DART 공시 제목 키워드 기반 간이 감성 분석
"""

import numpy as np
import pandas as pd
from loguru import logger


class AlternativeDataFeatures:
    """대체 데이터 기반 피처 생성기.

    기존 FeatureEngineer와 별도로, 고급 통계적 피처를 추가 생성.
    주로 옵션 시장 정보의 프록시 역할.
    """

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 대체 데이터 피처 계산.

        Args:
            df: OHLCV DataFrame (date, open, high, low, close, volume, ticker)

        Returns:
            피처가 추가된 DataFrame
        """
        logger.info("Computing alternative data features...")
        df = df.sort_values(["ticker", "date"]).copy()
        groups = df.groupby("ticker", group_keys=False)
        df = groups.apply(self._compute_ticker_alt_features)
        df = df.reset_index(drop=True)

        alt_cols = [c for c in df.columns if c.startswith("alt_")]
        logger.info(f"Alternative features: {len(alt_cols)} computed")
        return df

    def _compute_ticker_alt_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """개별 종목 대체 데이터 피처."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        opn = df["open"]
        volume = df["volume"]

        # --- 1. IV Proxy: Parkinson Volatility ---
        # 고저가 기반 변동성 (IV와 높은 상관관계)
        log_hl = np.log(high / low.clip(lower=1e-8))
        for w in [5, 20]:
            parkinson = np.sqrt(
                (1 / (4 * np.log(2))) * (log_hl ** 2).rolling(w).mean()
            ) * np.sqrt(252)
            df[f"alt_parkinson_vol_{w}d"] = parkinson

        # --- 2. IV Proxy: Garman-Klass Volatility ---
        # OHLC 전부 사용 → Parkinson보다 효율적
        log_hl2 = log_hl ** 2
        log_co = np.log(close / opn.clip(lower=1e-8)) ** 2
        gk_daily = 0.5 * log_hl2 - (2 * np.log(2) - 1) * log_co
        for w in [5, 20]:
            gk_vol = np.sqrt(gk_daily.rolling(w).mean().clip(lower=0)) * np.sqrt(252)
            df[f"alt_garman_klass_vol_{w}d"] = gk_vol

        # --- 3. IV Skew Proxy: Downside/Upside Vol Ratio ---
        # 풋-콜 비율의 프록시 — 하락 변동성이 상승보다 크면 skew 음수
        ret_1d = close.pct_change()
        for w in [20, 60]:
            up_ret = ret_1d.where(ret_1d > 0, 0)
            dn_ret = ret_1d.where(ret_1d < 0, 0)
            up_vol = up_ret.rolling(w).std()
            dn_vol = dn_ret.abs().rolling(w).std()
            # skew proxy: >1 means more downside risk (like negative skew / high put demand)
            df[f"alt_vol_skew_{w}d"] = dn_vol / (up_vol + 1e-8)

        # --- 4. Tail Risk Proxy (4th moment: kurtosis) ---
        for w in [20, 60]:
            df[f"alt_return_kurtosis_{w}d"] = ret_1d.rolling(w).apply(
                lambda x: x.kurtosis() if len(x) >= 4 else 0, raw=False
            )

        # --- 5. Volume Regime (high vs low volume periods) ---
        for w in [20]:
            vol_mean = volume.rolling(w).mean()
            vol_std = volume.rolling(w).std()
            df[f"alt_volume_zscore_{w}d"] = np.where(
                vol_std > 0,
                (volume - vol_mean) / vol_std,
                0,
            ).clip(-3, 3)

        # --- 6. Amihud Illiquidity ---
        # 가격 변동 / 거래대금 → 유동성 역수 (높을수록 비유동적)
        dollar_volume = close * volume
        for w in [20]:
            amihud = (ret_1d.abs() / (dollar_volume + 1e-8)).rolling(w).mean()
            # 로그 스케일 (극단값 압축)
            df[f"alt_amihud_illiq_{w}d"] = np.log1p(amihud * 1e6)

        # --- 7. Realized-Implied Vol Spread Proxy ---
        # Close-to-close vol vs Parkinson vol 차이
        # 양수 = realized < implied proxy (일반적으로 vol premium 존재)
        cc_vol_20 = ret_1d.rolling(20).std() * np.sqrt(252)
        park_vol_20 = df.get("alt_parkinson_vol_20d")
        if park_vol_20 is not None:
            df["alt_vol_premium_20d"] = park_vol_20 - cc_vol_20

        # --- 8. Overnight vs Intraday Return ---
        # 시간외 정보 프록시: 야간 수익률 = open/prev_close
        overnight = opn / close.shift(1) - 1
        intraday = close / opn - 1
        df["alt_overnight_return"] = overnight
        df["alt_intraday_return"] = intraday
        # 야간 비중: 야간 수익이 장중보다 크면 정보비대칭 존재
        for w in [10]:
            df[f"alt_overnight_ratio_{w}d"] = (
                overnight.abs().rolling(w).mean()
                / (intraday.abs().rolling(w).mean() + 1e-8)
            )

        return df


class DisclosureNLP:
    """공시 제목 기반 간이 NLP 감성 분석.

    FinBERT 등 대형 모델 대신 키워드 기반으로 빠르게 처리.
    DART 공시 데이터와 함께 사용.
    """

    # 긍정 키워드 (점수 +1)
    POSITIVE_KEYWORDS = [
        "자기주식취득", "배당", "무상증자", "분할", "흑자전환",
        "매출증가", "영업이익증가", "순이익증가", "수주", "계약체결",
        "특허", "승인", "인수합병", "사업확장", "투자결정",
        "상향", "목표가상향", "BUY", "매수", "신사업",
    ]

    # 부정 키워드 (점수 -1)
    NEGATIVE_KEYWORDS = [
        "감사의견거절", "관리종목", "상장폐지", "횡령", "배임",
        "소송제기", "벌금", "과징금", "적자전환", "적자확대",
        "매출감소", "영업손실", "유상증자", "감자", "부도",
        "워크아웃", "회생", "파산", "리콜", "하향",
        "목표가하향", "SELL", "매도", "자기주식처분",
    ]

    def score_text(self, text: str) -> float:
        """텍스트 감성 스코어 (키워드 기반).

        Args:
            text: 공시 제목 또는 본문

        Returns:
            -1.0 ~ +1.0 감성 스코어
        """
        if not isinstance(text, str) or not text:
            return 0.0

        pos_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text)
        neg_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        raw_score = (pos_count - neg_count) / total
        return max(-1.0, min(1.0, raw_score))

    def batch_score(self, texts: list) -> list:
        """여러 텍스트 일괄 감성 분석."""
        return [self.score_text(t) for t in texts]
