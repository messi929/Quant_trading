"""Data collection from Yahoo Finance for KOSPI and NASDAQ markets."""

from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from loguru import logger
from pykrx import stock as krx

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class MarketDataCollector:
    """Collects OHLCV data from Yahoo Finance for KOSPI and NASDAQ."""

    def __init__(
        self,
        save_dir: str = "data/raw",
        history_years: int = 20,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.start_date = (
            datetime.now() - timedelta(days=history_years * 365)
        ).strftime("%Y-%m-%d")
        logger.info(f"Collector period: {self.start_date} ~ {self.end_date}")

    # ------------------------------------------------------------------
    # Ticker listing
    # ------------------------------------------------------------------

    def get_kospi_tickers(self) -> pd.DataFrame:
        """Get all KOSPI listed tickers from KRX.

        get_market_ticker_list(date)는 OHLCV 기반 엔드포인트라 장 미개장 시 0건 반환.
        상장종목검색 API는 날짜 무관하게 현재 상장 목록을 반환하므로 이를 사용.
        """
        try:
            from pykrx.website.krx.market.core import 상장종목검색
            raw = 상장종목검색().fetch("STK")  # STK = 유가증권(KOSPI)
            if raw.empty:
                raise ValueError("상장종목검색 빈 결과")
            records = []
            for _, row in raw.iterrows():
                ticker = row["short_code"]
                name = row["codeName"]
                records.append(
                    {
                        "ticker": ticker,
                        "name": name,
                        "yf_ticker": f"{ticker}.KS",
                        "market": "KOSPI",
                    }
                )
            df = pd.DataFrame(records)
            logger.info(f"KOSPI tickers: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"Failed to get KOSPI tickers: {e}")
            raise

    def get_nasdaq_tickers(self) -> pd.DataFrame:
        """Get NASDAQ composite tickers via yfinance screening.

        Falls back to a curated large-cap + mid-cap list
        if full listing is unavailable.
        """
        try:
            # Use S&P 500 + NASDAQ-100 as representative universe
            sp500_url = (
                "https://en.wikipedia.org/wiki/"
                "List_of_S%26P_500_companies"
            )
            resp = requests.get(sp500_url, headers=_HTTP_HEADERS, timeout=30)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            sp500 = tables[0]
            tickers_sp500 = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()

            # NASDAQ-100
            nq100_url = (
                "https://en.wikipedia.org/wiki/"
                "Nasdaq-100#Components"
            )
            resp_nq = requests.get(nq100_url, headers=_HTTP_HEADERS, timeout=30)
            resp_nq.raise_for_status()
            tables_nq = pd.read_html(StringIO(resp_nq.text))
            # Find table with Ticker column
            nq100_tickers = []
            for t in tables_nq:
                cols = [str(c) for c in t.columns]
                for col in cols:
                    if "ticker" in col.lower() or "symbol" in col.lower():
                        nq100_tickers = (
                            t[col].dropna().astype(str)
                            .str.replace(".", "-", regex=False).tolist()
                        )
                        break
                if nq100_tickers:
                    break

            all_tickers = list(set(tickers_sp500 + nq100_tickers))
            records = [
                {
                    "ticker": t,
                    "name": t,
                    "yf_ticker": t,
                    "market": "NASDAQ",
                }
                for t in all_tickers
            ]
            df = pd.DataFrame(records)
            logger.info(f"US tickers (S&P500 + NQ100): {len(df)}")
            return df
        except Exception as e:
            logger.error(f"Failed to get US tickers: {e}")
            raise

    def get_kospi_delisted_tickers(self, years_back: int = 5) -> pd.DataFrame:
        """상장폐지 종목 목록 수집 (생존편향 제거용).

        pykrx의 상장폐지종목 API를 사용하여 최근 N년간 폐지된 종목을 수집.
        폐지 종목의 과거 OHLCV도 학습 데이터에 포함하여 생존편향을 제거.

        Args:
            years_back: 몇 년 전까지의 폐지 종목을 포함할지

        Returns:
            DataFrame with ticker, name, yf_ticker, market, delist_date columns
        """
        records = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)

        try:
            # pykrx에서 상장폐지 종목 조회
            from pykrx.website.krx.market.core import 상장폐지종목검색
            raw = 상장폐지종목검색().fetch("STK")  # STK = 유가증권(KOSPI)

            if raw is None or raw.empty:
                logger.warning("상장폐지 종목 데이터 없음 — pykrx API 미지원 가능")
                return pd.DataFrame(columns=["ticker", "name", "yf_ticker", "market", "delist_date"])

            for _, row in raw.iterrows():
                ticker = str(row.get("short_code", ""))
                name = str(row.get("codeName", ""))
                # 폐지일 파싱 시도
                delist_date_str = str(row.get("delist_date", row.get("상장폐지일", "")))

                if not ticker:
                    continue

                records.append({
                    "ticker": ticker,
                    "name": name,
                    "yf_ticker": f"{ticker}.KS",
                    "market": "KOSPI_DELISTED",
                    "delist_date": delist_date_str,
                })

            df = pd.DataFrame(records)
            logger.info(f"상장폐지 종목: {len(df)}개 수집")
            return df

        except ImportError:
            logger.warning("pykrx 상장폐지종목검색 미지원 — 대체 방법 시도")
        except Exception as e:
            logger.warning(f"상장폐지 종목 조회 실패: {e}")

        # 대체 방법: KRX API로 과거 시점별 상장 목록 비교
        try:
            current_tickers = set()
            try:
                from pykrx.website.krx.market.core import 상장종목검색
                raw_current = 상장종목검색().fetch("STK")
                current_tickers = set(raw_current["short_code"].tolist()) if not raw_current.empty else set()
            except Exception:
                pass

            # 과거 각 연도 말일의 상장 목록에서 현재 없는 종목 = 폐지 종목
            for year_offset in range(1, years_back + 1):
                check_date = (end_date - timedelta(days=year_offset * 365)).strftime("%Y%m%d")
                try:
                    past_tickers = krx.get_market_ticker_list(check_date, market="KOSPI")
                    for ticker in past_tickers:
                        if ticker not in current_tickers:
                            try:
                                name = krx.get_market_ticker_name(ticker)
                            except Exception:
                                name = ticker
                            records.append({
                                "ticker": ticker,
                                "name": name,
                                "yf_ticker": f"{ticker}.KS",
                                "market": "KOSPI_DELISTED",
                                "delist_date": f"~{end_date.year - year_offset}",
                            })
                            current_tickers.add(ticker)  # 중복 방지
                except Exception as e:
                    logger.debug(f"과거 {check_date} 목록 조회 실패: {e}")
                    continue

            df = pd.DataFrame(records)
            if not df.empty:
                df = df.drop_duplicates(subset="ticker")
            logger.info(f"상장폐지 추정 종목: {len(df)}개 (과거 목록 비교)")
            return df

        except Exception as e:
            logger.warning(f"상장폐지 종목 대체 조회 실패: {e}")
            return pd.DataFrame(columns=["ticker", "name", "yf_ticker", "market", "delist_date"])

    # ------------------------------------------------------------------
    # OHLCV download
    # ------------------------------------------------------------------

    def download_ohlcv(
        self,
        tickers: list[str],
        market: str,
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """Download OHLCV data in batches.

        Args:
            tickers: List of yfinance-compatible ticker symbols
            market: Market identifier ("KOSPI" or "NASDAQ")
            batch_size: Number of tickers per download batch

        Returns:
            DataFrame with multi-index (date, ticker) and OHLCV columns
        """
        all_data = []
        failed = []
        total = len(tickers)

        for i in range(0, total, batch_size):
            batch = tickers[i : i + batch_size]
            batch_str = " ".join(batch)
            logger.info(
                f"[{market}] Downloading batch {i // batch_size + 1}"
                f"/{(total + batch_size - 1) // batch_size} "
                f"({len(batch)} tickers)"
            )
            try:
                data = yf.download(
                    batch_str,
                    start=self.start_date,
                    end=self.end_date,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                )
                if data.empty:
                    failed.extend(batch)
                    continue

                # Reshape multi-ticker download
                if len(batch) == 1:
                    data.columns = pd.MultiIndex.from_product(
                        [[batch[0]], data.columns]
                    )

                for ticker in batch:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            df = data[ticker].copy()
                            df = df.dropna(how="all")
                            if len(df) > 0:
                                df["ticker"] = ticker
                                df["market"] = market
                                df.index.name = "date"
                                all_data.append(df.reset_index())
                    except (KeyError, Exception):
                        failed.append(ticker)

            except Exception as e:
                logger.warning(f"Batch download failed: {e}")
                failed.extend(batch)

        if failed:
            logger.warning(f"[{market}] Failed tickers: {len(failed)}")

        if not all_data:
            logger.error(f"[{market}] No data collected")
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        logger.info(
            f"[{market}] Collected {len(result)} rows "
            f"for {result['ticker'].nunique()} tickers"
        )
        return result

    # ------------------------------------------------------------------
    # Full collection
    # ------------------------------------------------------------------

    def collect_all(self, save: bool = True) -> dict[str, pd.DataFrame]:
        """Collect data for all markets.

        Returns:
            Dict mapping market name to DataFrame
        """
        results = {}

        # KOSPI
        logger.info("=== Collecting KOSPI data ===")
        kospi_info = self.get_kospi_tickers()
        kospi_data = self.download_ohlcv(
            kospi_info["yf_ticker"].tolist(), "KOSPI"
        )
        results["KOSPI"] = kospi_data

        # NASDAQ / US
        logger.info("=== Collecting NASDAQ data ===")
        nasdaq_info = self.get_nasdaq_tickers()
        nasdaq_data = self.download_ohlcv(
            nasdaq_info["yf_ticker"].tolist(), "NASDAQ"
        )
        results["NASDAQ"] = nasdaq_data

        # KOSPI Delisted (survivorship bias removal)
        logger.info("=== Collecting KOSPI Delisted data ===")
        delisted_info = pd.DataFrame()
        try:
            delisted_info = self.get_kospi_delisted_tickers()
            if not delisted_info.empty:
                delisted_data = self.download_ohlcv(
                    delisted_info["yf_ticker"].tolist(), "KOSPI_DELISTED"
                )
                if not delisted_data.empty:
                    # Mark as KOSPI market but flag delisted
                    delisted_data["market"] = "KOSPI"
                    delisted_data["is_delisted"] = True
                    results["KOSPI_DELISTED"] = delisted_data
                    logger.info(
                        f"Delisted KOSPI data: {len(delisted_data)} rows, "
                        f"{delisted_data['ticker'].nunique()} tickers"
                    )
        except Exception as e:
            logger.warning(f"Delisted data collection failed (continuing): {e}")

        if save:
            for market, df in results.items():
                if market == "KOSPI_DELISTED":
                    continue  # Merged into KOSPI below
                if not df.empty:
                    path = self.save_dir / f"{market.lower()}_ohlcv.parquet"
                    df.to_parquet(path, engine="pyarrow", compression="snappy")
                    logger.info(f"Saved {path} ({len(df)} rows)")

            # Merge delisted into KOSPI if available
            if "KOSPI_DELISTED" in results and not results["KOSPI_DELISTED"].empty:
                kospi_combined = pd.concat(
                    [results.get("KOSPI", pd.DataFrame()), results["KOSPI_DELISTED"]],
                    ignore_index=True,
                )
                if "is_delisted" not in kospi_combined.columns:
                    kospi_combined["is_delisted"] = False
                kospi_combined["is_delisted"] = kospi_combined["is_delisted"].fillna(False)
                path = self.save_dir / "kospi_ohlcv.parquet"
                kospi_combined.to_parquet(path, engine="pyarrow", compression="snappy")
                logger.info(f"Saved KOSPI+delisted: {path} ({len(kospi_combined)} rows)")

            # Save ticker info (including delisted)
            all_info = [kospi_info, nasdaq_info]
            if not delisted_info.empty:
                all_info.append(delisted_info)
            ticker_info = pd.concat(all_info, ignore_index=True)
            ticker_path = self.save_dir / "ticker_info.parquet"
            ticker_info.to_parquet(ticker_path)
            logger.info(f"Saved ticker info: {ticker_path}")

        return results


class DataCollector:
    """Settings-aware wrapper for incremental market data collection.

    Used by daily_runner for incremental updates with date range support.
    """

    def __init__(self, settings: dict):
        self.settings = settings
        self.save_dir = Path(settings["paths"]["raw_data"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def collect_all(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Collect incremental data for the given date range.

        Args:
            start_date: Start date string "YYYY-MM-DD"
            end_date:   End date string "YYYY-MM-DD"

        Returns:
            Combined DataFrame with KOSPI + NASDAQ data, or None on failure
        """
        collector = MarketDataCollector(
            save_dir=str(self.save_dir),
            history_years=1,  # Placeholder — overridden below
        )
        # Override date range
        collector.start_date = start_date
        collector.end_date = end_date

        results = []

        try:
            kospi_info = collector.get_kospi_tickers()
            if kospi_info.empty or "yf_ticker" not in kospi_info.columns:
                raise ValueError(f"KOSPI ticker 없음 (KRX 0건 반환)")
            kospi_data = collector.download_ohlcv(
                kospi_info["yf_ticker"].tolist(), "KOSPI"
            )
            if not kospi_data.empty:
                kospi_data["market"] = "KOSPI"
                results.append(kospi_data)
        except Exception as e:
            logger.warning(f"KOSPI 수집 실패 (계속 진행): {e}")

        try:
            nasdaq_info = collector.get_nasdaq_tickers()
            nasdaq_data = collector.download_ohlcv(
                nasdaq_info["yf_ticker"].tolist(), "NASDAQ"
            )
            if not nasdaq_data.empty:
                nasdaq_data["market"] = "NASDAQ"
                results.append(nasdaq_data)
        except Exception as e:
            logger.warning(f"NASDAQ 수집 실패 (계속 진행): {e}")

        if not results:
            return None

        combined = pd.concat(results, ignore_index=True)
        logger.info(
            f"증분 수집 완료: {len(combined):,}행, "
            f"{start_date} ~ {end_date}"
        )
        return combined
