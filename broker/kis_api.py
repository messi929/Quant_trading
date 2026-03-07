"""한국투자증권 KIS OpenAPI 래퍼.

모의투자(sandbox)와 실투자(production) 모드를 모두 지원합니다.
실제 거래 전 반드시 sandbox 모드로 충분히 테스트하세요.

참고: https://apiportal.koreainvestment.com
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import urllib3
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# KIS 모의투자 서버(openapivts.koreainvestment.com)는 SSL 인증서 검증 실패
# → InsecureRequestWarning 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class KISApi:
    """한국투자증권 REST API 클라이언트."""

    BASE_URL_SANDBOX        = "https://openapivts.koreainvestment.com:29443"  # 계좌/주문 API
    BASE_URL_SANDBOX_QUOTE  = "https://openapivts.koreainvestment.com:9443"   # 시세/토큰 API
    BASE_URL_PROD           = "https://openapi.koreainvestment.com:9443"

    # TR ID 매핑 (sandbox vs production)
    TR_IDS = {
        "sandbox": {
            "domestic_buy":      "VTTC0802U",
            "domestic_sell":     "VTTC0801U",
            "overseas_buy":      "VTTT1002U",
            "overseas_sell":     "VTTT1001U",
            "balance_domestic":  "VTTC8434R",
            "balance_overseas":  "VTTS3012R",
            "price_domestic":    "FHKST01010100",
            "price_overseas":    "HHDFS00000300",
            "orderbook_domestic":"FHKST01010200",   # 호가 조회
            "pending_domestic":  "VTTC8036R",       # 미체결 조회
            "cancel_domestic":   "VTTC0803U",       # 주문 취소
        },
        "production": {
            "domestic_buy":      "TTTC0802U",
            "domestic_sell":     "TTTC0801U",
            "overseas_buy":      "TTTT1002U",
            "overseas_sell":     "TTTT1006U",
            "balance_domestic":  "TTTC8434R",
            "balance_overseas":  "TTTS3012R",
            "price_domestic":    "FHKST01010100",
            "price_overseas":    "HHDFS00000300",
            "orderbook_domestic":"FHKST01010200",   # 호가 조회
            "pending_domestic":  "TTTC8036R",       # 미체결 조회
            "cancel_domestic":   "TTTC0803U",       # 주문 취소
        },
    }

    def __init__(self, mode: Optional[str] = None, market_type: str = "domestic"):
        """KIS API 초기화.

        Args:
            mode:        "sandbox" (모의투자) | "production" (실투자)
            market_type: "domestic" (국내) | "overseas" (해외)
                         국내/해외 모의투자 계좌는 앱키와 계좌번호가 다릅니다.
        """
        self.market_type = market_type

        # 시장별 자격증명 로드 (신규 DOMESTIC_/OVERSEAS_ 키 우선, 없으면 기존 키 폴백)
        if market_type == "overseas":
            self.app_key   = (os.getenv("KIS_OVERSEAS_APP_KEY")
                              or os.getenv("KIS_APP_KEY", ""))
            self.app_secret = (os.getenv("KIS_OVERSEAS_APP_SECRET")
                               or os.getenv("KIS_APP_SECRET", ""))
            self.cano      = (os.getenv("KIS_OVERSEAS_CANO")
                              or os.getenv("KIS_CANO", ""))
            self.acnt_prdt = (os.getenv("KIS_OVERSEAS_ACNT_PRDT_CD")
                              or os.getenv("KIS_ACNT_PRDT_CD", "01"))
        else:  # domestic
            self.app_key   = (os.getenv("KIS_DOMESTIC_APP_KEY")
                              or os.getenv("KIS_APP_KEY", ""))
            self.app_secret = (os.getenv("KIS_DOMESTIC_APP_SECRET")
                               or os.getenv("KIS_APP_SECRET", ""))
            self.cano      = (os.getenv("KIS_DOMESTIC_CANO")
                              or os.getenv("KIS_CANO", ""))
            self.acnt_prdt = (os.getenv("KIS_DOMESTIC_ACNT_PRDT_CD")
                              or os.getenv("KIS_ACNT_PRDT_CD", "01"))

        self.mode = mode or os.getenv("KIS_MODE", "sandbox")

        if not self.app_key or not self.app_secret:
            raise ValueError(
                f"KIS {market_type} API 자격증명이 없습니다. "
                f".env 파일에 KIS_{'OVERSEAS' if market_type == 'overseas' else 'DOMESTIC'}_APP_KEY/SECRET을 설정하세요."
            )

        self.base_url = (
            self.BASE_URL_SANDBOX if self.mode == "sandbox"
            else self.BASE_URL_PROD
        )
        # 시세 조회 및 토큰 발급: sandbox는 9443 포트 사용
        self._quote_url = (
            self.BASE_URL_SANDBOX_QUOTE if self.mode == "sandbox"
            else self.BASE_URL_PROD
        )
        self._auth_url = self._quote_url
        self.tr_ids = self.TR_IDS[self.mode]

        self._token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        # 토큰 캐시 파일: 같은 APP_KEY면 동일 캐시 공유 (앱키 앞 8자리로 구분)
        key_prefix = self.app_key[:8] if self.app_key else market_type
        self._token_cache_path = Path(f"tracking/.kis_token_{key_prefix}.json")
        self._load_token_cache()

        # KIS 모의투자 서버는 SSL 인증서가 유효하지 않으므로 verify=False 고정
        self._session = requests.Session()
        self._session.verify = False

        logger.info(
            f"KIS API 초기화: mode={self.mode}, market={market_type}, URL={self.base_url}"
        )

    # ------------------------------------------------------------------
    # 인증
    # ------------------------------------------------------------------

    def _load_token_cache(self):
        """디스크에서 토큰 캐시 로드 (재시작 간 토큰 재사용)."""
        try:
            if self._token_cache_path.exists():
                with open(self._token_cache_path, encoding="utf-8") as f:
                    cache = json.load(f)
                expires = datetime.fromisoformat(cache["expires"])
                if expires > datetime.now() + timedelta(minutes=10):
                    self._token = cache["token"]
                    self._token_expires = expires
                    logger.debug("KIS 토큰 캐시에서 로드 완료")
        except Exception:
            pass

    def _save_token_cache(self):
        """토큰을 디스크에 저장."""
        try:
            self._token_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._token_cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "token": self._token,
                    "expires": self._token_expires.isoformat(),
                }, f)
        except Exception as e:
            logger.warning(f"토큰 캐시 저장 실패: {e}")

    def get_token(self) -> str:
        """Access Token 발급 (캐시된 토큰이 유효하면 재사용)."""
        if (self._token and self._token_expires
                and datetime.now() < self._token_expires - timedelta(minutes=10)):
            return self._token

        url = f"{self._auth_url}/oauth2/tokenP"
        body = {
            "grant_type":  "client_credentials",
            "appkey":      self.app_key,
            "appsecret":   self.app_secret,
        }
        import time as _time
        for attempt in range(3):
            resp = self._session.post(url, json=body, timeout=10, verify=False)
            if resp.status_code == 403 and attempt < 2:
                logger.warning(f"토큰 발급 403 (시도 {attempt+1}/3) → 5초 후 재시도")
                _time.sleep(5)
                continue
            resp.raise_for_status()
            break
        data = resp.json()

        self._token = data["access_token"]
        expires_in = int(data.get("expires_in", 86400))
        self._token_expires = datetime.now() + timedelta(seconds=expires_in)
        self._save_token_cache()

        logger.info(f"KIS 토큰 발급 완료 (만료: {self._token_expires:%Y-%m-%d %H:%M})")
        return self._token

    def _headers(self, tr_id: str, extra: Optional[dict] = None) -> dict:
        """공통 HTTP 헤더 생성."""
        h = {
            "content-type":  "application/json; charset=utf-8",
            "authorization": f"Bearer {self.get_token()}",
            "appkey":        self.app_key,
            "appsecret":     self.app_secret,
            "tr_id":         tr_id,
            "custtype":      "P",  # 개인
        }
        if extra:
            h.update(extra)
        return h

    # ------------------------------------------------------------------
    # 시세 조회
    # ------------------------------------------------------------------

    def get_domestic_price(self, ticker: str) -> dict:
        """국내 주식/ETF 현재가 조회.

        Args:
            ticker: 종목코드 (예: "005930" = 삼성전자)

        Returns:
            {"ticker": ..., "price": ..., "volume": ..., "change_pct": ...}
        """
        url = f"{self._quote_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # 주식/ETF
            "FID_INPUT_ISCD": ticker,
        }
        resp = self._session.get(
            url,
            headers=self._headers(self.tr_ids["price_domestic"]),
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"시세 조회 실패: {data.get('msg1')}")

        output = data["output"]
        return {
            "ticker":     ticker,
            "price":      int(output["stck_prpr"]),       # 현재가
            "volume":     int(output["acml_vol"]),         # 누적거래량
            "change_pct": float(output["prdy_ctrt"]),      # 전일대비등락율
            "open":       int(output["stck_oprc"]),        # 시가
            "high":       int(output["stck_hgpr"]),        # 고가
            "low":        int(output["stck_lwpr"]),        # 저가
        }

    def get_domestic_bid_ask(self, ticker: str) -> dict:
        """국내 주식 매수/매도 1호가 조회.

        Returns:
            {"bid": int, "ask": int, "bid_qty": int, "ask_qty": int}
        """
        url = f"{self._quote_url}/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
        }
        resp = self._session.get(
            url,
            headers=self._headers(self.tr_ids["orderbook_domestic"]),
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"호가 조회 실패 [{ticker}]: {data.get('msg1')}")

        output = data["output1"]
        return {
            "ask":     int(output.get("askp1", 0)),       # 매도 1호가
            "bid":     int(output.get("bidp1", 0)),       # 매수 1호가
            "ask_qty": int(output.get("askp_rsqn1", 0)),  # 매도 1호가 잔량
            "bid_qty": int(output.get("bidp_rsqn1", 0)),  # 매수 1호가 잔량
        }

    def get_pending_orders(self) -> list[dict]:
        """국내 미체결 주문 조회.

        Returns:
            [{"order_no", "ticker", "side", "qty", "remaining_qty", "price"}]
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-psble-rvsecncl"
        params = {
            "CANO":           self.cano,
            "ACNT_PRDT_CD":   self.acnt_prdt,
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
            "INQR_DVSN_1":    "0",
            "INQR_DVSN_2":    "0",
        }
        resp = self._session.get(
            url,
            headers=self._headers(self.tr_ids["pending_domestic"]),
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"미체결 조회 실패: {data.get('msg1')}")

        orders = []
        for item in data.get("output", []):
            remaining = int(item.get("rmn_qty", 0))
            if remaining > 0:
                orders.append({
                    "order_no":      item.get("odno", ""),
                    "ticker":        item.get("pdno", ""),
                    "side":          "buy" if item.get("sll_buy_dvsn_cd") == "02" else "sell",
                    "qty":           int(item.get("ord_qty", 0)),
                    "remaining_qty": remaining,
                    "price":         int(item.get("ord_unpr", 0)),
                })
        return orders

    def cancel_order(
        self,
        order_no: str,
        ticker: str,
        side: str,
        qty: int,
    ) -> bool:
        """국내 미체결 주문 취소.

        Returns:
            True if cancelled, False if failed
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-rvsecncl"
        tr_id = self.tr_ids["cancel_domestic"]

        body = {
            "CANO":              self.cano,
            "ACNT_PRDT_CD":      self.acnt_prdt,
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO":         order_no,
            "ORD_DVSN":          "00",   # 지정가 (원주문 유형)
            "RVSE_CNCL_DVSN_CD": "02",  # 취소
            "ORD_QTY":           str(qty),
            "ORD_UNPR":          "0",
            "QTY_ALL_ORD_YN":    "Y",    # 잔량 전부 취소
        }

        resp = self._session.post(
            url,
            headers=self._headers(tr_id),
            json=body,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            logger.warning(f"주문 취소 실패 [{order_no}]: {data.get('msg1')}")
            return False

        logger.info(f"주문 취소 완료: {ticker} {order_no}")
        return True

    def get_overseas_price(self, ticker: str, exchange: str = "NAS") -> dict:
        """해외 주식/ETF 현재가 조회.

        Args:
            ticker: 티커 (예: "XLK")
            exchange: 거래소 코드 (NAS=나스닥, NYSE=뉴욕)
        """
        url = f"{self._quote_url}/uapi/overseas-stock/v1/quotations/price"
        params = {
            "AUTH":         "",
            "EXCD":         exchange,
            "SYMB":         ticker,
        }
        resp = self._session.get(
            url,
            headers=self._headers(self.tr_ids["price_overseas"]),
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"해외 시세 조회 실패: {data.get('msg1')}")

        output = data["output"]
        return {
            "ticker":     ticker,
            "price":      float(output["last"]),
            "volume":     int(output.get("tvol", 0)),
            "change_pct": float(output.get("rate", 0)),
        }

    # ------------------------------------------------------------------
    # 잔고 조회
    # ------------------------------------------------------------------

    def get_domestic_balance(self) -> dict:
        """국내 계좌 잔고 조회.

        Returns:
            {"cash": ..., "positions": [{"ticker", "qty", "avg_price", "current_price", "eval_amount"}]}
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        params = {
            "CANO":                 self.cano,
            "ACNT_PRDT_CD":         self.acnt_prdt,
            "AFHR_FLPR_YN":         "N",
            "OFL_YN":               "",
            "INQR_DVSN":            "02",
            "UNPR_DVSN":            "01",
            "FUND_STTL_ICLD_YN":    "N",
            "FNCG_AMT_AUTO_RDPT_YN":"N",
            "PRCS_DVSN":            "01",
            "CTX_AREA_FK100":       "",
            "CTX_AREA_NK100":       "",
        }
        resp = self._session.get(
            url,
            headers=self._headers(self.tr_ids["balance_domestic"]),
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"잔고 조회 실패: {data.get('msg1')}")

        positions = []
        for item in data.get("output1", []):
            qty = int(item.get("hldg_qty", 0))
            if qty > 0:
                positions.append({
                    "ticker":        item["pdno"],
                    "name":          item["prdt_name"],
                    "qty":           qty,
                    "avg_price":     float(item["pchs_avg_pric"]),
                    "current_price": float(item["prpr"]),
                    "eval_amount":   float(item["evlu_amt"]),
                    "profit_loss":   float(item["evlu_pfls_amt"]),
                    "profit_pct":    float(item["evlu_pfls_rt"]),
                })

        output2 = data.get("output2", [{}])[0]
        return {
            "cash":          float(output2.get("dnca_tot_amt", 0)),    # 예수금
            "total_eval":    float(output2.get("tot_evlu_amt", 0)),    # 총평가금액
            "total_profit":  float(output2.get("evlu_pfls_smtl_amt", 0)),
            "positions":     positions,
        }

    def get_overseas_balance(self, currency: str = "USD") -> dict:
        """해외 계좌 잔고 조회."""
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
        params = {
            "CANO":         self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt,
            "OVRS_EXCG_CD": "NASD",   # 나스닥
            "TR_CRCY_CD":   currency,
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        resp = self._session.get(
            url,
            headers=self._headers(self.tr_ids["balance_overseas"]),
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"해외 잔고 조회 실패: {data.get('msg1')}")

        positions = []
        for item in data.get("output1", []):
            qty = float(item.get("ovrs_cblc_qty", 0))
            if qty > 0:
                positions.append({
                    "ticker":        item["ovrs_pdno"],
                    "name":          item["ovrs_item_name"],
                    "qty":           qty,
                    "avg_price":     float(item["pchs_avg_pric"]),
                    "current_price": float(item["now_pric2"]),
                    "eval_amount":   float(item["ovrs_stck_evlu_amt"]),
                    "profit_loss":   float(item["frcr_evlu_pfls_amt"]),
                    "profit_pct":    float(item["evlu_pfls_rt"]),
                })

        output2 = data.get("output2", {})
        return {
            "cash":       float(output2.get("frcr_dncl_amt_2", 0)),
            "total_eval": float(output2.get("tot_evlu_pfls_amt", 0)),
            "positions":  positions,
        }

    # ------------------------------------------------------------------
    # 주문
    # ------------------------------------------------------------------

    def order_domestic(
        self,
        ticker: str,
        side: str,           # "buy" or "sell"
        qty: int,
        price: int = 0,      # 0 = 시장가
        order_type: str = "01",  # "00"=지정가, "01"=시장가
    ) -> dict:
        """국내 주식/ETF 주문.

        Args:
            ticker: 종목코드
            side: "buy" or "sell"
            qty: 수량
            price: 주문가 (시장가=0)
            order_type: "00"=지정가, "01"=시장가

        Returns:
            주문 결과 dict
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        tr_id = (self.tr_ids["domestic_buy"] if side == "buy"
                 else self.tr_ids["domestic_sell"])

        body = {
            "CANO":         self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt,
            "PDNO":         ticker,
            "ORD_DVSN":     order_type,
            "ORD_QTY":      str(qty),
            "ORD_UNPR":     str(price),
        }

        logger.info(f"주문: {side.upper()} {ticker} x{qty} (mode={self.mode})")

        resp = self._session.post(
            url,
            headers=self._headers(tr_id),
            json=body,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(
                f"주문 실패 [{ticker}]: {data.get('msg1')} (code={data.get('msg_cd')})"
            )

        output = data.get("output", {})
        logger.info(
            f"주문 완료: {side.upper()} {ticker} x{qty} "
            f"→ 주문번호={output.get('ODNO')}"
        )
        return {
            "order_no":  output.get("ODNO"),
            "ticker":    ticker,
            "side":      side,
            "qty":       qty,
            "price":     price,
            "timestamp": datetime.now().isoformat(),
            "mode":      self.mode,
        }

    def order_overseas(
        self,
        ticker: str,
        side: str,
        qty: int,
        price: float = 0.0,
        exchange: str = "NASD",
    ) -> dict:
        """해외 주식/ETF 주문 (나스닥 기본).

        Args:
            ticker: 티커 (예: "XLK")
            side: "buy" or "sell"
            qty: 수량 (정수)
            price: 주문가 (0=시장가)
            exchange: 거래소 ("NASD"=나스닥, "NYSE"=뉴욕)
        """
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        tr_id = (self.tr_ids["overseas_buy"] if side == "buy"
                 else self.tr_ids["overseas_sell"])

        order_type = "00" if price == 0 else "00"  # 해외는 00=지정가, 시장가 없음
        if price == 0:
            # 해외주식은 시장가 없음 → 현재가의 ±2% 지정가로 처리
            price_info = self.get_overseas_price(ticker, exchange[:3])
            price = round(price_info["price"] * (1.02 if side == "buy" else 0.98), 2)

        body = {
            "CANO":         self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt,
            "OVRS_EXCG_CD": exchange,
            "PDNO":         ticker,
            "ORD_DVSN":     "00",
            "ORD_QTY":      str(qty),
            "OVRS_ORD_UNPR": f"{price:.2f}",
            "ORD_SVR_DVSN_CD": "0",
        }

        logger.info(f"해외주문: {side.upper()} {ticker} x{qty} @{price} (mode={self.mode})")

        resp = self._session.post(
            url,
            headers=self._headers(tr_id),
            json=body,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(
                f"해외 주문 실패 [{ticker}]: {data.get('msg1')}"
            )

        output = data.get("output", {})
        logger.info(f"해외 주문 완료: {ticker} x{qty} → {output.get('ODNO')}")
        return {
            "order_no":  output.get("ODNO"),
            "ticker":    ticker,
            "side":      side,
            "qty":       qty,
            "price":     price,
            "exchange":  exchange,
            "timestamp": datetime.now().isoformat(),
            "mode":      self.mode,
        }

    def cancel_all_pending(self):
        """미체결 주문 전량 취소 (긴급 청산 시 사용)."""
        logger.warning("미체결 주문 전량 취소 요청")
        # TODO: 미체결 조회 후 취소 구현
        raise NotImplementedError("미체결 취소는 추후 구현 예정")
