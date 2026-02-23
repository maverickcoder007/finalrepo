from __future__ import annotations

import asyncio
import csv
import io
import ssl
from typing import Any, Optional

import aiohttp
from aiolimiter import AsyncLimiter

from src.auth.authenticator import KiteAuthenticator
from src.data.models import (
    Exchange,
    GTTOrder,
    GTTRequest,
    Holding,
    HistoricalCandle,
    Instrument,
    Interval,
    LTPQuote,
    Margins,
    OHLCQuote,
    Order,
    OrderModifyRequest,
    OrderRequest,
    OrderVariety,
    Position,
    Positions,
    Quote,
    SegmentMargin,
    Trade,
    UserProfile,
)
from src.utils.config import get_settings
from src.utils.exceptions import (
    AuthenticationError,
    DataError,
    KiteError,
    NetworkError,
    OrderError,
    RateLimitError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KiteClient:
    def __init__(self, authenticator: Optional[KiteAuthenticator] = None) -> None:
        self._settings = get_settings()
        self._auth = authenticator or KiteAuthenticator()
        self._session: Optional[aiohttp.ClientSession] = None

        self._default_limiter = AsyncLimiter(self._settings.rate_limit_default, 1)
        self._quote_limiter = AsyncLimiter(self._settings.rate_limit_quote, 1)
        self._historical_limiter = AsyncLimiter(self._settings.rate_limit_historical, 1)
        self._order_limiter = AsyncLimiter(self._settings.rate_limit_orders, 1)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # Skip SSL verification if disable_ssl_verify is set (for development only)
            ssl_context = None
            if self._settings.disable_ssl_verify:
                logger.error(
                    "security_warning",
                    msg="SSL verification is DISABLED - this is not secure for production!"
                )
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self._session = aiohttp.ClientSession(
                base_url=self._settings.kite_base_url,
                timeout=aiohttp.ClientTimeout(total=30),
                connector=connector,
            )
        return self._session

    async def _recreate_session(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        limiter: Optional[AsyncLimiter] = None,
        params: Any = None,
        data: Optional[dict[str, Any]] = None,
        parse_json: bool = True,
    ) -> Any:
        rate_limiter = limiter or self._default_limiter
        settings = self._settings

        last_error: Optional[Exception] = None
        for attempt in range(settings.max_retries):
            try:
                async with rate_limiter:
                    session = await self._get_session()
                    headers = self._auth.get_auth_header()
                    async with session.request(
                        method, endpoint, params=params, data=data, headers=headers
                    ) as response:
                        if response.status == 429:
                            wait_time = settings.retry_delay * (2 ** attempt)
                            logger.warning(
                                "rate_limited", endpoint=endpoint, wait=wait_time
                            )
                            await asyncio.sleep(wait_time)
                            continue

                        if response.status == 403:
                            try:
                                body = await response.json()
                                error_msg = body.get("message", "Token expired or invalid")
                                error_type = body.get("error_type", "")
                            except Exception:
                                error_msg = "Token expired or invalid"
                                error_type = ""
                            logger.error("auth_failed", endpoint=endpoint, status=403, error_msg=error_msg, error_type=error_type)
                            self._auth.invalidate_session()
                            await self._recreate_session()
                            raise AuthenticationError(
                                f"{error_msg} (HTTP 403)", response.status
                            )

                        if not parse_json:
                            return await response.text()

                        result = await response.json()

                        if response.status != 200:
                            error_msg = result.get("message", "Request failed")
                            error_type = result.get("error_type", "")
                            if response.status in (401, 403):
                                raise AuthenticationError(error_msg, response.status)
                            elif 400 <= response.status < 500:
                                raise OrderError(error_msg, response.status)
                            else:
                                raise NetworkError(error_msg, response.status)

                        return result.get("data", result)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                wait_time = settings.retry_delay * (2 ** attempt)
                logger.warning(
                    "request_retry",
                    endpoint=endpoint,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < settings.max_retries - 1:
                    await asyncio.sleep(wait_time)
            except KiteError:
                raise

        raise NetworkError(f"Request failed after {settings.max_retries} attempts: {last_error}")

    async def get_profile(self) -> UserProfile:
        data = await self._request("GET", "/user/profile")
        return UserProfile(**data)

    async def get_margins(self, segment: Optional[str] = None) -> Margins:
        endpoint = f"/user/margins/{segment}" if segment else "/user/margins"
        data = await self._request("GET", endpoint)
        if segment:
            return Margins(**{segment: SegmentMargin(**data)})
        result = {}
        for seg_name in ("equity", "commodity"):
            if seg_name in data:
                result[seg_name] = SegmentMargin(**data[seg_name])
        return Margins(**result)

    async def get_orders(self) -> list[Order]:
        data = await self._request("GET", "/orders")
        if not data:
            return []
        return [Order(**o) for o in data]

    async def get_order_history(self, order_id: str) -> list[Order]:
        data = await self._request("GET", f"/orders/{order_id}")
        if not data:
            return []
        return [Order(**o) for o in data]

    async def get_trades(self) -> list[Trade]:
        data = await self._request("GET", "/trades")
        if not data:
            return []
        return [Trade(**t) for t in data]

    async def get_order_trades(self, order_id: str) -> list[Trade]:
        data = await self._request("GET", f"/orders/{order_id}/trades")
        if not data:
            return []
        return [Trade(**t) for t in data]

    async def place_order(self, order: OrderRequest) -> str:
        payload: dict[str, Any] = {
            "tradingsymbol": order.tradingsymbol,
            "exchange": order.exchange.value,
            "transaction_type": order.transaction_type.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "product": order.product.value,
            "validity": order.validity.value,
        }
        if order.price is not None:
            payload["price"] = order.price
        if order.trigger_price is not None:
            payload["trigger_price"] = order.trigger_price
        if order.disclosed_quantity is not None:
            payload["disclosed_quantity"] = order.disclosed_quantity
        if order.validity_ttl is not None:
            payload["validity_ttl"] = order.validity_ttl
        if order.iceberg_legs is not None:
            payload["iceberg_legs"] = order.iceberg_legs
        if order.iceberg_quantity is not None:
            payload["iceberg_quantity"] = order.iceberg_quantity
        if order.tag is not None:
            payload["tag"] = order.tag

        variety = order.variety.value
        data = await self._request(
            "POST", f"/orders/{variety}", limiter=self._order_limiter, data=payload
        )
        order_id = data.get("order_id", "")
        logger.info(
            "order_placed",
            order_id=order_id,
            tradingsymbol=order.tradingsymbol,
            transaction_type=order.transaction_type.value,
            quantity=order.quantity,
        )
        return order_id

    async def modify_order(self, request: OrderModifyRequest) -> str:
        payload: dict[str, Any] = {}
        if request.order_type is not None:
            payload["order_type"] = request.order_type.value
        if request.quantity is not None:
            payload["quantity"] = request.quantity
        if request.price is not None:
            payload["price"] = request.price
        if request.trigger_price is not None:
            payload["trigger_price"] = request.trigger_price
        if request.disclosed_quantity is not None:
            payload["disclosed_quantity"] = request.disclosed_quantity
        if request.validity is not None:
            payload["validity"] = request.validity.value

        variety = request.variety.value
        data = await self._request(
            "PUT", f"/orders/{variety}/{request.order_id}",
            limiter=self._order_limiter, data=payload,
        )
        order_id = data.get("order_id", request.order_id)
        logger.info("order_modified", order_id=order_id)
        return order_id

    async def cancel_order(
        self, order_id: str, variety: OrderVariety = OrderVariety.REGULAR
    ) -> str:
        data = await self._request(
            "DELETE", f"/orders/{variety.value}/{order_id}",
            limiter=self._order_limiter,
        )
        result_id = data.get("order_id", order_id)
        logger.info("order_cancelled", order_id=result_id)
        return result_id

    async def get_positions(self) -> Positions:
        data = await self._request("GET", "/portfolio/positions")
        return Positions(
            net=[Position(**p) for p in data.get("net", [])],
            day=[Position(**p) for p in data.get("day", [])],
        )

    async def get_holdings(self) -> list[Holding]:
        data = await self._request("GET", "/portfolio/holdings")
        if not data:
            return []
        return [Holding(**h) for h in data]

    async def convert_position(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        position_type: str,
        quantity: int,
        old_product: str,
        new_product: str,
    ) -> bool:
        payload = {
            "tradingsymbol": tradingsymbol,
            "exchange": exchange,
            "transaction_type": transaction_type,
            "position_type": position_type,
            "quantity": quantity,
            "old_product": old_product,
            "new_product": new_product,
        }
        await self._request("PUT", "/portfolio/positions", data=payload)
        return True

    async def get_instruments(self, exchange: Optional[str] = None) -> list[Instrument]:
        endpoint = f"/instruments/{exchange}" if exchange else "/instruments"
        csv_text = await self._request("GET", endpoint, parse_json=False)
        instruments: list[Instrument] = []
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            instruments.append(
                Instrument(
                    instrument_token=int(row.get("instrument_token", 0)),
                    exchange_token=int(row.get("exchange_token", 0)),
                    tradingsymbol=row.get("tradingsymbol", ""),
                    name=row.get("name", ""),
                    last_price=float(row.get("last_price", 0)),
                    expiry=row.get("expiry") or None,
                    strike=float(row.get("strike", 0)),
                    tick_size=float(row.get("tick_size", 0)),
                    lot_size=int(row.get("lot_size", 0)),
                    instrument_type=row.get("instrument_type", ""),
                    segment=row.get("segment", ""),
                    exchange=row.get("exchange", ""),
                )
            )
        return instruments

    async def get_quote(self, instruments: list[str]) -> dict[str, Quote]:
        params = [("i", inst) for inst in instruments]
        data = await self._request(
            "GET", "/quote", limiter=self._quote_limiter, params=params
        )
        return {key: Quote(**val) for key, val in data.items()}

    async def get_ohlc(self, instruments: list[str]) -> dict[str, OHLCQuote]:
        params = [("i", inst) for inst in instruments]
        data = await self._request(
            "GET", "/quote/ohlc", limiter=self._quote_limiter, params=params
        )
        return {key: OHLCQuote(**val) for key, val in data.items()}

    async def get_ltp(self, instruments: list[str]) -> dict[str, LTPQuote]:
        params = [("i", inst) for inst in instruments]
        data = await self._request(
            "GET", "/quote/ltp", limiter=self._quote_limiter, params=params
        )
        return {key: LTPQuote(**val) for key, val in data.items()}

    async def get_historical_data(
        self,
        instrument_token: int,
        interval: Interval,
        from_date: str,
        to_date: str,
        continuous: bool = False,
        oi: bool = False,
    ) -> list[HistoricalCandle]:
        endpoint = f"/instruments/historical/{instrument_token}/{interval.value}"
        params: dict[str, Any] = {"from": from_date, "to": to_date}
        if continuous:
            params["continuous"] = 1
        if oi:
            params["oi"] = 1

        data = await self._request(
            "GET", endpoint, limiter=self._historical_limiter, params=params
        )
        candles = data.get("candles", []) if isinstance(data, dict) else []
        result: list[HistoricalCandle] = []
        for c in candles:
            candle = HistoricalCandle(
                timestamp=c[0],
                open=c[1],
                high=c[2],
                low=c[3],
                close=c[4],
                volume=c[5],
                oi=c[6] if len(c) > 6 else None,
            )
            result.append(candle)
        return result

    async def place_gtt(self, gtt: GTTRequest) -> int:
        payload: dict[str, Any] = {
            "type": gtt.trigger_type.value,
            "condition": {
                "exchange": gtt.exchange.value,
                "tradingsymbol": gtt.tradingsymbol,
                "trigger_values": gtt.trigger_values,
                "last_price": gtt.last_price,
            },
            "orders": gtt.orders,
        }
        data = await self._request("POST", "/gtt/triggers", data=payload)
        trigger_id = data.get("trigger_id", 0)
        logger.info("gtt_placed", trigger_id=trigger_id)
        return trigger_id

    async def get_gtts(self) -> list[GTTOrder]:
        data = await self._request("GET", "/gtt/triggers")
        if not data:
            return []
        return [GTTOrder(**g) for g in data]

    async def get_gtt(self, trigger_id: int) -> GTTOrder:
        data = await self._request("GET", f"/gtt/triggers/{trigger_id}")
        return GTTOrder(**data)

    async def modify_gtt(self, trigger_id: int, gtt: GTTRequest) -> int:
        payload: dict[str, Any] = {
            "type": gtt.trigger_type.value,
            "condition": {
                "exchange": gtt.exchange.value,
                "tradingsymbol": gtt.tradingsymbol,
                "trigger_values": gtt.trigger_values,
                "last_price": gtt.last_price,
            },
            "orders": gtt.orders,
        }
        data = await self._request("PUT", f"/gtt/triggers/{trigger_id}", data=payload)
        return data.get("trigger_id", trigger_id)

    async def delete_gtt(self, trigger_id: int) -> int:
        data = await self._request("DELETE", f"/gtt/triggers/{trigger_id}")
        logger.info("gtt_deleted", trigger_id=trigger_id)
        return data.get("trigger_id", trigger_id)
