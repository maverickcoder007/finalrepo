from __future__ import annotations

import hashlib
import json
import os
import ssl
import time
from typing import Optional

import aiohttp

from src.data.models import SessionData
from src.utils.config import get_settings
from src.utils.exceptions import AuthenticationError
from src.utils.logger import get_logger, sanitize_log_data

logger = get_logger(__name__)

TOKEN_STORE_FILE = ".kite_session.json"


class KiteAuthenticator:
    def __init__(self, api_key: str = "", api_secret: str = "", zerodha_id: str = "") -> None:
        self._settings = get_settings()
        self._user_api_key = api_key or self._settings.kite_api_key
        self._user_api_secret = api_secret or self._settings.kite_api_secret
        self._zerodha_id = zerodha_id
        self._session_data: Optional[SessionData] = None
        self._token_timestamp: float = 0.0
        self._token_file = f".kite_session_{zerodha_id}.json" if zerodha_id else TOKEN_STORE_FILE
        self._load_stored_token()

    def get_login_url(self) -> str:
        return (
            f"{self._settings.kite_login_url}"
            f"?v=3&api_key={self._user_api_key}"
        )

    async def generate_session(self, request_token: str) -> SessionData:
        api_key = self._user_api_key
        api_secret = self._user_api_secret

        if not api_key or not api_secret:
            raise AuthenticationError("API key and secret are required")

        checksum = hashlib.sha256(
            f"{api_key}{request_token}{api_secret}".encode("utf-8")
        ).hexdigest()

        url = f"{self._settings.kite_base_url}/session/token"
        payload = {
            "api_key": api_key,
            "request_token": request_token,
            "checksum": checksum,
        }

        connector = self._get_ssl_connector()
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(url, data=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    raise AuthenticationError(
                        f"Session generation failed: {text}", response.status
                    )
                result = await response.json()

        if result.get("status") != "success":
            raise AuthenticationError(
                f"Session generation failed: {result.get('message', 'Unknown error')}"
            )

        self._session_data = SessionData(**result["data"])
        self._token_timestamp = time.time()
        self._store_token()

        logger.info(
            "session_generated",
            user_id=self._session_data.user_id,
            user_name=self._session_data.user_name,
        )
        return self._session_data

    @property
    def access_token(self) -> str:
        if self._session_data and self._session_data.access_token:
            return self._session_data.access_token
        settings_token = self._settings.kite_access_token
        if settings_token:
            return settings_token
        raise AuthenticationError("No access token available. Please authenticate first.")

    @property
    def api_key(self) -> str:
        return self._user_api_key

    @property
    def is_authenticated(self) -> bool:
        return bool(self.access_token)

    def is_token_expired(self, max_age_hours: float = 8.0) -> bool:
        if self._token_timestamp == 0:
            return True
        elapsed = time.time() - self._token_timestamp
        return elapsed > (max_age_hours * 3600)

    def get_auth_header(self) -> dict[str, str]:
        return {
            "X-Kite-Version": "3",
            "Authorization": f"token {self.api_key}:{self.access_token}",
        }

    def _store_token(self) -> None:
        if not self._session_data:
            return
        data = {
            "access_token": self._session_data.access_token,
            "public_token": self._session_data.public_token,
            "user_id": self._session_data.user_id,
            "timestamp": self._token_timestamp,
        }
        try:
            with open(self._token_file, "w") as f:
                json.dump(data, f)
            os.chmod(self._token_file, 0o600)
        except OSError as e:
            logger.warning("token_store_failed", error=str(e))

    def _load_stored_token(self) -> None:
        if not os.path.exists(self._token_file):
            return
        try:
            with open(self._token_file) as f:
                data = json.load(f)
            self._session_data = SessionData(
                access_token=data.get("access_token", ""),
                public_token=data.get("public_token", ""),
                user_id=data.get("user_id", ""),
            )
            self._token_timestamp = data.get("timestamp", 0.0)
            logger.info("token_loaded", user_id=self._session_data.user_id)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("token_load_failed", error=str(e))

    def invalidate_session(self) -> None:
        self._session_data = None
        self._token_timestamp = 0.0
        if os.path.exists(self._token_file):
            os.remove(self._token_file)
        logger.info("session_invalidated")

    def _get_ssl_connector(self) -> aiohttp.TCPConnector:
        """Create a TCP connector respecting DISABLE_SSL_VERIFY setting."""
        ssl_context = None
        if self._settings.disable_ssl_verify:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        return aiohttp.TCPConnector(ssl=ssl_context)

    async def invalidate_access_token(self) -> None:
        try:
            url = f"{self._settings.kite_base_url}/session/token"
            headers = self.get_auth_header()
            connector = self._get_ssl_connector()
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.delete(url, headers=headers) as response:
                    if response.status == 200:
                        logger.info("access_token_invalidated")
                    else:
                        text = await response.text()
                        logger.warning("token_invalidation_failed", response=text)
        except Exception as e:
            logger.warning("token_invalidation_error", error=str(e))
        finally:
            self.invalidate_session()
