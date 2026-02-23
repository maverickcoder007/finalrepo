from __future__ import annotations

import json
import os
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

USER_CONFIG_FILE = "data/users.json"


class UserConfig:
    def __init__(self, zerodha_id: str, api_key: str, api_secret: str, email: str = "") -> None:
        self.zerodha_id = zerodha_id.upper()
        self.api_key = api_key
        self.api_secret = api_secret
        self.email = email


class UserConfigService:
    _instance: Optional[UserConfigService] = None

    def __init__(self) -> None:
        self._users: dict[str, UserConfig] = {}
        self._load()

    @classmethod
    def get_instance(cls) -> UserConfigService:
        if cls._instance is None:
            cls._instance = UserConfigService()
        return cls._instance

    def _load(self) -> None:
        if os.path.exists(USER_CONFIG_FILE):
            try:
                with open(USER_CONFIG_FILE, "r") as f:
                    data = json.load(f)
                for uid, info in data.items():
                    self._users[uid.upper()] = UserConfig(
                        zerodha_id=uid.upper(),
                        api_key=info.get("api_key", ""),
                        api_secret=info.get("api_secret", ""),
                        email=info.get("email", ""),
                    )
                logger.info("users_loaded", count=len(self._users))
            except Exception as e:
                logger.error("user_config_load_failed", error=str(e))
        else:
            api_key = os.environ.get("KITE_API_KEY", "")
            api_secret = os.environ.get("KITE_API_SECRET", "")
            if api_key:
                default_id = "AC3735"
                self._users[default_id] = UserConfig(
                    zerodha_id=default_id,
                    api_key=api_key,
                    api_secret=api_secret,
                    email="theprotrader007@gmail.com",
                )
                self._save()
                logger.info("default_user_created", zerodha_id=default_id)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(USER_CONFIG_FILE), exist_ok=True)
        try:
            data = {}
            for uid, uc in self._users.items():
                data[uid] = {
                    "api_key": uc.api_key,
                    "api_secret": uc.api_secret,
                    "email": uc.email,
                }
            with open(USER_CONFIG_FILE, "w") as f:
                json.dump(data, f, indent=2)
            os.chmod(USER_CONFIG_FILE, 0o600)
        except Exception as e:
            logger.error("user_config_save_failed", error=str(e))

    def get_user(self, zerodha_id: str) -> Optional[UserConfig]:
        return self._users.get(zerodha_id.upper())

    def validate_user(self, zerodha_id: str) -> bool:
        return zerodha_id.upper() in self._users

    def add_user(self, zerodha_id: str, api_key: str, api_secret: str, email: str = "") -> UserConfig:
        uc = UserConfig(zerodha_id=zerodha_id, api_key=api_key, api_secret=api_secret, email=email)
        self._users[uc.zerodha_id] = uc
        self._save()
        logger.info("user_added", zerodha_id=uc.zerodha_id)
        return uc

    def remove_user(self, zerodha_id: str) -> bool:
        uid = zerodha_id.upper()
        if uid in self._users:
            del self._users[uid]
            self._save()
            return True
        return False

    def list_users(self) -> list[dict[str, str]]:
        return [{"zerodha_id": uc.zerodha_id, "email": uc.email} for uc in self._users.values()]
