from __future__ import annotations

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    kite_api_key: str = Field(default="", description="Kite Connect API key")
    kite_api_secret: str = Field(default="", description="Kite Connect API secret")
    kite_access_token: str = Field(default="", description="Kite Connect access token")
    kite_request_token: str = Field(default="", description="Kite Connect request token")

    kite_base_url: str = Field(default="https://api.kite.trade", description="Kite API base URL")
    kite_login_url: str = Field(default="https://kite.zerodha.com/connect/login", description="Kite login URL")
    kite_ws_url: str = Field(default="wss://ws.kite.trade", description="Kite WebSocket URL")

    max_daily_loss: float = Field(default=10000.0, description="Maximum daily loss in INR")
    max_position_size: int = Field(default=100, description="Maximum position size per instrument")
    max_exposure: float = Field(default=500000.0, description="Maximum total exposure in INR")
    default_stop_loss_pct: float = Field(default=2.0, description="Default stop loss percentage")
    kill_switch_loss: float = Field(default=25000.0, description="Kill switch threshold in INR")

    rate_limit_default: float = Field(default=8.0, description="Default requests per second")
    rate_limit_quote: float = Field(default=1.0, description="Quote requests per second")
    rate_limit_historical: float = Field(default=3.0, description="Historical data requests per second")
    rate_limit_orders: float = Field(default=8.0, description="Order requests per second")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Base retry delay in seconds")

    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/trading.log", description="Log file path")
    trade_journal_file: str = Field(default="logs/trade_journal.json", description="Trade journal path")

    backtest_data_dir: str = Field(default="data/historical", description="Backtest data directory")
    instruments_cache_file: str = Field(default="data/instruments.csv", description="Instruments cache path")
    disable_ssl_verify: bool = Field(default=False, description="Disable SSL verification (development only)")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
