from __future__ import annotations

from enum import Enum
from typing import Optional


class ErrorCategory(str, Enum):
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    ORDER = "order"
    DATA = "data"
    RISK = "risk"
    STRATEGY = "strategy"
    SYSTEM = "system"


class KiteError(Exception):
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
    ) -> None:
        self.message = message
        self.category = category
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [f"[{self.category.value}] {self.message}"]
        if self.status_code:
            parts.append(f"(HTTP {self.status_code})")
        if self.error_type:
            parts.append(f"Type: {self.error_type}")
        return " | ".join(parts)


class AuthenticationError(KiteError):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message, ErrorCategory.AUTHENTICATION, status_code)


class NetworkError(KiteError):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message, ErrorCategory.NETWORK, status_code)


class RateLimitError(KiteError):
    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, ErrorCategory.RATE_LIMIT, 429)


class OrderError(KiteError):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message, ErrorCategory.ORDER, status_code)


class DataError(KiteError):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message, ErrorCategory.DATA, status_code)


class RiskLimitError(KiteError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCategory.RISK)


class KillSwitchError(KiteError):
    def __init__(self, message: str = "Kill switch activated") -> None:
        super().__init__(message, ErrorCategory.RISK)


class StrategyError(KiteError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCategory.STRATEGY)
