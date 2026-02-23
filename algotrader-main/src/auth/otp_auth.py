from __future__ import annotations

import hashlib
import json
import os
import secrets
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from dotenv import load_dotenv

from src.utils.logger import get_logger
from src.utils.exceptions import AuthenticationError

# Load .env file explicitly before reading environment variables
load_dotenv()

logger = get_logger(__name__)

# Security constants
OTP_EXPIRY_SECONDS = 300  # 5 minutes
SESSION_EXPIRY_SECONDS = 86400 * 7  # 7 days
MAX_OTP_ATTEMPTS = 3  # Max incorrect OTP attempts
MAX_OTP_REQUESTS_PER_HOUR = 5  # Prevent OTP spam
SESSION_FILE = os.path.expanduser("~/.kta_sessions.json")  # Home directory, not /tmp


class OTPAuth:
    def __init__(self) -> None:
        self._pending_otps: dict[str, dict] = {}
        self._sessions: dict[str, dict] = self._load_sessions()
        self._otp_request_count: dict[str, list[float]] = {}  # Track OTP requests per user
        
        # Load ADMIN_EMAIL from environment (loaded by dotenv.load_dotenv above)
        self._admin_email = os.environ.get("ADMIN_EMAIL", "")
        if not self._admin_email:
            logger.warning("security_warning", msg="ADMIN_EMAIL not configured - OTP emails will not be sent")
            self._admin_email = "admin@example.com"  # Fallback to prevent crashes
        
        # SESSION_SECRET is REQUIRED in production
        self._secret = os.environ.get("SESSION_SECRET", "")
        if not self._secret:
            logger.warning("security_warning", msg="SESSION_SECRET not set - sessions will be invalidated on server restart")
            self._secret = secrets.token_hex(32)
        
        self._smtp_email = os.environ.get("SMTP_EMAIL", "")
        self._smtp_password = os.environ.get("SMTP_PASSWORD", "")
        self._smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self._smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        
        # Secure session file permissions
        self._secure_session_file()
        logger.info("otp_auth_initialized", admin_email=self._admin_email[:3] + "***")

    def _secure_session_file(self) -> None:
        """Ensure session file has secure permissions."""
        if os.path.exists(SESSION_FILE):
            try:
                os.chmod(SESSION_FILE, 0o600)  # Read/write owner only
            except Exception as e:
                logger.warning("session_file_chmod_failed", error=str(e))

    def _load_sessions(self) -> dict[str, dict]:
        """Load and validate sessions from file, removing expired ones."""
        try:
            if os.path.exists(SESSION_FILE):
                with open(SESSION_FILE, "r") as f:
                    sessions = json.load(f)
                now = time.time()
                # Filter expired sessions
                valid = {k: v for k, v in sessions.items() if now - v.get("created_at", 0) < SESSION_EXPIRY_SECONDS}
                if len(valid) < len(sessions):
                    self._sessions = valid
                    self._save_sessions()
                return valid
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("session_load_failed", error=str(e))
        return {}

    def _save_sessions(self) -> None:
        """Save sessions to file with secure permissions."""
        try:
            # Create parent directory if needed
            os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
            # Write with restricted permissions
            with open(SESSION_FILE, "w") as f:
                json.dump(self._sessions, f)
            os.chmod(SESSION_FILE, 0o600)  # Read/write owner only
        except OSError as e:
            logger.error("session_save_failed", error=str(e))

    def generate_otp(self, zerodha_id: str = "") -> str:
        """Generate OTP with rate limiting.
        
        Args:
            zerodha_id: Zerodha ID to generate OTP for
        
        Returns:
            6-digit OTP string
        
        Raises:
            AuthenticationError: If rate limit exceeded
        """
        now = time.time()
        hour_ago = now - 3600
        
        # Check rate limit: max 5 requests per hour per user
        if zerodha_id not in self._otp_request_count:
            self._otp_request_count[zerodha_id] = []
        
        # Remove old requests outside 1-hour window
        self._otp_request_count[zerodha_id] = [
            t for t in self._otp_request_count[zerodha_id] if t > hour_ago
        ]
        
        if len(self._otp_request_count[zerodha_id]) >= MAX_OTP_REQUESTS_PER_HOUR:
            logger.warning("otp_rate_limit_exceeded", zerodha_id=zerodha_id)
            raise AuthenticationError("Too many OTP requests. Try again later.")
        
        self._otp_request_count[zerodha_id].append(now)
        
        # Generate cryptographically secure OTP
        otp = str(secrets.randbelow(900000) + 100000)
        key = zerodha_id.upper()
        self._pending_otps[key] = {
            "otp": otp,
            "created_at": now,
            "attempts": 0,
            "zerodha_id": zerodha_id.upper(),
        }
        logger.info("otp_generated", zerodha_id=zerodha_id)
        return otp

    def send_otp_email(self, otp: str, zerodha_id: str = "") -> bool:
        """Send OTP email to configured admin email.
        
        Args:
            otp: OTP code to send
            zerodha_id: Zerodha ID (for logging only)
        
        Returns:
            True if sent successfully
        """
        if not self._smtp_email or not self._smtp_password:
            logger.error("smtp_not_configured", msg="SMTP credentials not configured")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"Kite Trading Agent - Login OTP: {otp}"
            msg["From"] = self._smtp_email
            msg["To"] = self._admin_email

            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background: #0d1821; color: #e0e6ed; padding: 30px;">
                <div style="max-width: 400px; margin: 0 auto; background: #1a2332; border-radius: 12px; padding: 30px; border: 1px solid #2a3a4a;">
                    <h2 style="color: #00d4aa; margin-top: 0;">Kite Trading Agent</h2>
                    <p>Your one-time login code is:</p>
                    <div style="font-size: 36px; font-weight: bold; color: #00d4aa; text-align: center; padding: 20px; background: #0d1821; border-radius: 8px; letter-spacing: 8px; margin: 20px 0;">
                        {otp}
                    </div>
                    <p style="color: #8899aa; font-size: 13px;">This code expires in 5 minutes. Do not share it with anyone.</p>
                </div>
            </body>
            </html>
            """
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.starttls()
                server.login(self._smtp_email, self._smtp_password)
                server.sendmail(self._smtp_email, self._admin_email, msg.as_string())

            logger.info("otp_email_sent", zerodha_id=zerodha_id)
            return True
        except Exception as e:
            logger.error("otp_email_failed", zerodha_id=zerodha_id, error=str(e))
            return False

    def verify_otp(self, otp: str, zerodha_id: str = "") -> Optional[str]:
        """Verify OTP with attempt limiting.
        
        Args:
            otp: OTP code to verify
            zerodha_id: Associated Zerodha ID
        
        Returns:
            Session token if valid, None otherwise
        """
        key = zerodha_id.upper()
        pending = self._pending_otps.get(key)
        if not pending:
            logger.warning("otp_verify_no_pending", zerodha_id=zerodha_id)
            return None

        now = time.time()
        if now - pending["created_at"] > OTP_EXPIRY_SECONDS:
            del self._pending_otps[key]
            logger.warning("otp_expired", zerodha_id=zerodha_id)
            return None

        # Increment attempt counter
        pending["attempts"] += 1
        if pending["attempts"] > MAX_OTP_ATTEMPTS:
            del self._pending_otps[key]
            logger.warning("otp_max_attempts_exceeded", zerodha_id=zerodha_id, attempts=pending["attempts"])
            return None

        # Validate OTP (timing-safe comparison)
        if not secrets.compare_digest(pending["otp"], otp):
            logger.warning("otp_invalid", zerodha_id=zerodha_id, attempt=pending["attempts"])
            return None

        stored_zid = pending.get("zerodha_id", "")
        del self._pending_otps[key]
        session_token = self._create_session(stored_zid)
        logger.info("otp_verified", zerodha_id=zerodha_id)
        return session_token

    def _create_session(self, zerodha_id: str = "") -> str:
        """Create cryptographically secure session token."""
        token = secrets.token_urlsafe(32)  # URL-safe base64 token
        self._sessions[token] = {
            "zerodha_id": zerodha_id.upper(),
            "created_at": time.time(),
        }
        self._save_sessions()
        return token

    def validate_session(self, token: str) -> bool:
        if not token:
            return False
        session = self._sessions.get(token)
        if not session:
            return False
        if time.time() - session["created_at"] > SESSION_EXPIRY_SECONDS:
            del self._sessions[token]
            return False
        return True

    def get_session_user(self, token: str) -> Optional[str]:
        if not token:
            return None
        session = self._sessions.get(token)
        if not session:
            return None
        if time.time() - session["created_at"] > SESSION_EXPIRY_SECONDS:
            return None
        return session.get("zerodha_id", "")

    def logout(self, token: str) -> None:
        self._sessions.pop(token, None)
        self._save_sessions()


otp_auth = OTPAuth()
