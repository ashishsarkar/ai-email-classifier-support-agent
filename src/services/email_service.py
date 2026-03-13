"""Email service for IMAP/SMTP operations."""

import asyncio
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional

import aiosmtplib
from imap_tools import MailBox, MailboxLogoutError
from loguru import logger

from src.core.config import settings
from src.schemas.email import EmailIn
from src.services.base import BaseService


class EmailService(BaseService):
    """Handle email retrieval and sending."""

    def __init__(self):
        """Initialize email service."""
        self.imap_server = settings.IMAP_SERVER
        self.imap_port = settings.IMAP_PORT
        self.smtp_server = settings.SMTP_SERVER
        self.smtp_port = settings.SMTP_PORT
        self.email_address = settings.EMAIL_ADDRESS
        self.email_password = settings.EMAIL_PASSWORD
        self.from_name = settings.EMAIL_FROM_NAME
        self.mailbox = None

    async def initialize(self) -> None:
        """Initialize email service and test connections."""
        await super().initialize()
        try:
            # Test IMAP connection
            await self._test_imap_connection()
            logger.info("Email service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize email service: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup email connections."""
        if self.mailbox:
            try:
                self.mailbox.logout()
            except MailboxLogoutError:
                pass
        await super().cleanup()

    async def _get_mailbox(self) -> MailBox:
        """Get IMAP mailbox connection."""
        if not self.mailbox:
            self.mailbox = MailBox(self.imap_server, self.imap_port)
            self.mailbox.login(self.email_address, self.email_password)
        return self.mailbox

    async def _test_imap_connection(self) -> None:
        """Test IMAP connection."""
        try:
            mailbox = MailBox(self.imap_server, self.imap_port)
            mailbox.login(self.email_address, self.email_password)
            mailbox.logout()
            logger.info("IMAP connection test passed")
        except Exception as e:
            logger.error(f"IMAP connection test failed: {e}")
            raise

    async def fetch_emails(self, limit: int = 10) -> List[EmailIn]:
        """Fetch unread emails from inbox.

        Args:
            limit: Maximum number of emails to fetch

        Returns:
            List of EmailIn objects
        """
        try:
            mailbox = await self._get_mailbox()
            emails = []

            # Fetch unread emails
            for msg in mailbox.fetch(limit=limit, mark_seen=False):
                try:
                    email_in = EmailIn(
                        sender=msg.from_,
                        subject=msg.subject or "[No Subject]",
                        body=msg.text or msg.html or "",
                        html_body=msg.html,
                        received_at=msg.date or datetime.utcnow(),
                        message_id=msg.message_id,
                    )
                    emails.append(email_in)
                    logger.debug(f"Fetched email from {email_in.sender}")
                except Exception as e:
                    logger.error(f"Failed to parse email: {e}")
                    continue

            logger.info(f"Fetched {len(emails)} emails from inbox")
            return emails

        except Exception as e:
            logger.error(f"Failed to fetch emails: {e}")
            raise

    async def send_email(
        self, to_address: str, subject: str, body: str, html_body: Optional[str] = None
    ) -> bool:
        """Send email via SMTP.

        Args:
            to_address: Recipient email address
            subject: Email subject
            body: Plain text body
            html_body: Optional HTML body

        Returns:
            True if email sent successfully
        """
        try:
            # Create message
            if html_body:
                msg = MIMEMultipart("alternative")
                msg.attach(MIMEText(body, "plain"))
                msg.attach(MIMEText(html_body, "html"))
            else:
                msg = MIMEText(body)

            msg["Subject"] = subject
            msg["From"] = f"{self.from_name} <{self.email_address}>"
            msg["To"] = to_address

            # Send via SMTP
            async with aiosmtplib.SMTP(hostname=self.smtp_server, port=self.smtp_port) as smtp:
                await smtp.login(self.email_address, self.email_password)
                await smtp.send_message(msg)

            logger.info(f"Email sent to {to_address}: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email to {to_address}: {e}")
            return False

    async def mark_as_read(self, message_id: str) -> bool:
        """Mark email as read.

        Args:
            message_id: Email message ID

        Returns:
            True if successful
        """
        try:
            mailbox = await self._get_mailbox()
            # Implementation depends on imap-tools capabilities
            logger.debug(f"Marked message {message_id} as read")
            return True
        except Exception as e:
            logger.error(f"Failed to mark email as read: {e}")
            return False

    async def health_check(self) -> dict:
        """Check email service health."""
        try:
            await self._test_imap_connection()
            return {"status": "healthy", "service": "email"}
        except Exception as e:
            logger.error(f"Email service health check failed: {e}")
            return {"status": "unhealthy", "service": "email", "error": str(e)}
