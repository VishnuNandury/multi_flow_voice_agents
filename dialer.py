"""
dialer.py — Provider-agnostic outbound call initiator.

Supports Twilio and Exotel. Returns the provider call SID on success.
"""

import os

import aiohttp
from loguru import logger


# ---------------------------------------------------------------------------
# Twilio
# ---------------------------------------------------------------------------

async def initiate_twilio_call(
    to: str,
    from_: str,
    webhook_url: str,
    status_url: str,
    creds: dict,
) -> str:
    """
    Initiate a Twilio outbound call.

    creds: {"account_sid": ..., "auth_token": ...}
    Returns: call SID string
    """
    account_sid = creds.get("account_sid") or os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token = creds.get("auth_token") or os.getenv("TWILIO_AUTH_TOKEN", "")

    if not account_sid or not auth_token:
        raise ValueError("Twilio credentials (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) not configured")

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json"
    payload = {
        "To": to,
        "From": from_,
        "Url": webhook_url,
        "StatusCallback": status_url,
        "StatusCallbackMethod": "POST",
        "StatusCallbackEvent": ["completed", "failed", "no-answer", "busy"],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            data=payload,
            auth=aiohttp.BasicAuth(account_sid, auth_token),
        ) as resp:
            body = await resp.json()
            if resp.status not in (200, 201):
                raise RuntimeError(f"Twilio call failed ({resp.status}): {body}")
            sid = body.get("sid", "")
            logger.info(f"Twilio call initiated: SID={sid}, to={to}")
            return sid


# ---------------------------------------------------------------------------
# Exotel
# ---------------------------------------------------------------------------

async def initiate_exotel_call(
    to: str,
    from_: str,
    webhook_url: str,
    status_url: str,
    creds: dict,
) -> str:
    """
    Initiate an Exotel outbound call.

    creds: {"api_key": ..., "api_token": ..., "sid": ..., "subdomain": ..., "phone_number": ...}
    Returns: call SID string
    """
    api_key = creds.get("api_key") or os.getenv("EXOTEL_API_KEY", "")
    api_token = creds.get("api_token") or os.getenv("EXOTEL_API_TOKEN", "")
    exotel_sid = creds.get("sid") or os.getenv("EXOTEL_SID", "")
    subdomain = creds.get("subdomain") or os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")

    if not api_key or not api_token or not exotel_sid:
        raise ValueError("Exotel credentials (EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_SID) not configured")

    url = f"https://{api_key}:{api_token}@{subdomain}/v1/Accounts/{exotel_sid}/Calls/connect.json"
    payload = {
        "From": to,
        "To": from_,  # Exotel: From=customer, To=agent/number
        "CallerId": from_,
        "Url": webhook_url,
        "StatusCallback": status_url,
        "StatusCallbackContentType": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=payload) as resp:
            body = await resp.json()
            if resp.status not in (200, 201):
                raise RuntimeError(f"Exotel call failed ({resp.status}): {body}")
            call_data = body.get("Call", {})
            sid = call_data.get("Sid", "")
            logger.info(f"Exotel call initiated: SID={sid}, to={to}")
            return sid


# ---------------------------------------------------------------------------
# Provider router
# ---------------------------------------------------------------------------

async def initiate_call(
    provider: str,
    to: str,
    from_: str,
    webhook_url: str,
    status_url: str,
    creds: dict = None,
) -> str:
    """
    Route to the correct provider and initiate an outbound call.

    provider: "twilio" or "exotel"
    Returns: call SID string
    """
    creds = creds or {}
    if provider == "twilio":
        return await initiate_twilio_call(to, from_, webhook_url, status_url, creds)
    elif provider == "exotel":
        return await initiate_exotel_call(to, from_, webhook_url, status_url, creds)
    else:
        raise ValueError(f"Unsupported provider: {provider!r}. Choose 'twilio' or 'exotel'.")
