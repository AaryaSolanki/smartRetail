"""
Minimal API-key header auth, as suggested by C4:
"Add basic auth or API key header for the endpoints to simulate production security."

Set the real key via the API_KEY environment variable in production.
If API_KEY is unset (e.g. local dev in Colab), auth is disabled so you can test freely.
"""
import os
from fastapi import Header, HTTPException

API_KEY = os.environ.get("API_KEY")  # leave unset locally to skip auth


async def verify_api_key(x_api_key: str = Header(default=None)):
    if API_KEY is None:
        return  # auth disabled (no key configured) — fine for dev/Colab
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
