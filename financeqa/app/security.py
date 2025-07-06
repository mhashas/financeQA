import os

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="Authorization", auto_error=True)


def get_current_api_key(api_key: str = Depends(api_key_header)):
    """Validate the API key from the request header against the expected API key

    Args:
        api_key (str): api key from the request header

    Raises:
        HTTPException: If the API key is invalid
    """
    expected_key = os.environ.get("FINANCEQA_API_KEY")

    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
