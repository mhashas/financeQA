from typing import Dict

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> Dict[str, str]:
    """
    Health check endpoint.
    """
    # ideally check the database connection here
    return {"status": "alive"}
