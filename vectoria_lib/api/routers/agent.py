import logging

from fastapi import APIRouter, Response, status

logger = logging.getLogger("api")

router = APIRouter(tags=["agent"])


@router.get("/health")
async def health(response: Response):
    response.status_code = status.HTTP_200_OK
    return {"status": "OK"}
