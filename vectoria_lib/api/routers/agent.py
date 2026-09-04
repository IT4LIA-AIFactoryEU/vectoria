#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import logging

from fastapi import APIRouter, Response, status

logger = logging.getLogger("api")

router = APIRouter(tags=["agent"])


@router.get("/health")
async def health(response: Response):
    response.status_code = status.HTTP_200_OK
    return {"status": "OK"}
