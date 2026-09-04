#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import os
import re

from fastapi import HTTPException, Request

_TENANT_ID_RE = re.compile(r"^[a-z0-9_-]{1,64}$")


def _check_token(request: Request):
    """Optional token-based auth. Skipped when PERMITTED_KEYS is not set."""
    permitted = os.environ.get("PERMITTED_KEYS", "")
    if not permitted:
        return
    token = request.headers.get("x-token", "")
    if token not in permitted.split(","):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _get_tenant_id(request: Request) -> str:
    """
    Reads the X-Tenant-ID request header and returns a validated tenant identifier.

    Falls back to 'default' when the header is absent, preserving backward
    compatibility with single-tenant deployments.

    Raises HTTP 400 if the provided value contains invalid characters.
    """
    raw = request.headers.get("x-tenant-id", "default").strip().lower()
    if not _TENANT_ID_RE.match(raw):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid X-Tenant-ID '{raw}'. Allowed: lowercase alphanumeric, hyphens, underscores, max 64 chars."
            ),
        )
    return raw
