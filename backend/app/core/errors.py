"""Excepciones de dominio y manejadores uniformes de error."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException, status


class AppError(Exception):
    """Error base con código estable y mensaje apto para el cliente."""

    code: str = "APP_ERROR"
    status_code: int = status.HTTP_400_BAD_REQUEST

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class NotFoundError(AppError):
    code = "NOT_FOUND"
    status_code = status.HTTP_404_NOT_FOUND


class ValidationError(AppError):
    code = "VALIDATION_ERROR"
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY


class ConflictError(AppError):
    code = "CONFLICT"
    status_code = status.HTTP_409_CONFLICT


class AuthError(AppError):
    code = "AUTH_ERROR"
    status_code = status.HTTP_401_UNAUTHORIZED


class PermissionError_(AppError):
    code = "ACCESS_DENIED"
    status_code = status.HTTP_403_FORBIDDEN


class RateLimitError(AppError):
    code = "TOO_MANY_REQUESTS"
    status_code = status.HTTP_429_TOO_MANY_REQUESTS


class ServiceUnavailableError(AppError):
    code = "SERVICE_UNAVAILABLE"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


def to_http(exc: AppError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.to_dict()["error"])
