"""Router de autenticación y sesiones (M01)."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from app.core.deps import AuthContext, RequestContext, get_auth_context, get_context, get_db, require_permissions
from app.schemas.auth import (
    ChangePasswordRequest,
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    LoginRequest,
    MeResponse,
    RefreshRequest,
    ResetPasswordRequest,
    SessionInfo,
    TokenPair,
)
from app.schemas.common import Message
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["M01 Autenticación"])


@router.post("/login", response_model=TokenPair, summary="Iniciar sesión")
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_context)) -> TokenPair:
    result = auth_service.login(
        db,
        email=payload.email,
        password=payload.password,
        ip_address=ctx.ip_address,
        user_agent=ctx.user_agent,
        request_id=ctx.request_id,
    )
    return TokenPair(
        access_token=result["access_token"],
        refresh_token=result["refresh_token"],
        expires_in=int(result["expires_in"]),
        must_change_password=bool(result["must_change_password"]),
    )


@router.post("/refresh", response_model=TokenPair, summary="Renovar token de acceso")
def refresh(payload: RefreshRequest, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_context)) -> TokenPair:
    result = auth_service.refresh(db, payload.refresh_token, ip_address=ctx.ip_address, user_agent=ctx.user_agent)
    return TokenPair(
        access_token=result["access_token"],
        refresh_token=result["refresh_token"],
        expires_in=int(result["expires_in"]),
    )


@router.post("/logout", response_model=Message, summary="Cerrar sesión")
def logout(db: Session = Depends(get_db), auth: AuthContext = Depends(get_auth_context), ctx: RequestContext = Depends(get_context)) -> Message:
    auth_service.logout(db, auth.session_id, user=auth.user, ip_address=ctx.ip_address, user_agent=ctx.user_agent)
    return Message(message="Sesión cerrada correctamente.")


@router.get("/me", response_model=MeResponse, summary="Perfil del usuario autenticado")
def me(auth: AuthContext = Depends(get_auth_context)) -> MeResponse:
    user = auth.user
    return MeResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        position=user.position,
        organization=user.organization,
        roles=[{"slug": role.slug, "name": role.name} for role in user.roles],
        permissions=user.permission_codes,
        is_superuser=user.is_superuser,
        must_change_password=user.must_change_password,
        last_login_at=user.last_login_at,
    )


@router.post("/change-password", response_model=Message, summary="Cambiar contraseña")
def change_password(
    payload: ChangePasswordRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Message:
    auth_service.change_password(
        db, auth.user, payload.current_password, payload.new_password, user_role=",".join(auth.user.role_slugs)
    )
    return Message(message="Contraseña actualizada correctamente.")


@router.post("/forgot-password", response_model=ForgotPasswordResponse, summary="Solicitar restablecimiento")
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)) -> ForgotPasswordResponse:
    result = auth_service.request_password_reset(db, payload.email)
    return ForgotPasswordResponse(**result)


@router.post("/reset-password", response_model=Message, summary="Restablecer contraseña con token")
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)) -> Message:
    auth_service.reset_password(db, payload.token, payload.new_password)
    return Message(message="Contraseña restablecida. Inicie sesión con la nueva contraseña.")


@router.get("/sessions", response_model=List[SessionInfo], summary="Sesiones activas del usuario")
def list_sessions(db: Session = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> List[SessionInfo]:
    return [SessionInfo(**row) for row in auth_service.list_sessions(db, auth.user, auth.session_id)]


@router.delete("/sessions/{public_id}", response_model=Message, summary="Revocar una sesión")
def revoke_session(public_id: str, db: Session = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> Message:
    auth_service.revoke_session(db, auth.user, public_id)
    return Message(message="Sesión revocada.")
