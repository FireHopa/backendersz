import httpx
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from .db import get_session
from .models import User
from .schemas import UserCreate, UserLogin, GoogleAuth, Token
from .security import get_password_hash, verify_password, create_access_token
from .deps import get_current_user

router = APIRouter(prefix="/api/auth", tags=["Autenticação"])


def _token_payload(user: User, access_token: str) -> dict:
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_email": user.email,
        "user_name": user.full_name,
        "credits": user.credits,
        "has_linkedin": bool(user.linkedin_urn),
        "has_instagram": bool(user.instagram_account_id and user.instagram_meta_access_token),
        "instagram_username": user.instagram_username,
        "has_facebook": bool(user.facebook_page_id and user.facebook_page_access_token),
        "facebook_page_name": user.facebook_page_name,
        "facebook_page_username": user.facebook_page_username,
        "has_youtube": bool(user.youtube_channel_id and user.youtube_refresh_token),
        "youtube_channel_title": user.youtube_channel_title,
        "youtube_channel_handle": user.youtube_channel_handle,
        "has_tiktok": bool(user.tiktok_open_id and user.tiktok_refresh_token),
        "tiktok_display_name": user.tiktok_display_name,
        "tiktok_username": user.tiktok_username,
        "has_google_business_profile": bool(user.google_business_refresh_token),
        "google_business_account_display_name": user.google_business_account_display_name,
        "google_business_location_title": user.google_business_location_title,
    }


def check_and_reset_credits(user: User, session: Session) -> User:
    now = datetime.now(timezone.utc)
    if user.last_credit_reset.date() < now.date():
        user.credits = 100
        user.last_credit_reset = now
        session.add(user)
        session.commit()
        session.refresh(user)
    return user


@router.post("/register", response_model=Token)
def register(user_in: UserCreate, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == user_in.email)).first()
    if user:
        raise HTTPException(status_code=400, detail="Este e-mail já está em uso.")

    new_user = User(
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name,
        credits=100,
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    access_token = create_access_token(data={"sub": new_user.email})
    return _token_payload(new_user, access_token)


@router.post("/login", response_model=Token)
def login(user_in: UserLogin, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == user_in.email)).first()
    if not user or not user.hashed_password:
        raise HTTPException(status_code=400, detail="E-mail ou senha incorretos.")

    if not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="E-mail ou senha incorretos.")

    user = check_and_reset_credits(user, session)
    access_token = create_access_token(data={"sub": user.email})
    return _token_payload(user, access_token)


@router.post("/google", response_model=Token)
def google_auth(auth_in: GoogleAuth, session: Session = Depends(get_session)):
    try:
        response = httpx.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {auth_in.credential}"},
        )

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Token do Google inválido ou expirado.")

        idinfo = response.json()
        email = idinfo.get("email")
        name = idinfo.get("name")
        google_id = idinfo.get("sub")

        if not email:
            raise HTTPException(status_code=400, detail="Token do Google não contém e-mail válido.")

        user = session.exec(select(User).where(User.email == email)).first()
        if not user:
            user = User(email=email, full_name=name, google_id=google_id, credits=100)
            session.add(user)
            session.commit()
            session.refresh(user)
        else:
            if not user.google_id:
                user.google_id = google_id
                session.add(user)
                session.commit()
            user = check_and_reset_credits(user, session)

        access_token = create_access_token(data={"sub": user.email})
        return _token_payload(user, access_token)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro de comunicação com o Google: {str(e)}")


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    current_user = check_and_reset_credits(current_user, session)
    return {
        "email": current_user.email,
        "full_name": current_user.full_name,
        "google_id": current_user.google_id,
        "credits": current_user.credits,
        "has_linkedin": bool(current_user.linkedin_urn),
        "has_instagram": bool(current_user.instagram_account_id and current_user.instagram_meta_access_token),
        "instagram_username": current_user.instagram_username,
        "has_facebook": bool(current_user.facebook_page_id and current_user.facebook_page_access_token),
        "facebook_page_name": current_user.facebook_page_name,
        "facebook_page_username": current_user.facebook_page_username,
        "has_youtube": bool(current_user.youtube_channel_id and current_user.youtube_refresh_token),
        "youtube_channel_title": current_user.youtube_channel_title,
        "youtube_channel_handle": current_user.youtube_channel_handle,
        "has_tiktok": bool(current_user.tiktok_open_id and current_user.tiktok_refresh_token),
        "tiktok_display_name": current_user.tiktok_display_name,
        "tiktok_username": current_user.tiktok_username,
        "has_google_business_profile": bool(current_user.google_business_refresh_token),
        "google_business_account_display_name": current_user.google_business_account_display_name,
        "google_business_location_title": current_user.google_business_location_title,
    }
