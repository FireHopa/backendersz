import os
import httpx
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from .db import get_session
from .models import User
from .schemas import UserCreate, UserLogin, GoogleAuth, Token
from .security import get_password_hash, verify_password, create_access_token
from .deps import get_current_user

router = APIRouter(prefix="/api/auth", tags=["Autenticação"])

# NOVO: Função para verificar e renovar créditos diários
def check_and_reset_credits(user: User, session: Session) -> User:
    now = datetime.now(timezone.utc)
    # Se a data do último reset for menor que o dia de hoje, restaura os 100 créditos
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
        credits=100 # Inicia com 100
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    
    access_token = create_access_token(data={"sub": new_user.email})
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "user_email": new_user.email, 
        "user_name": new_user.full_name,
        "credits": new_user.credits,
        "has_linkedin": bool(new_user.linkedin_urn)
    }

@router.post("/login", response_model=Token)
def login(user_in: UserLogin, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == user_in.email)).first()
    if not user or not user.hashed_password:
        raise HTTPException(status_code=400, detail="E-mail ou senha incorretos.")
        
    if not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="E-mail ou senha incorretos.")
        
    user = check_and_reset_credits(user, session)
        
    access_token = create_access_token(data={"sub": user.email})
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "user_email": user.email, 
        "user_name": user.full_name,
        "credits": user.credits,
        "has_linkedin": bool(user.linkedin_urn)
    }

@router.post("/google", response_model=Token)
def google_auth(auth_in: GoogleAuth, session: Session = Depends(get_session)):
    try:
        response = httpx.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {auth_in.credential}"}
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
            # Conta totalmente nova
            user = User(email=email, full_name=name, google_id=google_id, credits=100)
            session.add(user)
            session.commit()
            session.refresh(user)
        else:
            if not user.google_id:
                user.google_id = google_id
                session.add(user)
                session.commit()
            
            # Conta existente, checa o reset
            user = check_and_reset_credits(user, session)
            
        access_token = create_access_token(data={"sub": user.email})
        return {
            "access_token": access_token, 
            "token_type": "bearer", 
            "user_email": user.email, 
            "user_name": user.full_name,
            "credits": user.credits,
            "has_linkedin": bool(user.linkedin_urn)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro de comunicação com o Google: {str(e)}")

@router.get("/me")
def get_me(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """Rota para o Frontend testar se o token ainda é válido e buscar créditos atualizados"""
    
    # Fazemos a checagem no /me também para garantir que se o user deixou
    # a tab aberta da noite pro dia, ao recarregar a página os créditos renovem.
    current_user = check_and_reset_credits(current_user, session)
    
    return {
        "email": current_user.email,
        "full_name": current_user.full_name,
        "google_id": current_user.google_id,
        "credits": current_user.credits,
        "has_linkedin": bool(current_user.linkedin_urn)
    }