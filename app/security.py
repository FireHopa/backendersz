import os
from datetime import datetime, timedelta, timezone
from typing import Optional
import bcrypt
from jose import jwt

# Configurações do JWT
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "uma-chave-secreta-muito-longa-e-dificil-de-adivinhar-12345")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 dias de duração

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica se a senha em texto plano corresponde ao hash."""
    # Bcrypt exige bytes, então limitamos a 72 caracteres e codificamos
    pwd_bytes = plain_password[:72].encode('utf-8')
    hash_bytes = hashed_password.encode('utf-8')
    try:
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except ValueError:
        return False

def get_password_hash(password: str) -> str:
    """Gera o hash de uma senha."""
    # Limita silenciosamente a senha a 72 caracteres (limite do bcrypt) para não dar erro
    pwd_bytes = password[:72].encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(pwd_bytes, salt)
    return hashed_password.decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Cria um token JWT para o usuário logado."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt