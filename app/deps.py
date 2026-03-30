from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlmodel import Session, select

from .credits import apply_daily_credit_allowance
from .db import get_session
from .models import User
from .security import SECRET_KEY, ALGORITHM

# Define onde o Swagger (e os clientes) devem buscar o token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciais inválidas ou token expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decodifica o JWT para ver quem é o dono do Token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Busca o usuário no banco para garantir que ele ainda existe
    user = session.exec(select(User).where(User.email == email)).first()
    if user is None:
        raise credentials_exception

    return apply_daily_credit_allowance(user, session)
