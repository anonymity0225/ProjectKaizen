from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os
from pydantic import BaseModel

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()


class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None


class User(BaseModel):
    user_id: str
    username: str
    email: Optional[str] = None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify JWT token from Authorization header.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        User: User information from token
        
    Raises:
        HTTPException: 401 if token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        
        if username is None or user_id is None:
            raise credentials_exception
            
        token_data = TokenData(username=username, user_id=user_id)
        
    except JWTError:
        raise credentials_exception
    
    # In a real application, you would fetch user from database
    # For now, return user info from token
    user = User(
        user_id=token_data.user_id,
        username=token_data.username,
        email=payload.get("email")
    )
    
    return user


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get current authenticated user.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        User: Current user information
    """
    return verify_token(credentials)


# Optional: Add user authentication function
def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate user with username and password.
    In production, this should check against a database.
    
    Args:
        username: User's username
        password: User's password
        
    Returns:
        User: User object if authentication successful, None otherwise
    """
    # This is a placeholder - in production, check against database
    # and verify password hash
    if username == "admin" and password == "password":  # Demo only!
        return User(user_id="1", username="admin", email="admin@example.com")
    return None


def create_user_token(user: User) -> str:
    """
    Create access token for authenticated user.
    
    Args:
        user: User object
        
    Returns:
        str: JWT access token
    """
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.user_id,
            "email": user.email
        },
        expires_delta=access_token_expires
    )
    return access_token