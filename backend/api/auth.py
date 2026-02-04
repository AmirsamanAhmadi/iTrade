import os
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

security = HTTPBasic()

def get_basic_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> bool:
    """Validate provided basic credentials against environment variables.
    Returns True if valid, else raises HTTPException(401).
    """
    user_env = os.getenv('UI_API_USER', 'admin')
    pass_env = os.getenv('UI_API_PASS', 'admin')

    correct_username = secrets.compare_digest(credentials.username, user_env)
    correct_password = secrets.compare_digest(credentials.password, pass_env)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')
    return True
