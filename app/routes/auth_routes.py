from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.auth import hash_password, verify_password, create_access_token
from app.auth_models import UserRegister
from app.models import User
from app.database import get_db
from app.auth import verify_password, create_access_token
from app.auth_models import UserLogin, Token

# Initialize the router
router = APIRouter()

# Register endpoint
@router.post("/auth/register/")
async def register_user(user: UserRegister, db: Session = Depends(get_db)):
    # Check if the username already exists in the database
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Hash the password
    hashed_password = hash_password(user.password)

    # Create and save the new user in the database
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User registered successfully"}


# Login endpoint
@router.post("/auth/login/", response_model=Token)
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    # Fetch the user from the database
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Verify the password
    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Create a JWT token
    access_token = create_access_token(data={"sub": db_user.username})
    return {"access_token": access_token, "token_type": "bearer"}

