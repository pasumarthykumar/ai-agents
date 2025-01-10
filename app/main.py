from fastapi import FastAPI
from .routes.auth_routes import router as auth_router
from app.routes.query_routes import router as query_router

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Domain-Specific AI Assistant"}

# Include authentication routes
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])

# Include Query Routes (Web Agent, Document Agent, etc.)
app.include_router(query_router, prefix="/query", tags=["Query Handlers"])