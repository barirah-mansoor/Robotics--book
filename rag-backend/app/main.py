from contextlib import asynccontextmanager
from fastapi import FastAPI
from .database.connection import engine
from .api.endpoints import health

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: database connection setup
    print("Connecting to database...")
    # Example: you might run migrations here, or check DB connection
    # For asyncpg, you might need to create an async pool here
    yield
    # Shutdown logic: database connection teardown
    print("Closing database connection...")
    engine.dispose()

app = FastAPI(lifespan=lifespan)

app.include_router(health.router)
