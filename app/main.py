from fastapi import FastAPI
from app.services.database import connect_to_mongo, close_mongo_connection
from app.routers import observations, weather, datasets
from contextlib import asynccontextmanager  




@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to mongo
    connect_to_mongo()
    yield
    # Close connection
    close_mongo_connection()



app = FastAPI(lifespan=lifespan)



# Routers
app.include_router(observations.router)
app.include_router(weather.router)
app.include_router(datasets.router)
