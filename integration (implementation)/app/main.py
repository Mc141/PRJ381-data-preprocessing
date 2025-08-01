from typing import Union
from fastapi import FastAPI
from app.routers import observations




app = FastAPI()

# router for retrieving observations from Inat API
app.include_router(observations.router)








