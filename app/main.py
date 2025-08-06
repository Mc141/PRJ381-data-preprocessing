from typing import Union
from fastapi import FastAPI
from app.routers import observations
from app.routers import weather




app = FastAPI()

# router for retrieving observations from Inat API
app.include_router(observations.router)


app.include_router(weather.router)





