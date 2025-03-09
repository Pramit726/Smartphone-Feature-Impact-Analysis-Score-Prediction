from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from sklearn import set_config

from ml.logger.logger import logging

from .routers import home, ratings

# Enable pandas output for the pipeline
set_config(transform_output="pandas")

app = FastAPI()

app.include_router(home.router)
app.include_router(ratings.router)
