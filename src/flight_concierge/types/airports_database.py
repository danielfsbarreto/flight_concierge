from typing import List

from pydantic import BaseModel

from .airport import Airport


class AirportsDatabase(BaseModel):
    data: List[Airport] = []
