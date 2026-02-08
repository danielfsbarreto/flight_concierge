from typing import List

from pydantic import BaseModel

from .city import City


class CitiesDatabase(BaseModel):
    data: List[City] = []
