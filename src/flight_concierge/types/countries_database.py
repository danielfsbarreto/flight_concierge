from typing import List

from pydantic import BaseModel

from .country import Country


class CountriesDatabase(BaseModel):
    data: List[Country] = []
