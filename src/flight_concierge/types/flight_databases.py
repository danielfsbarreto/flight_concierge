from pydantic import BaseModel

from .airports_database import AirportsDatabase
from .cities_database import CitiesDatabase
from .countries_database import CountriesDatabase


class FlightDatabases(BaseModel):
    countries: CountriesDatabase = CountriesDatabase()
    cities: CitiesDatabase = CitiesDatabase()
    airports: AirportsDatabase = AirportsDatabase()
