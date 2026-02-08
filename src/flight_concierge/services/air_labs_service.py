from get_flight_airport_codes import GetFlightAirportCodes
from get_flight_city_codes import GetFlightCityCodes
from get_flight_country_codes import GetFlightCountryCodes

from flight_concierge.types import AirportsDatabase, CitiesDatabase, CountriesDatabase


class AirLabsService:
    def get_countries(self):
        data = GetFlightCountryCodes().run()
        return CountriesDatabase(data=data["response"])

    def get_cities(self):
        data = GetFlightCityCodes().run()
        return CitiesDatabase(data=data["response"])

    def get_airports(self):
        data = GetFlightAirportCodes().run()
        return AirportsDatabase(data=data["response"])
