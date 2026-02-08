from typing import Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class QueryLocalAirportsDatabaseInput(BaseModel):
    """Input schema for QueryLocalAirportsDatabase."""

    search_query: str = Field(
        ...,
        description="Airport name, city code, IATA code, or ICAO code to search for",
    )
    filter_by: Literal["iata_code", "icao_code", "name", "city_code", "auto"] = Field(
        default="auto",
        description="Filter type: 'iata_code' for IATA code match, 'icao_code' for ICAO code match, 'name' for airport name match, 'city_code' for city code match, 'auto' to search all fields with priority on codes",
    )


class QueryLocalAirportsDatabase(BaseTool):
    name: str = "Query Local Airports Database"
    description: str = (
        "Query the local airports database for airport information. "
        "Can filter by IATA code, ICAO code, airport name, or city code. "
        "Prioritizes exact code matches over name matches when using 'auto' mode. "
        "Returns a list of airport dictionaries with all fields."
    )
    database: BaseModel
    args_schema: Type[BaseModel] = QueryLocalAirportsDatabaseInput

    def _run(
        self,
        search_query: str,
        filter_by: Literal[
            "iata_code", "icao_code", "name", "city_code", "auto"
        ] = "auto",
    ) -> list[dict]:
        query_lower = search_query.lower()
        airports = self.database.data

        # Priority 1: Exact IATA code match (always check first)
        if filter_by in ["iata_code", "auto"]:
            iata_matches = [
                airport
                for airport in airports
                if airport.iata_code and query_lower == airport.iata_code.lower()
            ]
            if iata_matches:
                return [airport.model_dump() for airport in iata_matches]

        # Priority 2: Exact ICAO code match
        if filter_by in ["icao_code", "auto"]:
            icao_matches = [
                airport
                for airport in airports
                if airport.icao_code and query_lower == airport.icao_code.lower()
            ]
            if icao_matches:
                return [airport.model_dump() for airport in icao_matches]

        # Priority 3: Exact city code match
        if filter_by in ["city_code", "auto"]:
            city_code_matches = [
                airport
                for airport in airports
                if airport.city_code and query_lower == airport.city_code.lower()
            ]
            if city_code_matches:
                return [airport.model_dump() for airport in city_code_matches]

        # Priority 4: Name matches (partial allowed)
        if filter_by in ["name", "auto"]:
            name_matches = [
                airport
                for airport in airports
                if airport.name and query_lower in airport.name.lower()
            ]
            if name_matches:
                return [airport.model_dump() for airport in name_matches]

        return []
