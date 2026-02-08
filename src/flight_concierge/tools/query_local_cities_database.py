from typing import Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class QueryLocalCitiesDatabaseInput(BaseModel):
    """Input schema for QueryLocalCitiesDatabase."""

    search_query: str = Field(..., description="City name or city code to search for")
    filter_by: Literal["city_code", "name", "auto"] = Field(
        default="auto",
        description="Filter type: 'city_code' for city code match, 'name' for city name match, 'auto' to search both with priority on city code",
    )


class QueryLocalCitiesDatabase(BaseTool):
    name: str = "Query Local Cities Database"
    description: str = (
        "Query the local cities database for city information. "
        "Can filter by city code or city name. "
        "Prioritizes exact city code matches over name matches when using 'auto' mode. "
        "Returns a list of city dictionaries with all fields."
    )
    database: BaseModel
    args_schema: Type[BaseModel] = QueryLocalCitiesDatabaseInput

    def _run(
        self,
        search_query: str,
        filter_by: Literal["city_code", "name", "auto"] = "auto",
    ) -> list[dict]:
        query_lower = search_query.lower()
        cities = self.database.data

        # Priority 1: Exact city code match
        if filter_by in ["city_code", "auto"]:
            city_code_matches = [
                city
                for city in cities
                if city.city_code and query_lower == city.city_code.lower()
            ]
            if city_code_matches:
                return [city.model_dump() for city in city_code_matches]

        # Priority 2: Name matches (partial allowed)
        if filter_by in ["name", "auto"]:
            name_matches = [
                city
                for city in cities
                if city.name and query_lower in city.name.lower()
            ]
            if name_matches:
                return [city.model_dump() for city in name_matches]

        return []
