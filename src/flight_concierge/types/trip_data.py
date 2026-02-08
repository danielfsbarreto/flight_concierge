from datetime import date

from pydantic import BaseModel, field_validator

from flight_concierge.types import Airport, City, Country


class TripData(BaseModel):
    departure_country: Country
    departure_city: City
    departure_airport: Airport
    departure_date: str

    arrival_country: Country
    arrival_city: City
    arrival_airport: Airport
    arrival_date: str

    @field_validator("departure_date", "arrival_date", mode="before")
    @classmethod
    def validate_date(cls, v):
        if isinstance(v, date):
            return v.isoformat()
        if isinstance(v, str):
            # Validate the string is a valid date format
            try:
                date.fromisoformat(v)
                return v
            except ValueError:
                raise ValueError(f"Invalid date format: {v}. Expected YYYY-MM-DD")
        raise ValueError(f"Date must be a string or date object, got {type(v)}")

    def all_filled(self) -> bool:
        return all(
            [
                # Departure data
                self.departure_country.is_valid(),
                self.departure_city.is_valid(),
                self.departure_airport.is_valid(),
                self.departure_date is not None,
                # Arrival data
                self.arrival_country.is_valid(),
                self.arrival_city.is_valid(),
                self.arrival_airport.is_valid(),
                self.arrival_date is not None,
            ]
        )
