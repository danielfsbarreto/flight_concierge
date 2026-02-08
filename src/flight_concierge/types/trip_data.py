from pydantic import BaseModel

from flight_concierge.types import ArrivalData, DepartureData


class TripData(BaseModel):
    departure: DepartureData | None = None
    arrival: ArrivalData | None = None
