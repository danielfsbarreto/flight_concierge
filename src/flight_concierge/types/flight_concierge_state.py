from pydantic import BaseModel

from .interaction import Interaction
from .message import Message
from .trip_data import TripData


class FlightConciergeState(BaseModel):
    # inputs
    message: Message | None = None

    # processing
    messages: list[Message] = []
    interactions: list[Interaction] = []
    trip_data: TripData = TripData()
