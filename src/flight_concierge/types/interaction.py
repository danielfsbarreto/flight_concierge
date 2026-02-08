from pydantic import BaseModel

from .message import Message
from .trip_data import TripData


class Interaction(BaseModel):
    assistant_response: Message
    trip_data: TripData
