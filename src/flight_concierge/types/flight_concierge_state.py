from pydantic import BaseModel

from .flight_databases import FlightDatabases
from .interaction import Interaction
from .message import Message


class FlightConciergeState(BaseModel):
    # inputs
    message: Message | None = None

    # processing
    flight_databases: FlightDatabases = FlightDatabases()
    messages: list[Message] = []
    interactions: list[Interaction] = []
