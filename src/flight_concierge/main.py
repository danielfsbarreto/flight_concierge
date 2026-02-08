#!/usr/bin/env python
from crewai.flow import Flow, and_, listen, persist, router, start

from flight_concierge.agents.flight_concierge_agent import FlightConciergeAgent
from flight_concierge.services import AirLabsService
from flight_concierge.types import FlightConciergeState


@persist()
class FlightConciergeFlow(Flow[FlightConciergeState]):
    @start()
    def load_initial_context(self):
        self.air_labs_service = AirLabsService()
        self.concierge_agent = FlightConciergeAgent(state=self.state)

        self.state.messages.append(self.state.message)

    @listen(load_initial_context)
    def collect_country_codes(self):
        if not self.state.flight_databases.countries.data:
            self.state.flight_databases.countries = (
                self.air_labs_service.get_countries()
            )

    @listen(load_initial_context)
    def collect_city_codes(self):
        if not self.state.flight_databases.cities.data:
            self.state.flight_databases.cities = self.air_labs_service.get_cities()

    @listen(load_initial_context)
    def collect_airport_codes(self):
        if not self.state.flight_databases.airports.data:
            self.state.flight_databases.airports = self.air_labs_service.get_airports()

    @listen(and_(collect_country_codes, collect_city_codes, collect_airport_codes))
    def handle_user_message(self):
        result = self.concierge_agent.handle_user_message()
        self.state.interactions.append(result)
        self.state.messages.append(result.assistant_response)

    @router(handle_user_message)
    def identify_routes(self):
        if self.state.interactions[-1].trip_data.all_filled():
            return "proceed_to_booking"

    @listen("proceed_to_booking")
    def booking_routes(self):
        pass


def kickoff():
    FlightConciergeFlow().kickoff(
        inputs={
            "id": "ba1c31de-bc8c-4583-be56-be303efd97fg",
            "message": {
                "role": "user",
                "content": "Quero a volta pro dia 13",
            },
        }
    )


def plot():
    FlightConciergeFlow().plot()


if __name__ == "__main__":
    kickoff()
