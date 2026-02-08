#!/usr/bin/env python
from typing import Literal

from crewai.flow import Flow, and_, human_feedback, listen, or_, persist, start

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
        self.air_labs_service.ensure_countries_cached()

    @listen(load_initial_context)
    def collect_city_codes(self):
        self.air_labs_service.ensure_cities_cached()

    @listen(load_initial_context)
    def collect_airport_codes(self):
        self.air_labs_service.ensure_airports_cached()

    @listen(and_(collect_country_codes, collect_city_codes, collect_airport_codes))
    def acknowledge_user_message(self):
        result = self.concierge_agent.acknowledge_message()
        self.state.messages.append(result.assistant_response)

    @listen(acknowledge_user_message)
    def process_departure_details(self):
        result = self.concierge_agent.process_departure_information()
        self.state.trip_data.departure = result

    @listen(acknowledge_user_message)
    def process_arrival_details(self):
        result = self.concierge_agent.process_arrival_information()
        self.state.trip_data.arrival = result

    @listen(
        or_(
            and_(process_departure_details, process_arrival_details),
            "redo_plan",
        )
    )
    @human_feedback(
        message="Please review this trip planning details. Does it meet your needs?",
        emit=["redo_plan", "proceed_with_booking"],
        llm="gpt-4.1",
    )
    def draft_trip_plan(
        self, human_feedback_result
    ) -> Literal["redo_plan", "proceed_with_booking"]:
        result = self.concierge_agent.confirm_trip_data_with_user(
            human_feedback=human_feedback_result
        )
        self.state.interactions.append(result)
        self.state.messages.append(result.assistant_response)
        return result.assistant_response.content

    @listen("proceed_with_booking")
    def booking_route(self):
        pass


def kickoff():
    FlightConciergeFlow().kickoff(
        inputs={
            "id": "ba1c31de-bc8c-4583-be56-be303efd97fh",
            "message": {
                "role": "user",
                "content": "I want to travel from Recife to São Paulo (Guarulhos) on February 10 and return on February 13",
            },
        }
    )


def plot():
    FlightConciergeFlow().plot()


if __name__ == "__main__":
    kickoff()
