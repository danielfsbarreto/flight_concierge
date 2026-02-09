#!/usr/bin/env python
from typing import Literal

from crewai.flow import Flow, and_, human_feedback, listen, persist, start
from crewai.flow.human_feedback import HumanFeedbackResult

from flight_concierge.agents.flight_concierge_agent import FlightConciergeAgent
from flight_concierge.services import AirLabsService
from flight_concierge.types import FlightConciergeState, Message, Review


@persist()
class FlightConciergeFlow(Flow[FlightConciergeState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.air_labs_service = AirLabsService()
        self.concierge_agent = FlightConciergeAgent()

    @start()
    def load_initial_context(self):
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
        result = self.concierge_agent.acknowledge_message(self.state.messages)
        self.state.messages.append(result.assistant_response)

    @listen(acknowledge_user_message)
    def process_departure_details(self):
        result = self.concierge_agent.process_departure_information(self.state.messages)
        self.state.trip_data.legs[0].departure = result

    @listen(acknowledge_user_message)
    def process_arrival_details(self):
        result = self.concierge_agent.process_arrival_information(self.state.messages)
        self.state.trip_data.legs[0].arrival = result

    @listen(and_(process_departure_details, process_arrival_details))
    @human_feedback(
        message="Please review this trip planning details. Does it meet your needs?",
        emit=["needs_changes", "approved"],
        llm="gpt-4.1",
    )
    def draft_trip_plan(
        self, human_feedback_result
    ) -> Literal["needs_changes", "approved"]:
        result = self.concierge_agent.confirm_trip_data_with_user(
            messages=self.state.messages,
            trip_data=self.state.trip_data,
        )
        self.state.trip_data = result.metadata
        self.state.messages.append(result.assistant_response)
        self.state.interactions.append(result)
        return result.assistant_response.content

    @listen("needs_changes")
    def acknowledge_trip_plan_feedback(self, feedback_result: HumanFeedbackResult):
        user_message = Message(role="user", content=feedback_result.feedback)
        self.state.messages.append(user_message)
        self.state.trip_data.reviews.append(
            Review(
                agent_output=str(feedback_result.output),
                human_feedback=feedback_result.feedback,
                outcome=feedback_result.outcome,
            )
        )
        result = self.concierge_agent.acknowledge_trip_plan_feedback(
            messages=self.state.messages,
        )
        self.state.messages.append(result.assistant_response)

    @listen(acknowledge_trip_plan_feedback)
    @human_feedback(
        message="Please review the latest trip planning details. Is it better now?",
        emit=["needs_changes", "approved"],
        llm="gpt-4.1",
    )
    def act_on_trip_plan_feedback(self) -> Literal["needs_changes", "approved"]:
        result = self.concierge_agent.act_on_trip_plan_feedback(
            messages=self.state.messages,
            trip_data=self.state.trip_data,
        )
        self.state.trip_data = result.metadata
        self.state.messages.append(result.assistant_response)
        self.state.interactions.append(result)
        return result.assistant_response.content

    @listen("approved")
    def booking_route(self, feedback_result: HumanFeedbackResult):
        self.state.messages.append(
            Message(role="user", content=feedback_result.feedback)
        )
        result = self.concierge_agent.acknowledge_final_trip_planning_details(
            self.state.messages
        )
        self.state.messages.append(result.assistant_response)
        self.state.interactions.append(result)

    @listen(booking_route)
    def look_for_best_flights(self):
        result = self.concierge_agent.look_for_best_flights(
            trip_data=self.state.trip_data,
        )
        self.state.messages.append(result.assistant_response)
        self.state.interactions.append(result)


def kickoff():
    FlightConciergeFlow().kickoff(
        inputs={
            "message": {
                "role": "user",
                "content": "Gostaria de viajar de Recife para Campinas em 10 de fevereiro e retornar dia 13",
            },
        }
    )


def plot():
    FlightConciergeFlow().plot()


if __name__ == "__main__":
    kickoff()
