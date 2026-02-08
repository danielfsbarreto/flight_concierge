from datetime import datetime

from crewai import Agent
from crewai.flow.human_feedback import HumanFeedbackResult
from get_flight_airports_nearby import GetFlightAirportsNearby

from flight_concierge.tools import (
    QueryLocalAirportsDatabase,
    QueryLocalCitiesDatabase,
    QueryLocalCountriesDatabase,
)
from flight_concierge.types import (
    ArrivalData,
    DepartureData,
    FlightConciergeState,
    Interaction,
)


class FlightConciergeAgent:
    def __init__(self, state: FlightConciergeState):
        self._state = state
        self._agent = Agent(
            role="CrewAI Senior Travel Concierge",
            goal=f"""Find the most convenient flight routes and airport options for traveling FDEs,
            considering proximity, accessibility, and travel efficiency in a fluid conversational
            interface that respects the idiom being utilized by the user. You are also always
            aware of the current date - which is {datetime.now().strftime("%Y-%m-%d")}""".strip(),
            backstory="""You are a dedicated and professional travel concierge specializing in
            supporting CrewAI's Field Development Engineers (FDEs) who travel extensively around
            the globe. With years of experience in corporate travel logistics, you excel at
            identifying the most convenient airports, optimal routes, and practical travel options.
            You understand that FDEs value efficiency, proximity to city centers, and seamless
            connections. Your friendly yet professional approach ensures that every traveler feels
            supported and confident in their journey. You prioritize finding airports that minimize
            travel time and maximize convenience, always keeping the traveler's experience at the
            forefront of your recommendations.""".strip(),
            inject_date=True,
            tools=[
                QueryLocalCountriesDatabase(),
                QueryLocalCitiesDatabase(),
                QueryLocalAirportsDatabase(),
                GetFlightAirportsNearby(),
            ],
            llm="gpt-4.1",
        )

    def acknowledge_message(self):
        messages_context = "\n".join(
            [f"{msg.role.upper()}: {msg.content}" for msg in self._state.messages[-10:]]
        )

        prompt = f"""
        As a Senior Travel Concierge, acknowledge the user's latest message in a warm, professional way.

        CONVERSATION HISTORY:
        {messages_context}

        YOUR TASK:
        - Acknowledge what the user just said
        - Let them know you're processing their travel request
        - Keep it brief, friendly, and reassuring
        - Set expectations that you'll find the best flight options

        Return ONLY:
        - assistant_response: Your brief acknowledgment message (1-2 sentences max)
        """

        return self._agent.kickoff(prompt.strip(), response_format=Interaction).pydantic

    def process_departure_information(self):
        messages_context = "\n".join(
            [f"{msg.role.upper()}: {msg.content}" for msg in self._state.messages[-10:]]
        )

        prompt = f"""
        As a Senior Travel Concierge, focus on extracting and processing DEPARTURE information only.

        CONVERSATION HISTORY:
        {messages_context}

        YOUR TASK: Extract departure details using TOP-DOWN approach:

        1. DEPARTURE COUNTRY
           - Identify departure country from conversation
           - Use Query Local Countries Database to verify and get country code

        2. DEPARTURE CITY
           - Identify departure city from conversation
           - Use Query Local Cities Database to get: city_code, lat, lng, country_code
           - Verify country code matches step 1

        3. DEPARTURE AIRPORTS
           - Use Query Local Airports Database with city_code (FREE, try first)
           - If no results, use Get Flight Airports Nearby with lat/lng (PAID, max once)
           - Distance: 30 km, ignore heliports and low popularity airports

        4. DEPARTURE DATE
           - Extract departure date from conversation (format: YYYY-MM-DD)

        Return only the departure information.
        """

        return self._agent.kickoff(
            prompt.strip(), response_format=DepartureData
        ).pydantic

    def process_arrival_information(self):
        """Process arrival location details: country, city, and airports."""
        messages_context = "\n".join(
            [f"{msg.role.upper()}: {msg.content}" for msg in self._state.messages[-10:]]
        )

        prompt = f"""
        As a Senior Travel Concierge, focus on extracting and processing ARRIVAL information only.

        CONVERSATION HISTORY:
        {messages_context}

        YOUR TASK: Extract arrival details using TOP-DOWN approach:

        1. ARRIVAL COUNTRY
           - Identify arrival country from conversation
           - Use Query Local Countries Database to verify and get country code

        2. ARRIVAL CITY
           - Identify arrival city from conversation
           - Use Query Local Cities Database to get: city_code, lat, lng, country_code
           - Verify country code matches step 1

        3. ARRIVAL AIRPORTS
           - Use Query Local Airports Database with city_code (FREE, try first)
           - If no results, use Get Flight Airports Nearby with lat/lng (PAID, max once)
           - Distance: 30 km, ignore heliports and low popularity airports

        4. ARRIVAL DATE
           - Extract arrival date from conversation (format: YYYY-MM-DD)

        Return only the departure information.
        """

        return self._agent.kickoff(prompt.strip(), response_format=ArrivalData).pydantic

    def confirm_trip_data_with_user(
        self, human_feedback: HumanFeedbackResult | None = None
    ):
        messages_context = "\n".join(
            [f"{msg.role.upper()}: {msg.content}" for msg in self._state.messages[-10:]]
        )

        prompt = f"""
        As a Senior Travel Concierge, compile the trip details, present them to the user,
        and review them if necessary.

        CONVERSATION HISTORY:
        {messages_context}

        DEPARTURE INFORMATION:
        {self._state.trip_data.departure.model_dump_json()}

        ARRIVAL INFORMATION:
        {self._state.trip_data.arrival.model_dump_json()}

        HUMAN FEEDBACK:
        {human_feedback}

        YOUR TASK:
        1. Review all departure and arrival information gathered
        2. Present a comprehensive summary with:
           - Complete departure details (country, city, airport options, date)
           - Complete arrival details (country, city, airport options, date)
        3. Highlight the most convenient airport options based on proximity and popularity
        4. Ask if the traveler needs any clarification or has preferences

        CRITICAL RULES:
        - Never suggest airports more than 30 km away from the cities the user is interested in
        - Ignore heliports and airports with low popularity scores

        RETURN:
        - assistant_response: A friendly, professional summary of the trip details.
        The intent with this message is to confirm the trip details before proceeding with the
        booking process.
        - metadata: The complete TripData information with all known information filled

        If any critical information is still missing, clearly ask for it.
        """

        return self._agent.kickoff(prompt.strip(), response_format=Interaction).pydantic
