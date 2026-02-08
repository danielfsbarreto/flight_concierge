from datetime import datetime

from crewai import Agent
from get_flight_airports_nearby import GetFlightAirportsNearby

from flight_concierge.tools import (
    QueryLocalAirportsDatabase,
    QueryLocalCitiesDatabase,
    QueryLocalCountriesDatabase,
)
from flight_concierge.types import FlightConciergeState, Interaction


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
                QueryLocalCountriesDatabase(
                    database=self._state.flight_databases.countries,
                ),
                QueryLocalCitiesDatabase(
                    database=self._state.flight_databases.cities,
                ),
                QueryLocalAirportsDatabase(
                    database=self._state.flight_databases.airports
                ),
                GetFlightAirportsNearby(),
            ],
            llm="gpt-4.1",
        )

    def handle_user_message(self):
        messages_context = "\n".join(
            [f"{msg.role.upper()}: {msg.content}" for msg in self._state.messages[-10:]]
        )

        prompt = f"""
        As a Senior Travel Concierge, your job is to help users plan their trips by collecting
        travel information in a structured, top-down manner.

        Analyze the following conversation and gather trip information using a TOP-DOWN approach:

        CONVERSATION HISTORY:

        {messages_context}

        INFORMATION HIERARCHY (Top-Down):

        LEVEL 1 - Countries:
        - Departure country
        - Arrival country
        - If user mentions a city, use Query Local Cities Database to determine the country automatically
        - If user mentions an airport, use Query Local Airports Database to determine the city and country automatically

        LEVEL 2 - Cities:
        - Departure city
        - Arrival city
        - Verify cities belong to the countries identified in Level 1

        LEVEL 3 - Airports:
        - Departure airport options
        - Arrival airport options
        - Use Query Local Airports Database with city codes to find all airports
        - Only use Get Flight Airports Nearby if local database has no results

        LEVEL 4 - Dates:
        - Departure date and time
        - Return date and time (if applicable)

        SMART INFERENCE:
        - If user says "I'm flying from Paris", automatically infer:
          * Country: France (use Query Local Cities Database to verify)
          * City: Paris
          * Then find airports in Paris
        - If user says "I need to go to JFK", automatically infer:
          * Airport: JFK
          * City: New York (use Query Local Airports Database to get city_code)
          * Country: USA (from city lookup)

        YOUR RESPONSIBILITIES:
        1. Ask for missing information starting from the highest level (country) if not inferrable
        2. When user provides lower-level info (city/airport), automatically fill in higher levels using tools
        3. Always steer the conversation towards completing the trip data
        4. Be friendly, professional, and efficient
        5. Confirm inferred information with the user (e.g., "I see you're traveling from Paris, France - is that correct?")

        REQUIRED INFORMATION TO COMPLETE:
        - Departure: country, city, airport options, date
        - Arrival: country, city, airport options, date

        Return:
        - assistant_response: Your friendly response to the user with any clarifying questions
        - trip_data: The latest trip data collected (including auto-inferred information)
        """
        return self._agent.kickoff(prompt.strip(), response_format=Interaction).pydantic

    def extract_information_from_user_request(self):
        prompt = f"""
        Analyze the following user request and extract flight information using a TOP-DOWN approach:

        User Request: "{self._state.user_request}"

        STRICT WORKFLOW - Follow these steps in order:

        STEP 1: Identify Countries
        - Extract the departure country name from the user request
        - Extract the arrival country name from the user request
        - Use Query Local Countries Database to verify each country exists and get country codes
        - Save: departure_country_code, arrival_country_code

        STEP 2: Identify Cities
        - Extract the departure city name from the user request
        - Extract the arrival city name from the user request
        - Use Query Local Cities Database ONCE for departure city (filter by name if needed)
        - Use Query Local Cities Database ONCE for arrival city (filter by name if needed)
        - From results, save: city_code, lat, lng, country_code for each city
        - Verify the country codes match the ones from STEP 1

        STEP 3: Extract Travel Date/Time
        - Extract the travel date and time from the user request
        - Save this information

        STEP 4: Find Airports
        - For BOTH departure and arrival cities, you need to find airports
        - For this:
            Query Local Airports Database (FREE - use this first)
            - Use filter_by="city_code" with the city_code from STEP 2
            - This will return all airports in that city from the local database

            Get Flight Airports Nearby (PAID API - use only after Query Local Airports Database)
            - Use the lat/lng coordinates from STEP 2
            - Set distance parameter up to 30 km
            - Returns airports sorted by popularity scores and distance
            - Ignore heliports
            - Ignore airports with with low popularity scores

        Recommended approach:
        - Always prioritize using the Query Local Airports Database tool
        - Do not abuse the Get Flight Airports Nearby API since it is paid. Use it only once per city
        (i.e. once for the departure city and once for the arrival city)

        STEP 5: Format and Return Results
        Provide a structured response with:
        - Departure country and country code
        - Arrival country and country code
        - Departure city: name, city_code, coordinates (lat, lng)
        - Arrival city: name, city_code, coordinates (lat, lng)
        - Travel date and time
        - Departure airports: list of airports with name, IATA code, ICAO code, city_code
        - Arrival airports: list of airports with name, IATA code, ICAO code, city_code

        CRITICAL RULES:
        - Follow the TOP-DOWN approach: Countries → Cities → Airports
        - Query each database ONCE per entity (one query per country, one per city)
        - Get Flight Airports Nearby: PAID API - use it carefully
        - Prefer Query Local Airports Database over Get Flight Airports Nearby
        - Do NOT skip steps or change the order
        - Ignore heliports altogether
        - Ignore airports with low popularity scores

        If any required information is missing, report it clearly and stop.
        """

        return self._agent.kickoff(prompt.strip())
