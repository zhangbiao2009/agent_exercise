"""
Weather tool for the agent.
Uses the free Open-Meteo API (no API key required).
"""

import requests
from smolagents import tool


# ---------- helper: geocode a city name ----------
def _geocode(city: str) -> dict:
    """Return {latitude, longitude, name, country} for a city."""
    resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en"},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Could not find city: {city}")
    r = results[0]
    return {
        "latitude": r["latitude"],
        "longitude": r["longitude"],
        "name": r["name"],
        "country": r["country"],
    }


# ---------- the tool that the agent can call ----------
@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a given city.

    Args:
        city: The name of the city, e.g. "Beijing", "New York", "London".

    Returns:
        A human-readable string describing the current weather conditions.
    """
    geo = _geocode(city)

    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                       "weather_code,wind_speed_10m",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
        },
        timeout=10,
    )
    resp.raise_for_status()
    current = resp.json()["current"]

    # Map WMO weather codes to plain English
    wmo_codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
    }
    weather_desc = wmo_codes.get(current["weather_code"], "Unknown")

    return (
        f"Weather in {geo['name']}, {geo['country']}:\n"
        f"  Condition : {weather_desc}\n"
        f"  Temperature : {current['temperature_2m']}°C\n"
        f"  Feels like : {current['apparent_temperature']}°C\n"
        f"  Humidity : {current['relative_humidity_2m']}%\n"
        f"  Wind speed : {current['wind_speed_10m']} km/h"
    )
