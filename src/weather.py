"""Small Open-Meteo client for weather-aware outfit recommendation."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherServiceError(RuntimeError):
    """Raised when a city or its forecast cannot be retrieved."""


def _get_json(url, parameters):
    request_url = f"{url}?{urlencode(parameters)}"

    try:
        with urlopen(request_url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise WeatherServiceError("Weather service is temporarily unavailable.") from error


def geocode_city(city):
    """Resolve a city name to one unambiguous Open-Meteo location."""

    city = str(city).strip()
    if not city:
        raise WeatherServiceError("Please enter a city.")

    payload = _get_json(
        GEOCODING_URL,
        {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json",
        },
    )

    results = payload.get("results", [])
    if not results:
        raise WeatherServiceError(f"No weather location was found for '{city}'.")

    location = results[0]

    return {
        "name": location["name"],
        "country": location.get("country", ""),
        "latitude": float(location["latitude"]),
        "longitude": float(location["longitude"]),
        "timezone": location.get("timezone", "auto"),
    }


def describe_weather_code(code):
    """Convert a WMO weather code into a compact recommendation label."""

    code = int(code)

    if code == 0:
        return "Clear"
    if code in {1, 2, 3}:
        return "Cloudy"
    if code in {45, 48}:
        return "Fog"
    if code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}:
        return "Rain"
    if code in {71, 73, 75, 77, 85, 86}:
        return "Snow"
    if code in {95, 96, 99}:
        return "Thunderstorm"
    return "Unknown"


def get_city_weather(city):
    """Return current conditions and today's rain probability for a city."""

    location = geocode_city(city)

    payload = _get_json(
        FORECAST_URL,
        {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": (
                "temperature_2m,apparent_temperature,precipitation,"
                "rain,showers,snowfall,weather_code,wind_speed_10m"
            ),
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max"
            ),
            "forecast_days": 1,
            "timezone": "auto",
        },
    )

    current = payload.get("current", {})
    daily = payload.get("daily", {})

    def first_daily_value(name, default=0.0):
        values = daily.get(name, [])
        return float(values[0]) if values else float(default)

    weather_code = int(current.get("weather_code", -1))

    return {
        "city": location["name"],
        "country": location["country"],
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "temperature": float(current.get("temperature_2m", 20.0)),
        "feels_like": float(current.get("apparent_temperature", 20.0)),
        "precipitation": float(current.get("precipitation", 0.0)),
        "rain": float(current.get("rain", 0.0)),
        "showers": float(current.get("showers", 0.0)),
        "snowfall": float(current.get("snowfall", 0.0)),
        "weather_code": weather_code,
        "condition": describe_weather_code(weather_code),
        "wind_speed": float(current.get("wind_speed_10m", 0.0)),
        "temperature_max": first_daily_value("temperature_2m_max", 20.0),
        "temperature_min": first_daily_value("temperature_2m_min", 10.0),
        "rain_probability": first_daily_value(
            "precipitation_probability_max",
            0.0,
        ),
        "source": "Open-Meteo",
    }
