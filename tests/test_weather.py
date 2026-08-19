import unittest
from unittest.mock import patch

from src.weather import (
    WeatherServiceError,
    describe_weather_code,
    geocode_city,
    get_city_weather,
)


class WeatherTests(unittest.TestCase):
    def test_all_supported_wmo_groups(self):
        expected = {
            0: "Clear",
            3: "Cloudy",
            48: "Fog",
            82: "Rain",
            86: "Snow",
            99: "Thunderstorm",
            -1: "Unknown",
        }
        for code, condition in expected.items():
            with self.subTest(code=code):
                self.assertEqual(describe_weather_code(code), condition)

    def test_empty_city_is_a_user_facing_error(self):
        with self.assertRaisesRegex(WeatherServiceError, "Please enter a city"):
            geocode_city("   ")

    @patch("src.weather._get_json")
    def test_incomplete_location_is_wrapped(self, get_json):
        get_json.return_value = {"results": [{"name": "Nowhere"}]}
        with self.assertRaises(WeatherServiceError):
            geocode_city("Nowhere")

    @patch("src.weather._get_json")
    def test_null_forecast_values_use_safe_defaults(self, get_json):
        get_json.side_effect = [
            {
                "results": [
                    {
                        "name": "Test City",
                        "country": "Testland",
                        "latitude": 1,
                        "longitude": 2,
                    }
                ]
            },
            {
                "current": {
                    "temperature_2m": None,
                    "apparent_temperature": None,
                    "weather_code": None,
                },
                "daily": {
                    "temperature_2m_max": [None],
                    "precipitation_probability_max": None,
                },
            },
        ]

        result = get_city_weather("Test City")

        self.assertEqual(result["temperature"], 20.0)
        self.assertEqual(result["feels_like"], 20.0)
        self.assertEqual(result["condition"], "Unknown")
        self.assertEqual(result["rain_probability"], 0.0)


if __name__ == "__main__":
    unittest.main()
