import streamlit as st
import openai
import os

# Configure page
st.set_page_config(
    page_title="BrezyWeather",
    page_icon="🌤️",
    layout="centered"
)

# Title and description
st.title("🌤️ BrezyWeather")
st.markdown("Get current weather information for any city using AI")

# Sidebar for API key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    if api_key:
        openai.api_key = api_key

# Main input
city = st.text_input("Enter a city name", placeholder="e.g., London, Tokyo, New York")

def get_weather(city_name: str) -> str:
    """Use OpenAI to get weather information for a city."""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a weather assistant. When given a city name, provide a realistic current weather report including: temperature (in both Celsius and Fahrenheit), weather condition (sunny, cloudy, rainy, etc.), humidity percentage, and wind speed. Format the response in a clear, readable way. Note: This is simulated weather data for demonstration purposes."
                },
                {
                    "role": "user",
                    "content": f"What is the current weather in {city_name}?"
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except openai.AuthenticationError:
        return "❌ Invalid API key. Please check your OpenAI API key."
    except openai.RateLimitError:
        return "❌ Rate limit exceeded. Please try again later."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Search button
if st.button("Get Weather", type="primary"):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not city:
        st.warning("Please enter a city name.")
    else:
        with st.spinner(f"Fetching weather for {city}..."):
            weather_info = get_weather(city)
            st.markdown("---")
            st.subheader(f"Weather in {city}")
            st.markdown(weather_info)

# Footer
st.markdown("---")
st.caption("BrezyWeather - Powered by OpenAI | Weather data is AI-generated for demonstration")
