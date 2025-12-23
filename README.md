# BrezyWeather

A simple weather app built with Python and Streamlit that uses OpenAI API to display current weather information for any city.

## Features

- Clean, user-friendly interface
- Get weather information for any city worldwide
- Displays temperature (Celsius & Fahrenheit), conditions, humidity, and wind speed
- Secure API key input via sidebar

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd BrezyWeather
   ```

2. **Activate the virtual environment:**

   Windows (Command Prompt):
   ```cmd
   venv\Scripts\activate
   ```

   Windows (PowerShell):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the app:**
   ```bash
   streamlit run app.py
   ```

2. **Enter your OpenAI API key** in the sidebar

3. **Type a city name** and click "Get Weather"

## Project Structure

```
BrezyWeather/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── venv/               # Virtual environment
└── README.md           # This file
```

## Note

Weather data is AI-generated for demonstration purposes. For real-time weather data, consider integrating a dedicated weather API like OpenWeatherMap.

## License

MIT License
