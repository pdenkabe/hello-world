# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the app:**
```bash
streamlit run app.py
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Activate virtual environment:**
- Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
- Windows (CMD): `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

## Architecture

This is a single-file Streamlit application (`app.py`) that uses OpenAI's API to generate simulated weather data. The app takes a city name as input and returns AI-generated weather information (temperature, conditions, humidity, wind speed).

Key implementation notes:
- Weather data is AI-generated, not from a real weather API
- OpenAI API key is entered via the sidebar at runtime (not stored)
- Uses the `gpt-4o-mini` model for cost-effective responses
