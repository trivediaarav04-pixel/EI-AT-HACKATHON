# Urban Air Quality Intelligence Platform — Prototype

ET AI Hackathon 2026 · Problem Statement 5

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## What's inside
- **Module 1 — Hyperlocal AQI Forecasting**: 24–72hr ward-level AQI forecast with confidence band (simulated LSTM-style output; swap in a real CPCB-trained model + OpenWeatherMap API for production).
- **Module 2 — Geospatial Source Attribution**: interactive Folium map + per-ward source-mix breakdown (traffic, industrial, construction, biomass, road dust, waste burning) with confidence scores.
- **Enforcement Priority Dashboard**: ranks wards for proactive inspection deployment and shows illustrative impact metrics.

## Note on data
Runs fully offline on synthetic data calibrated to CPCB-reported AQI ranges, so it demos without any API keys. Each data source (CPCB CAAQMS, OpenWeatherMap, NASA FIRMS/Sentinel-2, OSM Overpass) is called out in the sidebar and in the Module 1 "Model details" expander — these are the integration points for the production build.

## Next steps for production
1. Replace `generate_ward_data` / `generate_forecast` with live CPCB CAAQMS + OpenWeatherMap API calls.
2. Train the LSTM on historical station data per city.
3. Replace the rule-based source-mix generator with the multi-agent attribution engine fusing OSM land-use, traffic telemetry, and NASA FIRMS thermal anomalies.
