# LoanWatch Backend

This is the backend API for the LoanWatch application, built with FastAPI.

## Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the development server:
   ```bash
   uvicorn app:app --reload
   ```

## API Endpoints

- `POST /predict`: Submit a loan application for prediction
- `GET /fairness`: Get fairness metrics
- `GET /visualizations/{viz_type}`: Get visualization images
- `POST /run-analysis`: Run the full analysis pipeline

## Structure

- `app.py`: FastAPI application with API endpoints
- `requirements.txt`: Python dependencies
