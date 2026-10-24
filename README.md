# VOICEMATRIX

AI-Powered Voice Clone Detection and Scam Prevention

## About

VOICEMATRIX is a voice-security prototype designed to help identify potentially AI-generated or manipulated speech.

## Features

- Audio voice analysis using available detection models
- Spectra-0 and AASIST detector options
- Live microphone monitoring
- Partial-fake analysis using overlapping audio windows
- Audio robustness testing
- Scam-prevention guidance
- React-based web interface

## Technology Stack

- Python
- FastAPI
- React
- Vite
- JavaScript
- ONNX Runtime and other project dependencies

## Project Structure

- backend: Python API and application logic
- frontend: React user interface and public assets
- scripts: Utility scripts, if included
- models: AI models, which must be obtained separately if excluded from the repository

## Requirements

Install Python 3 and Node.js with npm.

## Setup

### Backend

From the project root, create a virtual environment:

    py -3 -m venv .venv

Install backend dependencies:

    .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

Start the backend:

    cd backend

    ..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload

The API normally runs at http://127.0.0.1:8000.

### Frontend

Open another terminal in the project root and run:

    cd frontend

    npm install

    npm run dev

Open the local URL printed by Vite, normally http://localhost:5173.

## AI Model Installation

Large model weights are excluded from the source-code upload. Obtain the required models using the project's supported download instructions or model-download script.

Voice-analysis features requiring missing models will not work until those models are installed correctly.

## Limitations

A model prediction is not definitive proof that a recording is genuine or fake. Results may vary with language, recording quality, background noise, audio duration, and the voice-generation method.

This project is a prototype. Independent testing is necessary before using it for important security decisions.

## Privacy

Do not upload passwords, API keys, private recordings, or actual environment files containing secrets.

## License

Add an appropriate license before distributing this project publicly.
