"""Main entry point for the Sentiment Analysis Dashboard application."""

import os
import sys
from app import create_app

app = create_app()

if __name__ == '__main__':
    try:
        from models import socketio
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    except Exception:
        app.run(host='0.0.0.0', port=5000, debug=True)
