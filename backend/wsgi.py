"""
WSGI entry point for Render deployment
"""
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your Flask app
from app import app, startup_tasks

# Run startup tasks
with app.app_context():
    startup_tasks()

# Export app for Gunicorn
application = app