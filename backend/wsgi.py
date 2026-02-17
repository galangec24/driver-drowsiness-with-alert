"""
WSGI entry point for Render deployment
"""
import os
import sys

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your Flask app
from app import app

# Run startup tasks if needed (they'll run inside app context)
with app.app_context():
    # You can run startup tasks here if needed
    pass

# Export app for Gunicorn - this is what Gunicorn looks for
application = app

if __name__ == "__main__":
    app.run()