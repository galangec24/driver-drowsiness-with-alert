# Driver Drowsiness Detection System

A capstone project that detects driver drowsiness and sends alerts to registered guardians.

## Features
- Real-time drowsiness detection using AI
- Guardian registration and authentication
- Driver registration with face capture
- Mobile-responsive web interface
- Real-time alerts via WebSocket
- PWA support (install as mobile app)

## Local Development
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate

# 3. Install dependencies
cd backend
pip install -r requirements.txt

# 4. Run the application
python app.py