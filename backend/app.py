import os
from pathlib import Path
import sys
import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4'] 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / '.env')

from flask import Flask, request, jsonify, Response
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from datetime import datetime, timedelta
import uuid
import eventlet
import base64
import json
import secrets
import threading
import time
import re
import bcrypt
from contextlib import contextmanager
import urllib.parse
import traceback

# Cloudinary imports
import cloudinary
import cloudinary.uploader
import cloudinary.api
import cloudinary.exceptions  
from cloudinary.utils import cloudinary_url

#Facial Recognition
import io
import cv2
import face_recognition
import numpy as np
from PIL import Image


# Google OAuth imports
import jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# ==================== POSTGRESQL CONFIGURATION ====================
import psycopg2
from psycopg2.extras import RealDictCursor

# ==================== APP SETUP ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Patch for async
eventlet.monkey_patch()

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')
# ==================== CLOUDINARY CONFIGURATION ====================
CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME', '')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY', '')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET', '')

# Initialize Cloudinary with Render DNS fix
if all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]):
    print(f"☁️  Cloudinary configured for cloud: {CLOUDINARY_CLOUD_NAME}")
    
    # Configure Cloudinary with timeout settings for Render
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True
    )
    
    # Set up custom HTTP adapter with retries for Render's DNS issues
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    # Create a custom session with retries
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PUT", "DELETE"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

#region Cloudinary Configuration    
    # Configure Cloudinary to use our session
    import cloudinary
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True,
        api_proxy="https://api.cloudinary.com"
    )
    
    # Import after configuration
    import cloudinary.uploader
    import cloudinary.api
    
    CLOUDINARY_ENABLED = True
    print(f"✅ Cloudinary initialized successfully")
else:
    CLOUDINARY_ENABLED = False
    print("⚠️  Cloudinary not configured. Check environment variables.")
#End Region

# ==================== FIREBASE HOSTING INTEGRATION ====================
# List of allowed origins (Firebase domains + localhost for development)
ALLOWED_ORIGINS = [
    'https://guardian-drive-app.web.app',
    'https://guardian-drive-app.firebaseapp.com',
    'https://guardian-drive.firebaseapp.com',
    'http://localhost:5000',
    'http://localhost:8080',
    'http://localhost:3000',
    'http://localhost:5500',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:8080',
    'http://127.0.0.1:3000',
    'http://127.0.0.1:5500',
]

# Configure for production with proxy support
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

# Configure CORS for Firebase Hosting - STRICT CONFIGURATION
CORS(app, 
     supports_credentials=True,
     origins=ALLOWED_ORIGINS,
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-Admin-Username", "X-Admin-Token", "Origin", "Accept", "Cache-Control"],
     expose_headers=["Content-Type", "Authorization", "X-Admin-Token"],
     max_age=3600)

# Initialize SocketIO for real-time alerts with Firebase CORS
socketio = SocketIO(
    app,
    cors_allowed_origins=ALLOWED_ORIGINS,
    async_mode='eventlet',
    
    # CRITICAL: WebSocket specific settings
    ping_timeout=60,           # How long to wait for pong response
    ping_interval=25,          # How often to send pings
    max_http_buffer_size=1e6,  # 1MB buffer
    
    # Transport settings - order matters!
    transports=['polling', 'websocket'],  # Start with polling, upgrade to websocket
    allow_upgrades=True,
    
    # Connection management
    manage_session=False,       # Don't manage sessions automatically
    cookie=None,                # Disable cookies for security
    always_connect=True,        # Always connect even if auth fails initially
    
    # Logging - enable for debugging (disable in production)
    logger=True,
    engineio_logger=True,
    log_output=True,
    
    # Path configuration (important for Render)
    path='socket.io',            # Explicitly set the path
    
    # CORS settings
    cors_credentials=True,
    
    # Engine.IO settings
    max_guest_sessions=1000,     # Maximum number of guest sessions
    preserve_context=True,        # Preserve context between requests
    websocket_ping_interval=25,   
    websocket_ping_timeout=60     
)

# ==================== SECURITY CONFIGURATION ====================
# Generate bcrypt hash for admin password (admin123)
ADMIN_PASSWORD_HASH = bcrypt.hashpw(b'admin123', bcrypt.gensalt(rounds=12)).decode('utf-8')

ADMIN_CREDENTIALS = {
    'admin': {
        'password_hash': ADMIN_PASSWORD_HASH,
        'full_name': 'System Administrator',
        'role': 'super_admin',
        'email': 'admin@driveralert.com',
        'created_at': datetime.now().isoformat()
    }
}

admin_rate_limit = {}
admin_sessions = {}

# ==================== PASSWORD FUNCTIONS ====================
def clean_password(password):
    """Remove all spaces from password"""
    if not password:
        return ''
    return password.replace(' ', '')

def hash_password(password):
    """Hash password using bcrypt (password should already be cleaned)"""
    try:
        # Password should already be cleaned before calling this function
        if not password:
            raise ValueError("Password cannot be empty")
        
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    except Exception as e:
        print(f"❌ Error hashing password: {e}")
        raise

def verify_password(password, hashed_password):
    """Verify password against bcrypt hash after removing spaces"""
    try:
        # Clean the input password (remove all spaces)
        password = clean_password(password)
        if not password:
            return False
        
        if not hashed_password or len(hashed_password) < 60 or not hashed_password.startswith('$2'):
            return False
        
        password_bytes = password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        print(f"❌ Error verifying password: {e}")
        return False

def verify_admin_credentials(username, password):
    """Verify admin login credentials"""
    if username not in ADMIN_CREDENTIALS:
        return None
    
    stored_hash = ADMIN_CREDENTIALS[username]['password_hash']
    
    if verify_password(password, stored_hash):
        return {
            'username': username,
            'full_name': ADMIN_CREDENTIALS[username]['full_name'],
            'role': ADMIN_CREDENTIALS[username]['role'],
            'email': ADMIN_CREDENTIALS[username].get('email', '')
        }
    
    return None

def create_admin_session(username):
    """Create admin session token"""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=8)
    
    admin_sessions[username] = {
        'token': token,
        'expires': expires_at,
        'created': datetime.now(),
        'last_activity': datetime.now()
    }
    
    return token, expires_at

def validate_admin_token(username, token):
    """Validate admin session token"""
    if username not in admin_sessions:
        return False
    
    session = admin_sessions[username]
    
    if (session['token'] == token and 
        session['expires'] > datetime.now()):
        admin_sessions[username]['last_activity'] = datetime.now()
        return True
    
    return False

def cleanup_admin_sessions():
    """Clean up expired admin sessions"""
    current_time = datetime.now()
    expired_users = []
    
    for username, session in admin_sessions.items():
        if session['expires'] < current_time:
            expired_users.append(username)
    
    for username in expired_users:
        del admin_sessions[username]

def require_admin_auth(f):
    """Decorator to require admin authentication"""
    def decorated_function(*args, **kwargs):
        username = request.headers.get('X-Admin-Username')
        token = request.headers.get('X-Admin-Token')
        
        if not username or not token:
            return jsonify({
                'success': False,
                'error': 'Admin authentication required'
            }), 401
        
        if not validate_admin_token(username, token):
            return jsonify({
                'success': False,
                'error': 'Invalid or expired admin session'
            }), 401
        
        return f(*args, **kwargs)
    
    decorated_function.__name__ = f.__name__
    return decorated_function

def rate_limit_exceeded(ip, endpoint_type='general', limit=100):
    """Check if rate limit is exceeded"""
    current_time = time.time()
    key = f"{ip}_{endpoint_type}"
    
    if key not in admin_rate_limit:
        admin_rate_limit[key] = []
    
    admin_rate_limit[key] = [t for t in admin_rate_limit[key] if current_time - t < 60]
    
    if len(admin_rate_limit[key]) >= limit:
        return True
    
    admin_rate_limit[key].append(current_time)
    return False

# ==================== DATABASE CONNECTION ====================
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            raise Exception("DATABASE_URL not set")
        
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        url = urllib.parse.urlparse(database_url)
        port = url.port or 5432
        
        conn = psycopg2.connect(
            database=url.path[1:] if url.path.startswith('/') else url.path,
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=port,
            cursor_factory=RealDictCursor,
            connect_timeout=5
        )
        yield conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()

@contextmanager
def get_db_cursor():
    """Context manager for database cursor"""
    conn = None
    cursor = None
    try:
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            raise Exception("DATABASE_URL not set")
        
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        url = urllib.parse.urlparse(database_url)
        port = url.port or 5432
        
        conn = psycopg2.connect(
            database=url.path[1:] if url.path.startswith('/') else url.path,
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=port
        )
        cursor = conn.cursor()
        
        yield cursor
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ Database error: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Store connected clients and active sessions
connected_clients = {}
active_sessions = {}

# ==================== DATABASE INITIALIZATION ====================
def init_db():
    """Initialize database with all required tables"""
    print("🗄️  Initializing PostgreSQL database...")
    
    try:
        with get_db_cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS guardians (
                    guardian_id SERIAL PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    phone TEXT UNIQUE NOT NULL,
                    email TEXT,
                    password_hash VARCHAR(255) NOT NULL,
                    address TEXT,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP,
                    google_id TEXT UNIQUE,
                    auth_provider TEXT DEFAULT 'phone'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drivers (
                    driver_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    address TEXT,
                    phone TEXT NOT NULL,
                    email TEXT,
                    reference_number TEXT UNIQUE,
                    license_number TEXT,
                    guardian_id INTEGER REFERENCES guardians(guardian_id) ON DELETE CASCADE,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id SERIAL PRIMARY KEY,
                    driver_id TEXT NOT NULL,
                    guardian_id INTEGER,
                    severity TEXT NOT NULL CHECK(severity IN ('low', 'medium', 'high')),
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    detection_details TEXT,
                    source TEXT DEFAULT 'system',
                    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id) ON DELETE CASCADE,
                    FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_images (
                    image_id SERIAL PRIMARY KEY,
                    driver_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    capture_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id) ON DELETE CASCADE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drowsiness_events (
                    event_id SERIAL PRIMARY KEY,
                    driver_id TEXT NOT NULL,
                    guardian_id INTEGER,
                    confidence REAL,
                    state TEXT,
                    ear REAL,
                    mar REAL,
                    perclos REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id) ON DELETE CASCADE,
                    FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activity_log (
                    log_id SERIAL PRIMARY KEY,
                    guardian_id INTEGER,
                    admin_username TEXT,
                    action TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS session_tokens (
                    token_id SERIAL PRIMARY KEY,
                    guardian_id INTEGER NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_valid BOOLEAN DEFAULT TRUE,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS admin_activity_log (
                    log_id SERIAL PRIMARY KEY,
                    admin_username TEXT NOT NULL,
                    action TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_drivers_guardian ON drivers(guardian_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_guardian ON alerts(guardian_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tokens_expires ON session_tokens(expires_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tokens_valid ON session_tokens(is_valid)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_drowsiness_driver ON drowsiness_events(driver_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_drowsiness_timestamp ON drowsiness_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_guardians_active ON guardians(is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_guardians_google ON guardians(google_id)')
        
        print("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def update_db_schema():
    """Update existing database schema to add missing columns"""
    print("🔧 Updating database schema...")
    
    try:
        with get_db_cursor() as cursor:
            # Check if google_id column exists in guardians table
            cursor.execute('''
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'guardians' AND column_name = 'google_id'
            ''')
            
            result = cursor.fetchone()
            if not result:
                print("   Adding google_id column to guardians table...")
                cursor.execute('''
                    ALTER TABLE guardians 
                    ADD COLUMN google_id TEXT UNIQUE
                ''')
                print("   ✅ google_id column added")
            else:
                print("   ✅ google_id column already exists")
            
            # Check for other missing columns
            cursor.execute('''
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'guardians' AND column_name = 'auth_provider'
            ''')
            
            result = cursor.fetchone()
            if not result:
                print("   Adding auth_provider column to guardians table...")
                cursor.execute('''
                    ALTER TABLE guardians 
                    ADD COLUMN auth_provider TEXT DEFAULT 'phone'
                ''')
                print("   ✅ auth_provider column added")
            else:
                print("   ✅ auth_provider column already exists")
        
        print("✅ Database schema updated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database schema update failed: {e}")
        return False

# ==================== SESSION MANAGEMENT ====================
def generate_session_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def create_session(guardian_id, ip_address=None, user_agent=None):
    """Create a new session for guardian"""
    token = generate_session_token()
    expires_at = datetime.now() + timedelta(hours=24)
    
    try:
        with get_db_cursor() as cursor:
            # Invalidate any existing sessions for this guardian
            cursor.execute('UPDATE session_tokens SET is_valid = FALSE WHERE guardian_id = %s AND is_valid = TRUE', (guardian_id,))
            
            # Create new session
            cursor.execute('''
                INSERT INTO session_tokens (guardian_id, token, expires_at, ip_address, user_agent)
                VALUES (%s, %s, %s, %s, %s)
            ''', (guardian_id, token, expires_at, ip_address, user_agent))
        
        # Store in memory for quick access
        active_sessions[guardian_id] = {
            'token': token,
            'expires': expires_at,
            'created': datetime.now(),
            'ip_address': ip_address
        }
        
        return token
    except Exception as e:
        print(f"❌ Error creating session: {e}")
        return None

def validate_session(guardian_id, token):
    """Validate a guardian session"""
    if not guardian_id or not token:
        return False
    
    # Check memory cache first
    if guardian_id in active_sessions:
        session_data = active_sessions[guardian_id]
        if (session_data['token'] == token and 
            session_data['expires'] > datetime.now()):
            return True
    
    # Check database
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM session_tokens 
                WHERE guardian_id = %s AND token = %s AND is_valid = TRUE AND expires_at > %s
            ''', (guardian_id, token, datetime.now()))
            
            result = cursor.fetchone()
            # FIX: Handle both dict and tuple
            if isinstance(result, dict):
                count = result['count']
            else:
                count = result[0] if result else 0
            exists = count > 0
            
            if exists and guardian_id not in active_sessions:
                active_sessions[guardian_id] = {
                    'token': token,
                    'expires': datetime.now() + timedelta(hours=23),
                    'created': datetime.now()
                }
            
            return exists
    except Exception as e:
        print(f"❌ Error validating session: {e}")
        return False

def invalidate_session(guardian_id, token=None):
    """Invalidate a guardian session"""
    try:
        with get_db_cursor() as cursor:
            if token:
                cursor.execute('UPDATE session_tokens SET is_valid = FALSE WHERE guardian_id = %s AND token = %s', (guardian_id, token))
            else:
                cursor.execute('UPDATE session_tokens SET is_valid = FALSE WHERE guardian_id = %s', (guardian_id,))
        
        # Remove from memory cache
        if guardian_id in active_sessions:
            del active_sessions[guardian_id]
        
        # Disconnect socket connections
        disconnected = []
        for client_id, client_info in connected_clients.items():
            if client_info.get('guardian_id') == guardian_id:
                disconnected.append(client_id)
        
        for client_id in disconnected:
            if client_id in connected_clients:
                del connected_clients[client_id]
        
        return True
    except Exception as e:
        print(f"❌ Error invalidating session: {e}")
        return False

# ==================== AUTHENTICATION FUNCTIONS ====================
def verify_guardian_credentials(identifier, password):
    """Verify guardian login credentials using bcrypt - Supports email or phone"""
    try:
        print(f"\n🔍 [LOGIN VERIFY] Starting verification")
        print(f"   Identifier received: '{identifier}'")
        print(f"   Password received: '{password}' (length: {len(password)})")
        
        # Check if identifier is email or phone
        is_email = '@' in identifier and '.' in identifier
        
        if is_email:
            # Email login - use as-is
            email = identifier.strip().lower()
            print(f"   Using email login: '{email}'")
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Query by email
                cursor.execute('''
                    SELECT guardian_id, full_name, password_hash, is_active, phone
                    FROM guardians 
                    WHERE email = %s
                ''', (email,))
                
                result = cursor.fetchone()
                
                if not result:
                    print(f"❌ [LOGIN VERIFY] No user found with email: '{email}'")
                    
                    # Debug: Show some emails in DB
                    cursor.execute('SELECT email, full_name FROM guardians WHERE email IS NOT NULL LIMIT 5')
                    all_emails = cursor.fetchall()
                    print(f"   First 5 emails in DB: {[e[0] for e in all_emails]}")
                    
                    return None
                
        else:
            # Phone login - clean and format
            print(f"   Using phone login")
            
            # Clean the phone number
            phone_clean = str(identifier).strip()
            phone_clean = re.sub(r'[\s\-\(\)\+]', '', phone_clean)
            print(f"   Cleaned phone: '{phone_clean}'")
            
            # Check if it's all digits
            if not phone_clean.isdigit():
                print(f"❌ [LOGIN VERIFY] Phone contains non-digits")
                return None
            
            # Convert to 09 format
            lookup_phone = phone_clean
            
            # Check if it's already in 09XXXXXXXXX format (11 digits)
            if len(lookup_phone) == 11 and lookup_phone.startswith('09'):
                # Already in correct format, no conversion needed
                print(f"   Already in 09XXXXXXXXX format, using as-is")
            elif len(lookup_phone) == 12 and lookup_phone.startswith('639'):
                # 639XXXXXXXXX -> 09XXXXXXXXX
                lookup_phone = '09' + lookup_phone[3:]
                print(f"   Converted 639XXXXXXXXX -> {lookup_phone}")
            elif len(lookup_phone) == 11 and lookup_phone.startswith('63'):
                # 63XXXXXXXXX -> 09XXXXXXXXX
                lookup_phone = '09' + lookup_phone[2:]
                print(f"   Converted 63XXXXXXXXX -> {lookup_phone}")
            elif len(lookup_phone) == 10 and lookup_phone.startswith('9'):
                # 9XXXXXXXXX -> 09XXXXXXXXX
                lookup_phone = '0' + lookup_phone
                print(f"   Converted 9XXXXXXXXX -> {lookup_phone}")
            elif len(lookup_phone) == 10:
                # XXXXXXXXXX -> 09XXXXXXXXX (assuming missing 09 prefix)
                lookup_phone = '09' + lookup_phone
                print(f"   Added 09 prefix -> {lookup_phone}")
            else:
                print(f"❌ [LOGIN VERIFY] Invalid phone length: {len(lookup_phone)} digits")
                return None
            
            # Final validation
            if not lookup_phone.startswith('09') or len(lookup_phone) != 11:
                print(f"❌ [LOGIN VERIFY] Invalid final format: '{lookup_phone}' (length: {len(lookup_phone)})")
                return None
            
            print(f"   Looking up in DB as: '{lookup_phone}'")
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Query by phone
                cursor.execute('''
                    SELECT guardian_id, full_name, password_hash, is_active, email
                    FROM guardians 
                    WHERE phone = %s
                ''', (lookup_phone,))
                
                result = cursor.fetchone()
                
                if not result:
                    print(f"❌ [LOGIN VERIFY] No user found with phone: '{lookup_phone}'")
                    
                    # Debug: Show what's in the database
                    cursor.execute('SELECT phone, full_name FROM guardians WHERE phone IS NOT NULL LIMIT 5')
                    all_phones = cursor.fetchall()
                    print(f"   First 5 phones in DB: {[p[0] for p in all_phones]}")
                    
                    return None
        
        # Handle the result (common for both email and phone paths)
        print(f"   Result type: {type(result)}")
        
        # Handle both dictionary and tuple results
        if isinstance(result, dict):
            # Dictionary from RealDictCursor
            guardian_id = result['guardian_id']
            full_name = result['full_name']
            stored_hash = result['password_hash']
            is_active = result['is_active']
            email_or_phone = result.get('email') or result.get('phone') or identifier
            print(f"   Using dictionary access")
        else:
            # Tuple from regular cursor
            # Handle different tuple structures based on which query we ran
            if is_email:
                guardian_id, full_name, stored_hash, is_active, phone = result
                email_or_phone = email
                print(f"   Using tuple unpacking (email query)")
            else:
                guardian_id, full_name, stored_hash, is_active, email = result
                email_or_phone = lookup_phone
                print(f"   Using tuple unpacking (phone query)")
        
        print(f"✅ [LOGIN VERIFY] User found: {full_name} (ID: {guardian_id})")
        
        # Check if account is active
        if not is_active:
            print(f"❌ [LOGIN VERIFY] Account is inactive")
            return None
        
        # Check if we have a hash
        if not stored_hash:
            print(f"❌ [LOGIN VERIFY] No password hash stored for user")
            return None
        
        print(f"   Stored hash: {stored_hash[:30]}...")
        print(f"   Hash length: {len(stored_hash)}")
        print(f"   Is bcrypt format: {stored_hash.startswith('$2') if stored_hash else False}")
        
        # Verify password
        print(f"   Verifying password...")
        
        if verify_password(password, stored_hash):
            print(f"✅ [LOGIN VERIFY] Password verified successfully!")
            
            # Update last login
            try:
                cursor.execute('UPDATE guardians SET last_login = %s WHERE guardian_id = %s', 
                             (datetime.now(), guardian_id))
                conn.commit()
            except Exception as update_error:
                print(f"⚠️ [LOGIN VERIFY] Error updating last login: {update_error}")
            
            return {
                'guardian_id': guardian_id, 
                'full_name': full_name,
                'identifier': email_or_phone,
                'is_email': is_email
            }
        else:
            print(f"❌ [LOGIN VERIFY] Password does not match")
            return None
                
    except Exception as e:
        print(f"❌ [LOGIN VERIFY] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def get_guardian_by_id(guardian_id):
    """Get guardian information by ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT guardian_id, full_name, phone, email, address, registration_date, last_login, auth_provider
                FROM guardians WHERE guardian_id = %s
            ''', (guardian_id,))
            
            result = cursor.fetchone()
            if result:
                return dict(result)
            return None
    except Exception as e:
        print(f"❌ Error getting guardian: {e}")
        return None

def log_activity(guardian_id=None, admin_username=None, action=None, details=None):
    """Log guardian or admin activity"""
    try:
        ip_address = request.remote_addr if request else None
        user_agent = request.headers.get('User-Agent') if request else None
        
        with get_db_cursor() as cursor:
            if admin_username:
                cursor.execute('''
                    INSERT INTO admin_activity_log (admin_username, action, details, ip_address, user_agent)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (admin_username, action, details, ip_address, user_agent))
            else:
                cursor.execute('''
                    INSERT INTO activity_log (guardian_id, action, details, ip_address, user_agent)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (guardian_id, action, details, ip_address, user_agent))
    except Exception as e:
        print(f"⚠️ Error logging activity: {e}")

#region Drivers
# ==================== UTILITY FUNCTIONS ====================
def get_guardian_drivers(guardian_id):
    """Get all drivers registered by a guardian"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT d.*, 
                       (SELECT COUNT(*) FROM alerts a WHERE a.driver_id = d.driver_id AND a.acknowledged = FALSE) as alert_count,
                       (SELECT COUNT(*) FROM face_images f WHERE f.driver_id = d.driver_id) as face_count
                FROM drivers d
                WHERE d.guardian_id = %s AND d.is_active = TRUE
                ORDER BY d.registration_date DESC
            ''', (guardian_id,))
            
            drivers = cursor.fetchall()
            return [dict(driver) for driver in drivers]
    except Exception as e:
        print(f"❌ Error in get_guardian_drivers: {e}")
        return []

#end Region Drivers

#region Facial Recognition 
def get_face_embedding_from_base64(image_base64):
    """Extract face embedding using face_recognition (dlib-based, no TensorFlow)"""
    try:
        # Remove header if present (e.g., "data:image/jpeg;base64,")
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64 to image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image.convert('RGB'))
        
        # Detect faces
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            print("   No face detected in image")
            return None
        
        # Get face encodings (128-dimensional vectors)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if not face_encodings:
            print("   Could not encode face")
            return None
        
        # Return first face encoding as list (for JSON serialization)
        embedding = face_encodings[0].tolist()
        print(f"   ✅ Face embedding generated (length: {len(embedding)})")
        return embedding
        
    except Exception as e:
        print(f"❌ Error extracting embedding: {e}")
        return None

#end Region Facial Recognition

def get_recent_alerts(guardian_id, limit=10):
    """Get recent alerts for a guardian"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT a.*, d.name as driver_name
                FROM alerts a
                JOIN drivers d ON a.driver_id = d.driver_id
                WHERE a.guardian_id = %s
                ORDER BY a.timestamp DESC
                LIMIT %s
            ''', (guardian_id, limit))
            
            alerts = cursor.fetchall()
            result = []
            for alert in alerts:
                alert_dict = dict(alert)
                if alert_dict.get('detection_details'):
                    try:
                        alert_dict['detection_details'] = json.loads(alert_dict['detection_details'])
                    except:
                        pass
                result.append(alert_dict)
            return result
    except Exception as e:
        print(f"❌ Error in get_recent_alerts: {e}")
        return []

def get_driver_by_name_or_id(identifier):
    """Get driver by name or ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Try by ID first
            cursor.execute('''
                SELECT d.*, g.full_name as guardian_name, g.phone as guardian_phone
                FROM drivers d
                JOIN guardians g ON d.guardian_id = g.guardian_id
                WHERE d.driver_id = %s AND d.is_active = TRUE
            ''', (identifier,))
            
            result = cursor.fetchone()
            
            # If not found by ID, try by name
            if not result:
                cursor.execute('''
                    SELECT d.*, g.full_name as guardian_name, g.phone as guardian_phone
                    FROM drivers d
                    JOIN guardians g ON d.guardian_id = g.guardian_id
                    WHERE d.name LIKE %s AND d.is_active = TRUE
                ''', (f'%{identifier}%',))
                result = cursor.fetchone()
            
            if result:
                return dict(result)
            return None
    except Exception as e:
        print(f"❌ Error getting driver: {e}")
        return None

# ==================== SECURITY MIDDLEWARE ====================
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    # CORS headers for Firebase Hosting
    origin = request.headers.get('Origin')
    if origin in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, X-Admin-Username, X-Admin-Token'
    
    # Standard security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self' https://guardian-drive-app.web.app; "
        "script-src 'self' https://cdn.jsdelivr.net https://accounts.google.com 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "connect-src 'self' wss://driver-drowsiness-with-alert.onrender.com https://driver-drowsiness-with-alert.onrender.com https://accounts.google.com; "
        "frame-src https://accounts.google.com;"
        "worker-src 'self' blob:;"
    )
    
    # Cache control
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    
    return response

# ==================== SOCKET.IO HANDLERS ====================
@socketio.on('connect')
def handle_connect():
    """Handle client connection with better error handling"""
    client_id = request.sid
    connected_clients[client_id] = {
        'connected_at': datetime.now(),
        'ip': request.remote_addr,
        'type': None,
        'guardian_id': None,
        'authenticated': False,
        'last_ping': datetime.now(),
        'user_agent': request.headers.get('User-Agent', 'Unknown'),
        'origin': request.headers.get('Origin', 'Unknown'),
        'transport': request.environ.get('HTTP_UPGRADE', 'polling')
    }
    
    print(f"\n{'='*60}")
    print(f"✅ WebSocket client connected: {client_id}")
    print(f"   IP: {request.remote_addr}")
    print(f"   Origin: {request.headers.get('Origin', 'Unknown')}")
    print(f"   Transport: {request.environ.get('HTTP_UPGRADE', 'polling')}")
    print(f"   User-Agent: {request.headers.get('User-Agent', 'Unknown')[:50]}...")
    print(f"{'='*60}\n")
    
    # Send immediate acknowledgment with connection details
    emit('connected', {
        'status': 'connected',
        'client_id': client_id,
        'timestamp': datetime.now().isoformat(),
        'transport': 'websocket' if request.environ.get('HTTP_UPGRADE') == 'websocket' else 'polling',
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection with cleanup"""
    client_id = request.sid
    if client_id in connected_clients:
        client_info = connected_clients[client_id]
        connected_duration = datetime.now() - client_info['connected_at']
        
        print(f"\n{'='*60}")
        print(f"⚠️ WebSocket client disconnected: {client_id}")
        print(f"   Guardian ID: {client_info.get('guardian_id', 'None')}")
        print(f"   Authenticated: {client_info.get('authenticated', False)}")
        print(f"   Connected for: {connected_duration.total_seconds():.1f} seconds")
        print(f"   IP: {client_info.get('ip', 'Unknown')}")
        print(f"{'='*60}\n")
        
        # Clean up
        del connected_clients[client_id]

@socketio.on('guardian_authenticate')
def handle_guardian_auth(data):
    """Guardian authentication via WebSocket - FIXED for Google auth"""
    client_id = request.sid
    guardian_id = data.get('guardian_id')
    token = data.get('token')
    auth_provider = data.get('auth_provider', 'unknown')
    
    print(f"\n🔐 WebSocket authentication attempt:")
    print(f"   Client ID: {client_id}")
    print(f"   Guardian ID: {guardian_id}")
    print(f"   Auth Provider: {auth_provider}")
    print(f"   Token present: {bool(token)}")
    print(f"   Token length: {len(token) if token else 0}")
    
    if not guardian_id or not token:
        print(f"❌ Missing credentials from client {client_id}")
        emit('auth_failed', {'error': 'Missing guardian_id or token'})
        return
    
    # Validate session (works for both phone and Google auth)
    is_valid = validate_session(guardian_id, token)
    
    if is_valid:
        print(f"   ✅ Session validated successfully")
        
        # Check for existing connection and disconnect it
        for existing_id, client_info in list(connected_clients.items()):
            if (client_info.get('guardian_id') == guardian_id and 
                client_info.get('authenticated') and 
                existing_id != client_id):
                print(f"   Disconnecting old connection: {existing_id}")
                try:
                    socketio.disconnect(existing_id, silent=True)
                except:
                    pass
                if existing_id in connected_clients:
                    del connected_clients[existing_id]
        
        # Update client info
        if client_id in connected_clients:
            connected_clients[client_id]['type'] = 'guardian'
            connected_clients[client_id]['guardian_id'] = guardian_id
            connected_clients[client_id]['authenticated'] = True
            connected_clients[client_id]['auth_time'] = datetime.now()
            connected_clients[client_id]['auth_provider'] = auth_provider
        
        # Join guardian room for targeted messages
        try:
            socketio.enter_room(client_id, f"guardian_{guardian_id}")
            print(f"   Joined room: guardian_{guardian_id}")
        except Exception as e:
            print(f"   Error joining room: {e}")
        
        # Get guardian info
        guardian = get_guardian_by_id(guardian_id)
        if guardian:
            print(f"✅ Guardian authenticated successfully: {guardian.get('full_name')}")
            emit('auth_confirmed', {
                'success': True,
                'guardian_id': guardian_id,
                'full_name': guardian.get('full_name', ''),
                'phone': guardian.get('phone', ''),
                'auth_provider': guardian.get('auth_provider', auth_provider),
                'timestamp': datetime.now().isoformat(),
                'message': 'WebSocket authentication successful'
            })
            return
        else:
            # Still success even if we can't get details
            emit('auth_confirmed', {
                'success': True,
                'guardian_id': guardian_id,
                'timestamp': datetime.now().isoformat()
            })
            return
    
    # If validation fails, check if this is a Google auth token that might need special handling
    if auth_provider == 'google':
        print(f"⚠️ Google auth token validation failed, checking alternative...")
        # You could add Google token validation here if needed
    
    print(f"❌ WebSocket authentication failed for client: {client_id}")
    emit('auth_failed', {
        'error': 'Authentication failed',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('ping')
def handle_ping():
    """Handle ping from client to keep connection alive"""
    client_id = request.sid
    if client_id in connected_clients:
        connected_clients[client_id]['last_ping'] = datetime.now()
        emit('pong', {
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id
        })

@socketio.on('error')
def handle_error(error):
    """Handle socket errors"""
    client_id = request.sid
    print(f"❌ Socket error for client {client_id}: {error}")

# ==================== MAIN ROUTES ====================
@app.route('/')
def serve_home():
    """Redirect to Firebase Hosting - pure backend server"""
    return jsonify({
        'success': True,
        'message': 'Driver Alert System API Server',
        'backend': 'Render.com',
        'frontend': 'Firebase Hosting',
        'frontend_url': 'https://guardian-drive-app.web.app',
        'api_docs': 'https://driver-drowsiness-with-alert.onrender.com/api/health',
        'version': '2.0.0',
        'cloudinary_enabled': CLOUDINARY_ENABLED
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health_info = {
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'server': 'Driver Alert System API',
            'version': '2.0.0',
            'connected_clients': len(connected_clients),
            'active_sessions': len(active_sessions),
            'database': 'postgresql',
            'google_auth': bool(GOOGLE_CLIENT_ID),
            'cloudinary_enabled': CLOUDINARY_ENABLED,
            'firebase_integration': True,
            'allowed_origins': ALLOWED_ORIGINS,
            'websocket_connections': len(connected_clients),
            'frontend_url': 'https://guardian-drive-app.web.app',
            'note': 'This is a pure API server. Frontend is hosted on Firebase.'
        }
        
        return jsonify(health_info)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'running_with_errors',
            'error': str(e)
        }), 200

# =================== WEBSOCKET CHECKER ======================
@app.route('/api/websocket-status', methods=['GET'])
def websocket_status():
    """Check WebSocket server status"""
    try:
        # Count authenticated clients
        authenticated_count = sum(1 for c in connected_clients.values() if c.get('authenticated'))
        
        # Group by guardian
        guardians_online = {}
        for client_id, info in connected_clients.items():
            if info.get('authenticated') and info.get('guardian_id'):
                guardians_online[info['guardian_id']] = {
                    'connected_since': info['auth_time'].isoformat() if info.get('auth_time') else None,
                    'last_ping': info['last_ping'].isoformat() if info.get('last_ping') else None,
                    'auth_provider': info.get('auth_provider', 'unknown')
                }
        
        return jsonify({
            'success': True,
            'websocket_enabled': True,
            'total_connections': len(connected_clients),
            'authenticated_connections': authenticated_count,
            'guardians_online': len(guardians_online),
            'guardians': guardians_online,
            'transports': ['polling', 'websocket'],
            'socketio_path': '/socket.io/',
            'websocket_url': 'wss://driver-drowsiness-with-alert.onrender.com/socket.io/',
            'allowed_origins': ALLOWED_ORIGINS,
            'server_time': datetime.now().isoformat(),
            'python_version': '3.10.11'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'websocket_enabled': True,
            'total_connections': len(connected_clients)
        }), 200

# ==================== GOOGLE AUTH CONFIG ====================
@app.route('/api/config/google', methods=['GET'])
def get_google_config():
    """Get Google OAuth configuration for frontend"""
    return jsonify({
        'success': True,
        'google_client_id': GOOGLE_CLIENT_ID,
        'firebase_domain': 'guardian-drive-app.web.app',
        'backend_url': 'https://driver-drowsiness-with-alert.onrender.com',
        'websocket_url': 'wss://driver-drowsiness-with-alert.onrender.com',
        'message': 'Google OAuth configuration loaded',
        'cloudinary_enabled': CLOUDINARY_ENABLED
    })

@app.route('/api/firebase-config', methods=['GET'])
def get_firebase_config():
    """Get Firebase configuration for clients"""
    return jsonify({
        'success': True,
        'firebase_domain': 'guardian-drive-app.web.app',
        'backend_url': 'https://driver-drowsiness-with-alert.onrender.com',
        'websocket_url': 'wss://driver-drowsiness-with-alert.onrender.com',
        'api_base': 'https://driver-drowsiness-with-alert.onrender.com/api',
        'message': 'Firebase Hosting configuration',
        'cloudinary_enabled': CLOUDINARY_ENABLED
    })

# ==================== GOOGLE LOGIN ENDPOINT ====================
@app.route('/api/google-login', methods=['POST'])
def google_login():
    """Handle Google OAuth login - FIXED VERSION"""
    try:
        data = request.json
        google_token = data.get('token')
        
        if not google_token:
            return jsonify({
                'success': False,
                'error': 'Google token required'
            }), 400
        
        print(f"🔐 [GOOGLE LOGIN] Received Google token")
        print(f"   Token preview: {google_token[:30]}...")
        
        email = None
        name = None
        google_id = None
        
        # Check if GOOGLE_CLIENT_ID is set
        if not GOOGLE_CLIENT_ID:
            print(f"❌ [GOOGLE LOGIN] GOOGLE_CLIENT_ID not set in environment")
            return jsonify({
                'success': False,
                'error': 'Google authentication not configured on server'
            }), 500
        
        try:
            # Try to verify the Google token
            from google.oauth2 import id_token
            from google.auth.transport import requests
            
            idinfo = id_token.verify_oauth2_token(
                google_token, 
                requests.Request(),
                GOOGLE_CLIENT_ID
            )
            
            email = idinfo.get('email')
            name = idinfo.get('name', '')
            google_id = idinfo.get('sub')
            
            print(f"✅ Google token verified: {email}")
            print(f"   Name: {name}")
            print(f"   Google ID: {google_id}")
            
        except ValueError as e:
            # Invalid token
            print(f"❌ Google token verification failed: {e}")
            
            # Try to decode without verification as fallback
            try:
                import jwt
                unverified_payload = jwt.decode(
                    google_token, 
                    options={"verify_signature": False}
                )
                email = unverified_payload.get('email')
                name = unverified_payload.get('name', '')
                google_id = unverified_payload.get('sub')
                
                print(f"⚠️ Using unverified token data: {email}")
            except Exception as decode_error:
                print(f"❌ Could not decode token: {decode_error}")
                return jsonify({
                    'success': False,
                    'error': 'Invalid Google token'
                }), 401
        except Exception as verify_error:
            print(f"❌ Unexpected error during token verification: {verify_error}")
            return jsonify({
                'success': False,
                'error': f'Token verification failed: {str(verify_error)}'
            }), 401
        
        if not email:
            return jsonify({
                'success': False,
                'error': 'Email not found in token'
            }), 400
        
        # Database operations
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if user exists by Google ID or email
                cursor.execute('''
                    SELECT guardian_id, full_name, email, phone, auth_provider, google_id
                    FROM guardians 
                    WHERE google_id = %s OR email = %s
                    LIMIT 1
                ''', (google_id, email))
                
                result = cursor.fetchone()
                
                if result:
                    # User exists - login
                    if isinstance(result, dict):
                        guardian_id = result['guardian_id']
                        full_name = result['full_name']
                        stored_email = result['email']
                        stored_google_id = result['google_id']
                    else:
                        # Handle tuple result
                        guardian_id, full_name, stored_email, phone, auth_provider, stored_google_id = result
                    
                    print(f"✅ Existing user found: {full_name} ({stored_email})")
                    
                    # Update Google ID if not set
                    if not stored_google_id and google_id:
                        cursor.execute('UPDATE guardians SET google_id = %s WHERE guardian_id = %s', 
                                     (google_id, guardian_id))
                        conn.commit()
                        print(f"   Updated Google ID for user: {google_id}")
                    
                    # Update last login
                    cursor.execute('UPDATE guardians SET last_login = %s WHERE guardian_id = %s', 
                                 (datetime.now(), guardian_id))
                    conn.commit()
                    
                else:
                    # Create new user with Google data
                    import random
                    import secrets
                    
                    # Generate a unique phone number
                    phone = '09' + ''.join([str(random.randint(0, 9)) for _ in range(9)])
                    
                    # Generate a random password hash
                    temp_password = secrets.token_urlsafe(16)
                    password_hash = hash_password(temp_password)
                    
                    cursor.execute('''
                        INSERT INTO guardians (
                            full_name, 
                            email, 
                            phone, 
                            password_hash, 
                            is_active,
                            registration_date,
                            last_login,
                            google_id,
                            auth_provider
                        )
                        VALUES (%s, %s, %s, %s, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %s, 'google')
                        RETURNING guardian_id
                    ''', (name, email, phone, password_hash, google_id))
                    
                    result = cursor.fetchone()
                    if isinstance(result, dict):
                        guardian_id = result['guardian_id']
                    else:
                        guardian_id = result[0] if result else None
                    
                    full_name = name
                    conn.commit()
                    
                    print(f"✅ New Google user created: {full_name} (ID: {guardian_id})")
                
                # Create session
                token = create_session(guardian_id, request.remote_addr, request.headers.get('User-Agent'))
                
                # Log activity
                log_activity(guardian_id, 'GOOGLE_LOGIN', f'Google login from {request.remote_addr}')
                
                return jsonify({
                    'success': True,
                    'guardian_id': guardian_id,
                    'full_name': full_name,
                    'email': email,
                    'session_token': token,
                    'auth_provider': 'google',
                    'message': 'Google login successful',
                    'is_google_user': True,
                    'redirect_url': f'https://guardian-drive-app.web.app/guardian-dashboard.html?guardian_id={guardian_id}&token={token}'
                })
                
        except Exception as db_error:
            print(f"❌ Database error in google_login: {db_error}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Database error: {str(db_error)}'
            }), 500
        
    except Exception as e:
        print(f"❌ Google login error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Google login failed: {str(e)}'
        }), 500

@app.route('/api/google-login-with-email', methods=['POST'])
def google_login_with_email():
    """Google login with manual email input"""
    try:
        data = request.json
        email = data.get('email')
        name = data.get('name')
        
        if not email:
            return jsonify({
                'success': False,
                'error': 'Email is required'
            }), 400
        
        if not name:
            # Extract name from email
            name = email.split('@')[0].replace('.', ' ').title()
        
        google_id = f"google_{hash(email) % 1000000}"
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if user exists by email
            cursor.execute('''
                SELECT guardian_id, full_name, email, auth_provider
                FROM guardians 
                WHERE email = %s
                LIMIT 1
            ''', (email,))
            
            result = cursor.fetchone()
            
            if result:
                # User exists - login
                if isinstance(result, dict):
                    guardian_id = result['guardian_id']
                    full_name = result['full_name']
                    stored_email = result['email']
                else:
                    guardian_id, full_name, stored_email, auth_provider = result
                
                # Update auth provider if not google
                if auth_provider != 'google':
                    cursor.execute('UPDATE guardians SET auth_provider = %s WHERE guardian_id = %s', 
                                 ('google', guardian_id))
                    conn.commit()
                
                print(f"✅ User found: {full_name} ({stored_email})")
                
            else:
                # Create new user
                import random
                phone = '09' + ''.join([str(random.randint(0, 9)) for _ in range(9)])
                
                cursor.execute('''
                    INSERT INTO guardians (
                        full_name, 
                        email, 
                        phone, 
                        password_hash, 
                        is_active,
                        registration_date,
                        last_login,
                        google_id,
                        auth_provider
                    )
                    VALUES (%s, %s, %s, %s, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %s, 'google')
                    RETURNING guardian_id
                ''', (name, email, phone, hash_password(secrets.token_urlsafe(16)), google_id))
                
                result = cursor.fetchone()
                guardian_id = result[0] if result else None
                full_name = name
                conn.commit()
                
                print(f"✅ New user created: {full_name} (ID: {guardian_id})")
        
        token = create_session(guardian_id, request.remote_addr, request.headers.get('User-Agent'))
        
        return jsonify({
            'success': True,
            'guardian_id': guardian_id,
            'full_name': full_name,
            'email': email,
            'session_token': token,
            'auth_provider': 'google',
            'message': 'Login successful',
            'method': 'email_based_google_login',
            'redirect_url': f'https://guardian-drive-app.web.app/guardian-dashboard.html?guardian_id={guardian_id}&token={token}'
        })
        
    except Exception as e:
        print(f"❌ Google login with email error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/google-login-simple', methods=['POST'])
def google_login_simple():
    """Simplified Google login for testing"""
    try:
        data = request.json
        email = data.get('email')
        name = data.get('name')
        
        if not email:
            return jsonify({'success': False, 'error': 'Email required'}), 400
        
        if not name:
            name = email.split('@')[0].replace('.', ' ').title()
        
        google_id = f"google_{hash(email) % 1000000}"
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if user exists by email
            cursor.execute('''
                SELECT guardian_id, full_name, email
                FROM guardians 
                WHERE email = %s
                LIMIT 1
            ''', (email,))
            
            result = cursor.fetchone()
            
            if result:
                guardian_id = result['guardian_id']
                full_name = result['full_name']
            else:
                # Create new user
                import random
                phone = '09' + ''.join([str(random.randint(0, 9)) for _ in range(9)])
                
                cursor.execute('''
                    INSERT INTO guardians (
                        full_name, email, phone, password_hash, 
                        is_active, google_id, auth_provider
                    )
                    VALUES (%s, %s, %s, %s, TRUE, %s, 'google')
                    RETURNING guardian_id
                ''', (name, email, phone, hash_password(secrets.token_urlsafe(16)), google_id))
                
                result = cursor.fetchone()
                guardian_id = result[0] if result else None
                full_name = name
                conn.commit()
        
        token = create_session(guardian_id, request.remote_addr, request.headers.get('User-Agent'))
        
        return jsonify({
            'success': True,
            'guardian_id': guardian_id,
            'full_name': full_name,
            'email': email,
            'session_token': token,
            'auth_provider': 'google',
            'message': 'Google login successful',
            'redirect_url': f'https://guardian-drive-app.web.app/guardian-dashboard.html?guardian_id={guardian_id}&token={token}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== GUARDIAN AUTHENTICATION ====================
@app.route('/api/login', methods=['POST'])
def login():
    """Guardian login with email/phone + password"""
    try:
        data = request.json
        identifier = data.get('identifier', '').strip()   # could be email or phone
        password = data.get('password', '')

        print(f"\n🔑 [LOGIN API] Attempting login with identifier: '{identifier}'")

        if not identifier or not password:
            return jsonify({'success': False, 'error': 'Identifier and password required'}), 400

        # Rate limiting
        ip = request.remote_addr
        if rate_limit_exceeded(ip, 'guardian_login', limit=10):
            return jsonify({'success': False, 'error': 'Too many login attempts'}), 429

        # Verify credentials (supports email or phone)
        guardian = verify_guardian_credentials(identifier, password)

        if guardian:
            token = create_session(guardian['guardian_id'], request.remote_addr, request.headers.get('User-Agent'))
            log_activity(guardian['guardian_id'], 'LOGIN', f'Logged in from {request.remote_addr}')

            # Get full user details for response
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute('''
                    SELECT guardian_id, full_name, email, phone
                    FROM guardians
                    WHERE guardian_id = %s
                ''', (guardian['guardian_id'],))
                user_details = cursor.fetchone()

            return jsonify({
                'success': True,
                'guardian_id': guardian['guardian_id'],
                'full_name': guardian['full_name'],
                'email': user_details['email'] if user_details else '',
                'phone': user_details['phone'] if user_details else '',
                'session_token': token,
                'auth_provider': 'phone',
                'message': 'Login successful',
                'redirect_url': f'https://guardian-drive-app.web.app/guardian-dashboard.html?guardian_id={guardian["guardian_id"]}&token={token}'
            })

        return jsonify({'success': False, 'error': 'Invalid identifier or password'}), 401

    except Exception as e:
        print(f"❌ Login error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Login failed. Please try again.'}), 500
    
@app.route('/api/troubleshoot-google-auth', methods=['GET'])
def troubleshoot_google_auth():
    """Comprehensive Google Auth troubleshooting endpoint"""
    try:
        import socket
        import requests
        from datetime import datetime
        
        # Get request information
        request_info = {
            'remote_addr': request.remote_addr,
            'origin': request.headers.get('Origin', 'Not provided'),
            'user_agent': request.headers.get('User-Agent', 'Not provided')[:100]
        }
        
        results = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'request_info': request_info,
            'google_client_id': {
                'value': os.environ.get('GOOGLE_CLIENT_ID', 'NOT SET'),
                'length': len(os.environ.get('GOOGLE_CLIENT_ID', '')),
                'format_valid': 'googleusercontent.com' in os.environ.get('GOOGLE_CLIENT_ID', ''),
                'starts_with_numbers': os.environ.get('GOOGLE_CLIENT_ID', '').split('-')[0].isdigit() if '-' in os.environ.get('GOOGLE_CLIENT_ID', '') else False,
                'exact_value': os.environ.get('GOOGLE_CLIENT_ID', 'NOT SET')  # Full value for debugging
            },
            'google_client_secret': {
                'present': bool(os.environ.get('GOOGLE_CLIENT_SECRET')),
                'length': len(os.environ.get('GOOGLE_CLIENT_SECRET', '')) if os.environ.get('GOOGLE_CLIENT_SECRET') else 0,
                'preview': os.environ.get('GOOGLE_CLIENT_SECRET', '')[:10] + '...' if os.environ.get('GOOGLE_CLIENT_SECRET') else None
            },
            'cloudinary': {
                'enabled': CLOUDINARY_ENABLED,
                'cloud_name': CLOUDINARY_CLOUD_NAME if CLOUDINARY_CLOUD_NAME else 'Not configured'
            },
            'environment': {
                'render': bool(os.environ.get('RENDER')),
                'render_service': os.environ.get('RENDER_SERVICE_NAME', 'unknown'),
                'database_configured': bool(os.environ.get('DATABASE_URL'))
            },
            'dns_resolution': [],
            'google_apis_reachable': False,
            'token_verification_test': None,
            'common_issues': [],
            'recommendations': []
        }
        
        # Test DNS resolution for Google APIs
        google_hosts = ['accounts.google.com', 'www.googleapis.com', 'oauth2.googleapis.com']
        for host in google_hosts:
            try:
                ip = socket.gethostbyname(host)
                results['dns_resolution'].append(f"✅ {host} resolved to {ip}")
            except Exception as e:
                results['dns_resolution'].append(f"❌ {host} DNS failed: {str(e)}")
                results['common_issues'].append(f"DNS issue with {host}")
        
        # Test connectivity to Google APIs
        try:
            response = requests.get('https://accounts.google.com/.well-known/openid-configuration', timeout=5)
            if response.status_code == 200:
                results['google_apis_reachable'] = True
                results['dns_resolution'].append("✅ Google APIs are reachable")
            else:
                results['google_apis_reachable'] = False
                results['dns_resolution'].append(f"⚠️ Google APIs returned status {response.status_code}")
        except Exception as e:
            results['google_apis_reachable'] = False
            results['dns_resolution'].append(f"❌ Cannot reach Google APIs: {str(e)}")
        
        # Test if we can import Google libraries
        try:
            from google.oauth2 import id_token
            from google.auth.transport import requests as google_requests
            results['token_verification_test'] = "✅ Google libraries are importable"
            
            # Test if GOOGLE_CLIENT_ID is valid format
            client_id = os.environ.get('GOOGLE_CLIENT_ID', '')
            if client_id and 'googleusercontent.com' in client_id:
                # Try a mock verification (without actual token) to check library
                results['token_verification_test'] += " - Libraries ready for verification"
            else:
                results['token_verification_test'] += " - Libraries OK, but client ID missing/invalid"
                
        except ImportError as e:
            results['token_verification_test'] = f"❌ Missing Google libraries: {str(e)}"
            results['common_issues'].append("Google Auth libraries not installed")
            results['recommendations'].append("Run: pip install google-auth")
        
        # Check for common configuration issues
        client_id = os.environ.get('GOOGLE_CLIENT_ID', '')
        
        if not client_id:
            results['common_issues'].append("❌ GOOGLE_CLIENT_ID environment variable is not set")
            results['recommendations'].append("Add GOOGLE_CLIENT_ID to your environment variables in Render")
        elif client_id == 'your-google-client-id-here':
            results['common_issues'].append("❌ GOOGLE_CLIENT_ID is still set to placeholder value")
            results['recommendations'].append("Replace with your actual Google Client ID from Google Cloud Console")
        elif len(client_id) < 50:
            results['common_issues'].append(f"⚠️ GOOGLE_CLIENT_ID seems too short ({len(client_id)} chars) - may be truncated")
            results['recommendations'].append("Check if the full client ID is copied correctly (should be ~72 chars)")
        elif not client_id.endswith('.apps.googleusercontent.com'):
            results['common_issues'].append("❌ GOOGLE_CLIENT_ID has incorrect format")
            results['recommendations'].append("Client ID should end with .apps.googleusercontent.com")
        else:
            results['google_client_id']['valid_format'] = True
            results['google_client_id']['preview'] = client_id[:30] + '...' + client_id[-20:]
        
        # Check client secret
        if not os.environ.get('GOOGLE_CLIENT_SECRET'):
            results['common_issues'].append("⚠️ GOOGLE_CLIENT_SECRET is not set (may be needed for some operations)")
            results['recommendations'].append("Add GOOGLE_CLIENT_SECRET to environment variables in Render")
        elif len(os.environ.get('GOOGLE_CLIENT_SECRET', '')) < 20:
            results['common_issues'].append("⚠️ GOOGLE_CLIENT_SECRET seems too short - may be invalid")
            results['recommendations'].append("Verify your client secret in Google Cloud Console")
        
        # Check if the exact client ID from your screenshot is being used
        expected_client_id = "164210458784-d9025r5nmdcjoo7enpa64b5aphdljras.apps.googleusercontent.com"
        if client_id and client_id != expected_client_id and '164210458784' in client_id:
            results['common_issues'].append("⚠️ Client ID starts correctly but doesn't match the exact one from your screenshot")
            results['recommendations'].append(f"Update to exact client ID: {expected_client_id}")
        
        # Check CORS/origins
        results['allowed_origins'] = ALLOWED_ORIGINS
        
        # Check if the request origin is allowed
        request_origin = request.headers.get('Origin')
        if request_origin:
            if request_origin in ALLOWED_ORIGINS:
                results['cors_status'] = f"✅ Origin '{request_origin}' is allowed"
            else:
                results['cors_status'] = f"⚠️ Origin '{request_origin}' is NOT in allowed origins"
                results['recommendations'].append(f"Add '{request_origin}' to ALLOWED_ORIGINS if this is your frontend")
        else:
            results['cors_status'] = "No Origin header in request"
        
        # Test database connection (basic)
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM guardians')
                count_result = cursor.fetchone()
                guardian_count = count_result[0] if count_result else 0
                results['database'] = {
                    'connected': True,
                    'guardian_count': guardian_count
                }
        except Exception as db_error:
            results['database'] = {
                'connected': False,
                'error': str(db_error)
            }
            results['common_issues'].append(f"Database connection issue: {str(db_error)}")
        
        # Provide overall assessment
        if len(results['common_issues']) == 0:
            results['assessment'] = "✅ All checks passed! Google Auth should work properly."
            results['status_emoji'] = "✅"
        else:
            results['assessment'] = f"⚠️ Found {len(results['common_issues'])} potential issue(s):"
            for issue in results['common_issues']:
                results['assessment'] += f"\n   • {issue}"
            results['status_emoji'] = "⚠️"
        
        # Add specific recommendation based on your screenshot
        if '164210458784' in client_id and 'joo' not in client_id:
            results['specific_fix'] = {
                'issue': "Missing 'o' in client ID",
                'current': client_id,
                'correct': "164210458784-d9025r5nmdcjoo7enpa64b5aphdljras.apps.googleusercontent.com",
                'action': "Update your GOOGLE_CLIENT_ID to include the missing 'o' (should be 'joo' not 'jo')"
            }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout guardian and invalidate session"""
    try:
        data = request.json or {}
        guardian_id = data.get('guardian_id')
        token = data.get('token')
        
        if not guardian_id or not token:
            return jsonify({
                'success': False,
                'error': 'Missing authentication data'
            }), 400
        
        # Validate session before logout
        if not validate_session(guardian_id, token):
            return jsonify({
                'success': False,
                'error': 'Invalid or expired session'
            }), 401
        
        # Invalidate session
        invalidate_session(guardian_id, token)
        log_activity(guardian_id, 'LOGOUT', 'Guardian logged out')
        
        return jsonify({
            'success': True,
            'message': 'Logged out successfully',
            'redirect_url': 'https://guardian-drive-app.web.app/?logged_out=true'
        })
        
    except Exception as e:
        print(f"❌ Logout error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/validate-session', methods=['POST'])
def validate_session_endpoint():
    """Validate a guardian session"""
    try:
        data = request.json or {}
        guardian_id = data.get('guardian_id')
        token = data.get('token')
        
        if not guardian_id or not token:
            return jsonify({
                'success': False,
                'valid': False,
                'error': 'Missing authentication data'
            }), 400
        
        is_valid = validate_session(guardian_id, token)
        
        return jsonify({
            'success': True,
            'valid': is_valid,
            'guardian_id': guardian_id if is_valid else None,
            'message': 'Session valid' if is_valid else 'Session invalid'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'valid': False,
            'error': str(e)
        }), 500

@app.route('/api/register-guardian', methods=['POST'])
def register_guardian():
    try:
        print("🔍 [REGISTRATION] Endpoint called")
        
        # Check if request has JSON
        if not request.is_json:
            print("❌ [REGISTRATION] No JSON data received")
            return jsonify({
                'success': False,
                'error': 'Invalid request format'
            }), 400
        
        data = request.json
        print(f"🔍 [REGISTRATION] Received data: {data}")
        
        required = ['full_name', 'phone', 'password']
        for field in required:
            if field not in data or not str(data[field]).strip():
                print(f"❌ [REGISTRATION] Missing field: {field}")
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Get values
        full_name = data['full_name'].strip()
        phone = data['phone'].strip()
        password = data['password']
        
        # 🔥 FIX: Clean password only once
        password = clean_password(password)
        print(f"🔍 [REGISTRATION] Password after cleaning: '{password}' (length: {len(password)})")
        
        # Validate password length (after cleaning)
        if len(password) < 6:
            print(f"❌ [REGISTRATION] Password too short: {len(password)} chars")
            return jsonify({
                'success': False,
                'error': 'Password must be at least 6 characters long'
            }), 400
        
        # Clean phone number
        phone_clean = re.sub(r'[\s\-\(\)\+]', '', phone)
        print(f"🔍 [REGISTRATION] Cleaned phone (digits only): '{phone_clean}'")
        
        if not phone_clean.isdigit():
            print(f"❌ [REGISTRATION] Phone contains non-digits: '{phone}'")
            return jsonify({
                'success': False,
                'error': 'Phone number can only contain digits'
            }), 400
        
        # Convert to 09XXXXXXXXX format
        final_phone = phone_clean
        
        # Check if it's already in 09XXXXXXXXX format (11 digits)
        if len(final_phone) == 11 and final_phone.startswith('09'):
            # Already in correct format, no conversion needed
            print(f"✅ [REGISTRATION] Phone already in correct 09XXXXXXXXX format")
        elif len(final_phone) == 12 and final_phone.startswith('639'):
            # 639XXXXXXXXX -> 09XXXXXXXXX
            final_phone = '09' + final_phone[3:]
            print(f"✅ [REGISTRATION] Converted 639XXXXXXXXX -> {final_phone}")
        elif len(final_phone) == 11 and final_phone.startswith('63'):
            # 63XXXXXXXXX -> 09XXXXXXXXX
            final_phone = '09' + final_phone[2:]
            print(f"✅ [REGISTRATION] Converted 63XXXXXXXXX -> {final_phone}")
        elif len(final_phone) == 10 and final_phone.startswith('9'):
            # 9XXXXXXXXX -> 09XXXXXXXXX
            final_phone = '0' + final_phone
            print(f"✅ [REGISTRATION] Converted 9XXXXXXXXX -> {final_phone}")
        elif len(final_phone) == 10:
            # XXXXXXXXXX -> 09XXXXXXXXX (assuming missing 09 prefix)
            final_phone = '09' + final_phone
            print(f"✅ [REGISTRATION] Added 09 prefix -> {final_phone}")
        else:
            print(f"❌ [REGISTRATION] Invalid phone length: {len(final_phone)} digits")
            return jsonify({
                'success': False,
                'error': f'Phone number must be 11 digits (09XXXXXXXXX). Current: {len(final_phone)} digits'
            }), 400
        
        # Final validation
        if not final_phone.startswith('09') or len(final_phone) != 11:
            print(f"❌ [REGISTRATION] Invalid final format: '{final_phone}' (length: {len(final_phone)})")
            return jsonify({
                'success': False,
                'error': 'Phone number must be 11 digits starting with 09'
            }), 400
        
        print(f"✅ [REGISTRATION] Final phone to store: '{final_phone}'")
        
        # Database operations
        with get_db_cursor() as cursor:
            # Check if phone already exists
            cursor.execute('SELECT guardian_id FROM guardians WHERE phone = %s', (final_phone,))
            existing = cursor.fetchone()
            if existing:
                print(f"❌ [REGISTRATION] Phone already registered: {final_phone}")
                return jsonify({
                    'success': False,
                    'error': 'Phone number already registered'
                }), 409
            
            # 🔑 Hash password (password is already cleaned)
            password_hash = hash_password(password)
            print(f"✅ [REGISTRATION] Password hash generated: {password_hash[:30]}...")
            print(f"   Hash length: {len(password_hash)}")
            print(f"   Is bcrypt format: {password_hash.startswith('$2')}")
            
            # Insert into database
            cursor.execute('''
                INSERT INTO guardians (
                    full_name, 
                    phone, 
                    email, 
                    password_hash, 
                    address, 
                    registration_date,
                    last_login,
                    is_active,
                    auth_provider
                )
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE, 'phone')
            ''', (
                full_name,
                final_phone,
                data.get('email', ''),
                password_hash,
                data.get('address', '')
            ))
            
            cursor.execute('SELECT guardian_id FROM guardians WHERE phone = %s', (final_phone,))
            result = cursor.fetchone()
            guardian_id = result[0] if result else None
            
            print(f"✅ [REGISTRATION] Database record created with guardian_id: {guardian_id}")
        
        response_data = {
            'success': True,
            'guardian_id': guardian_id,
            'full_name': full_name,
            'phone': final_phone,
            'email': data.get('email', ''),
            'message': 'Registration successful! You can now login.',
            'redirect_url': f'https://guardian-drive-app.web.app/login.html?registered=true&prefilled_phone={final_phone}'
        }
        
        print(f"✅ [REGISTRATION] Registration complete for guardian_id: {guardian_id}")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"❌ [REGISTRATION] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Registration failed. Please try again.'
        }), 500

# ==================== GUARDIAN DASHBOARD ====================
@app.route('/api/guardian/dashboard', methods=['GET'])
def guardian_dashboard():
    """Get guardian dashboard data"""
    try:
        guardian_id = request.args.get('guardian_id')
        token = request.args.get('token')
        
        if not guardian_id or not token:
            return jsonify({
                'success': False,
                'error': 'Authentication required'
            }), 401
        
        # Validate session
        if not validate_session(guardian_id, token):
            return jsonify({
                'success': False,
                'error': 'Session expired or invalid'
            }), 401
        
        guardian = get_guardian_by_id(guardian_id)
        if not guardian:
            return jsonify({
                'success': False,
                'error': 'Guardian not found'
            }), 404
        
        drivers = get_guardian_drivers(guardian_id)
        recent_alerts = get_recent_alerts(guardian_id, 5)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as count FROM drivers WHERE guardian_id = %s', (guardian_id,))
            driver_count_result = cursor.fetchone()
            driver_count = driver_count_result['count'] if isinstance(driver_count_result, dict) else driver_count_result[0]
            
            cursor.execute('SELECT COUNT(*) as count FROM alerts WHERE guardian_id = %s', (guardian_id,))
            total_alerts_result = cursor.fetchone()
            total_alerts = total_alerts_result['count'] if isinstance(total_alerts_result, dict) else total_alerts_result[0]
            
            cursor.execute('SELECT COUNT(*) as count FROM alerts WHERE guardian_id = %s AND acknowledged = FALSE', 
                         (guardian_id,))
            unread_alerts_result = cursor.fetchone()
            unread_alerts = unread_alerts_result['count'] if isinstance(unread_alerts_result, dict) else unread_alerts_result[0]
        
        return jsonify({
            'success': True,
            'guardian': guardian,
            'session_valid': True,
            'dashboard': {
                'driver_count': driver_count,
                'total_alerts': total_alerts,
                'unread_alerts': unread_alerts,
                'recent_alerts': recent_alerts
            },
            'drivers': drivers
        })
        
    except Exception as e:
        print(f"❌ Error in guardian_dashboard: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== GET GUARDIAN DETAILS ENDPOINT ====================
@app.route('/api/guardian/<int:guardian_id>', methods=['GET'])
def get_guardian_details(guardian_id):
    try:
        token = request.args.get('token')
        
        if not token:
            return jsonify({
                'success': False,
                'error': 'Authentication token required'
            }), 401
        
        # Validate session
        if not validate_session(guardian_id, token):
            return jsonify({
                'success': False,
                'error': 'Invalid or expired session'
            }), 401
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT guardian_id, full_name, phone, email, address, 
                       registration_date, last_login, auth_provider, is_active
                FROM guardians 
                WHERE guardian_id = %s
            ''', (guardian_id,))
            
            result = cursor.fetchone()
            
            if not result:
                return jsonify({
                    'success': False,
                    'error': 'Guardian not found'
                }), 404
            
            # Convert to dictionary
            if isinstance(result, dict):
                guardian = result
            else:
                # Handle tuple result
                guardian = {
                    'guardian_id': result[0],
                    'full_name': result[1],
                    'phone': result[2],
                    'email': result[3],
                    'address': result[4],
                    'registration_date': result[5].isoformat() if result[5] else None,
                    'last_login': result[6].isoformat() if result[6] else None,
                    'auth_provider': result[7],
                    'is_active': result[8]
                }
            
            return jsonify({
                'success': True,
                'guardian': guardian
            })
            
    except Exception as e:
        print(f"❌ Error in get_guardian_details: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== DRIVER REGISTRATION WITH CLOUDINARY ====================
@app.route('/api/register-driver', methods=['POST'])
def register_driver():
    """Register a new driver with face images uploaded to Cloudinary - FIXED VERSION with embeddings"""
    global CLOUDINARY_ENABLED 

    try:
        data = request.json
        print(f"   Received data keys: {list(data.keys())}")
        driver_name = data.get('driver_name')
        driver_phone = data.get('driver_phone')
        guardian_id = data.get('guardian_id')
        token = data.get('token')
        face_images = data.get('face_images', [])  
        
        print(f"\n🔍 [DRIVER REGISTRATION] Extracted data:")
        print(f"   Driver Name: {driver_name}")
        print(f"   Driver Phone: {driver_phone}")
        print(f"   Guardian ID: {guardian_id}")
        print(f"   Token present: {bool(token)}")
        print(f"   Token length: {len(token) if token else 0}")
        print(f"   Face images count: {len(face_images)}")
        
        # Validate required fields
        if not all([driver_name, driver_phone, guardian_id, token]):
            print("❌ [DRIVER REGISTRATION] Missing required fields:")
            print(f"   driver_name: {bool(driver_name)}")
            print(f"   driver_phone: {bool(driver_phone)}")
            print(f"   guardian_id: {bool(guardian_id)}")
            print(f"   token: {bool(token)}")
            return jsonify({
                'success': False,
                'error': 'Missing required fields: driver_name, driver_phone, guardian_id, or token'
            }), 400
        
        print(f"\n🔐 [DRIVER REGISTRATION] Validating session for guardian_id: {guardian_id}")
        
        # Validate guardian session
        if not validate_session(guardian_id, token):
            print(f"❌ [DRIVER REGISTRATION] Session validation failed for guardian_id: {guardian_id}")
            return jsonify({
                'success': False,
                'error': 'Invalid or expired guardian session. Please login again.'
            }), 401
        
        print("✅ [DRIVER REGISTRATION] Session validated successfully")
        
        # Validate face images - need at least 3 for different angles
        if len(face_images) < 3:
            print(f"❌ [DRIVER REGISTRATION] Not enough face images: {len(face_images)} (need at least 3)")
            return jsonify({
                'success': False,
                'error': f'At least 3 face images from different angles are required. Received {len(face_images)} images.'
            }), 400
        
        # Check if Cloudinary is configured
        cloudinary_enabled = CLOUDINARY_ENABLED

        if not cloudinary_enabled:
            print("⚠️ [DRIVER REGISTRATION] Cloudinary not enabled, using local storage")
            # Continue with local storage
        
        # Generate unique driver ID
        import time
        timestamp = int(time.time() * 1000)
        driver_id = f"DRV{str(timestamp)[-8:].upper()}"
        
        # Generate reference number if not provided
        reference_number = data.get('reference_number', f"REF{str(timestamp)[-8:].upper()}")
        
        # Get additional driver info
        driver_email = data.get('driver_email', '')
        driver_address = data.get('driver_address', '')
        license_number = data.get('license_number', '')
        capture_angles = data.get('capture_angles', ['front', 'left', 'right'])
        
        print(f"\n📝 [DRIVER REGISTRATION] Processing registration for driver: {driver_name}")
        print(f"   Driver ID: {driver_id}")
        print(f"   Reference Number: {reference_number}")
        
        with get_db_cursor() as cursor:
            # Check if phone already exists
            cursor.execute('SELECT driver_id FROM drivers WHERE phone = %s', (driver_phone,))
            existing = cursor.fetchone()
            if existing:
                print(f"❌ [DRIVER REGISTRATION] Phone already registered: {driver_phone}")
                return jsonify({
                    'success': False,
                    'error': f'Phone number {driver_phone} is already registered for another driver.'
                }), 409
            
            # Check if reference number already exists
            cursor.execute('SELECT driver_id FROM drivers WHERE reference_number = %s', 
                         (reference_number,))
            existing_ref = cursor.fetchone()
            if existing_ref:
                # Generate new reference number
                new_ref_number = f"REF{str(uuid.uuid4().int)[:8].upper()}"
                print(f"🔧 [DRIVER REGISTRATION] Reference number already exists. Generated new: {new_ref_number}")
                reference_number = new_ref_number
            
            print(f"\n🗄️ [DRIVER REGISTRATION] Inserting driver into database...")
            
            try:
                # Insert driver into database
                cursor.execute('''
                    INSERT INTO drivers (
                        driver_id, name, phone, email, address, 
                        reference_number, license_number, guardian_id, 
                        registration_date, is_active
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    driver_id, 
                    driver_name, 
                    driver_phone, 
                    driver_email, 
                    driver_address,
                    reference_number,
                    license_number,
                    guardian_id,
                    datetime.now(),
                    True
                ))
                
                print(f"✅ [DRIVER REGISTRATION] Driver registered: {driver_name} (ID: {driver_id})")
                
            except Exception as db_error:
                print(f"❌ [DRIVER REGISTRATION] Database insertion failed: {db_error}")
                return jsonify({
                    'success': False,
                    'error': f'Database error: {str(db_error)}'
                }), 500
            
            # ==================== IMAGE UPLOAD SECTION ====================
            saved_images = []
            upload_errors = []
            
            if cloudinary_enabled:
                print(f"\n☁️ [DRIVER REGISTRATION] Cloudinary enabled. Starting upload for {len(face_images)} images...")
                
                for i, face_image_data in enumerate(face_images[:3], 1):
                    if i <= len(capture_angles):
                        capture_angle = capture_angles[i-1]
                    else:
                        capture_angle = f"angle_{i}"
                    
                    print(f"\n📤 [DRIVER REGISTRATION] Uploading image {i}/{len(face_images)} to Cloudinary (angle: {capture_angle})...")
                    
                    # Clean base64 data
                    image_base64 = face_image_data
                    if ',' in image_base64:
                        image_base64 = image_base64.split(',')[1]
                    
                    if not image_base64 or len(image_base64) < 100:
                        print(f"⚠️ [DRIVER REGISTRATION] Skipping empty/invalid image")
                        upload_errors.append(f"Image {i} is empty or invalid")
                        continue
                    
                    try:
                        # Generate unique public ID for Cloudinary
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        public_id = f"driver_faces/{driver_id}/{capture_angle}_{timestamp_str}"
                        
                        print(f"📤 Uploading to Cloudinary...")
                        
                        # Upload to Cloudinary with timeout
                        upload_start = time.time()
                        
                        upload_result = cloudinary.uploader.upload(
                            f"data:image/jpeg;base64,{image_base64}",
                            public_id=public_id,
                            folder=f"driver_faces/{driver_id}",
                            resource_type="image",
                            overwrite=False,
                            faces=True,
                            quality="auto:good",
                            format="jpg",
                            timeout=30
                        )
                        
                        upload_time = time.time() - upload_start
                        
                        cloudinary_url = upload_result.get('secure_url')
                        public_id = upload_result.get('public_id')
                        
                        print(f"✅ Image {i} uploaded to Cloudinary successfully!")
                        print(f"   URL: {cloudinary_url[:60]}...")
                        print(f"   Upload time: {upload_time:.2f} seconds")
                        
                        # Store Cloudinary URL in database
                        cursor.execute('''
                            INSERT INTO face_images (driver_id, image_path, capture_date)
                            VALUES (%s, %s, %s)
                        ''', (driver_id, cloudinary_url, datetime.now()))
                        
                        saved_images.append({
                            'angle': capture_angle,
                            'public_id': public_id,
                            'url': cloudinary_url,
                            'upload_time': upload_time,
                            'upload_method': 'cloudinary'
                        })
                        
                        # Save the first image for embedding computation
                        if i == 1:
                            first_image_base64 = image_base64
                        
                        if i < len(face_images):
                            time.sleep(1)
                        
                    except cloudinary.exceptions.Error as cloudinary_error:  
                        error_msg = f"Cloudinary API error for image {i} ({capture_angle}): {str(cloudinary_error)}"
                        print(f"Cloudinary is not available: {error_msg}")
                        upload_errors.append(error_msg)
                        
                    except Exception as upload_error:
                        error_msg = f"Error uploading image {i} ({capture_angle}): {str(upload_error)}"
                        print(f" DRIVER REGISTRATION not successful: {error_msg}")
                        print(f"   Error type: {type(upload_error).__name__}")
                        upload_errors.append(error_msg)
                        continue
                
                print(f"\n📊 [DRIVER REGISTRATION] Cloudinary upload summary:")
                print(f"   Successfully uploaded: {len(saved_images)}")
                print(f"   Upload errors: {len(upload_errors)}")
                
                if len(saved_images) == 0 and len(upload_errors) > 0:
                    print(f"⚠️ [DRIVER REGISTRATION] All Cloudinary uploads failed, falling back to local storage")
                    cloudinary_enabled = False  # Force fallback
                    
            # ==================== LOCAL STORAGE FALLBACK ====================
            if not cloudinary_enabled or len(saved_images) == 0:
                print(f"\n💾 [DRIVER REGISTRATION] Using LOCAL storage (Cloudinary disabled or failed)...")
                
                # Clear any previous saved images if Cloudinary failed
                if saved_images:
                    saved_images = []
                    upload_errors = []
                
                for i, face_image_data in enumerate(face_images[:3], 1):
                    if i <= len(capture_angles):
                        capture_angle = capture_angles[i-1]
                    else:
                        capture_angle = f"angle_{i}"
                    
                    print(f"\n💾 [DRIVER REGISTRATION] Saving image {i}/{len(face_images)} locally (angle: {capture_angle})...")
                    
                    # Clean base64 data
                    image_base64 = face_image_data
                    if ',' in image_base64:
                        image_base64 = image_base64.split(',')[1]
                    
                    if not image_base64 or len(image_base64) < 100:
                        print(f"⚠️ [DRIVER REGISTRATION] Skipping empty/invalid image")
                        upload_errors.append(f"Image {i} is empty or invalid")
                        continue
                    
                    try:
                        # Generate filename
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"{driver_id}_{capture_angle}_{timestamp_str}.jpg"
                        
                        # Create directory if it doesn't exist
                        import os
                        uploads_dir = os.path.join(BASE_DIR, 'uploads', 'driver_faces')
                        os.makedirs(uploads_dir, exist_ok=True)
                        
                        # Save locally
                        filepath = os.path.join(uploads_dir, filename)
                        import base64
                        with open(filepath, 'wb') as f:
                            f.write(base64.b64decode(image_base64))
                        
                        # Create a URL path for the saved image
                        image_url = f"/uploads/driver_faces/{filename}"
                        
                        print(f"✅ [DRIVER REGISTRATION] Image {i} saved locally: {filename}")
                        print(f"   File size: {os.path.getsize(filepath) / 1024:.1f} KB")
                        
                        # Store in database
                        cursor.execute('''
                            INSERT INTO face_images (driver_id, image_path, capture_date)
                            VALUES (%s, %s, %s)
                        ''', (driver_id, image_url, datetime.now()))
                        
                        saved_images.append({
                            'angle': capture_angle,
                            'url': image_url,
                            'filepath': filepath,
                            'upload_method': 'local'
                        })
                        
                        # Save the first image for embedding computation
                        if i == 1:
                            first_image_base64 = image_base64
                        
                    except Exception as save_error:
                        error_msg = f"Error saving image {i} locally: {str(save_error)}"
                        print(f"❌ [DRIVER REGISTRATION] {error_msg}")
                        upload_errors.append(error_msg)
                
                print(f"\n📊 [DRIVER REGISTRATION] Local storage summary:")
                print(f"   Successfully saved: {len(saved_images)}")
                print(f"   Save errors: {len(upload_errors)}")
            
            # ==================== FACE EMBEDDING COMPUTATION ====================
            embedding_stored = False
            if saved_images and len(saved_images) > 0:
                try:
                    print(f"\n🧠 [DRIVER REGISTRATION] Computing face embedding from first image...")
                    
                    # Use the first image base64 we saved
                    if 'first_image_base64' in locals():
                        embedding = get_face_embedding_from_base64(first_image_base64)
                        
                        if embedding:
                            # Store embedding in database
                            cursor.execute('''
                                UPDATE drivers SET face_embedding = %s WHERE driver_id = %s
                            ''', (json.dumps(embedding), driver_id))
                            embedding_stored = True
                            print(f"✅ [DRIVER REGISTRATION] Face embedding stored successfully!")
                            print(f"   Embedding vector length: {len(embedding)}")
                        else:
                            print(f"⚠️ [DRIVER REGISTRATION] Could not compute face embedding - no face detected in image")
                    else:
                        print(f"⚠️ [DRIVER REGISTRATION] No image data available for embedding computation")
                except Exception as embed_error:
                    print(f"❌ [DRIVER REGISTRATION] Error computing face embedding: {embed_error}")
                    import traceback
                    traceback.print_exc()
            
            # ==================== REGISTRATION COMPLETION ====================
            if len(saved_images) == 0:
                print(f"❌ [DRIVER REGISTRATION] NO face images were saved for driver {driver_id}")
                print(f"   Errors: {upload_errors}")
                
                # Rollback driver registration if no images were saved
                cursor.execute('DELETE FROM drivers WHERE driver_id = %s', (driver_id,))
                print(f"   Rolled back driver registration")
                
                return jsonify({
                    'success': False,
                    'error': 'Failed to save any face images. Registration cancelled.',
                    'upload_errors': upload_errors if upload_errors else ['Unknown error occurred']
                }), 500
            
            if len(upload_errors) > 0:
                print(f"⚠️ [DRIVER REGISTRATION] Some images failed:")
                for error in upload_errors:
                    print(f"   - {error}")
            
            # Log activity
            cursor.execute('''
                INSERT INTO activity_log (guardian_id, action, details)
                VALUES (%s, %s, %s)
            ''', (guardian_id, 'DRIVER_REGISTERED', 
                f'Registered driver: {driver_name} (ID: {driver_id}) with {len(saved_images)} face images. Embedding stored: {embedding_stored}'))
            
            # Get guardian info
            cursor.execute('SELECT full_name FROM guardians WHERE guardian_id = %s', 
                         (guardian_id,))
            guardian_info = cursor.fetchone()
            
            if isinstance(guardian_info, dict):
                guardian_name = guardian_info['full_name']
            else:
                guardian_name = guardian_info[0] if guardian_info else 'Unknown Guardian'
            
            print(f"\n✅ [DRIVER REGISTRATION] Registration COMPLETE!")
            print(f"   Driver: {driver_name} (ID: {driver_id})")
            print(f"   Images saved: {len(saved_images)}")
            print(f"   Embedding stored: {embedding_stored}")
            print(f"   Storage method: {'Cloudinary' if cloudinary_enabled and saved_images and saved_images[0].get('upload_method') == 'cloudinary' else 'Local'}")
            
            # Prepare success response
            response_data = {
                'success': True,
                'driver_id': driver_id,
                'driver_name': driver_name,
                'driver_phone': driver_phone,
                'reference_number': reference_number,
                'face_images_saved': len(saved_images),
                'image_urls': [img['url'] for img in saved_images],
                'registration_date': datetime.now().isoformat(),
                'guardian_name': guardian_name,
                'storage_method': 'cloudinary' if cloudinary_enabled and saved_images and saved_images[0].get('upload_method') == 'cloudinary' else 'local',
                'embedding_stored': embedding_stored,
                'message': f'Driver registered successfully with {len(saved_images)} face images' + (' and face embedding' if embedding_stored else ' (warning: no face embedding)')
            }
            
            # Send real-time notification to guardian
            try:
                notification_data = {
                    'type': 'driver_registered',
                    'driver_id': driver_id,
                    'driver_name': driver_name,
                    'guardian_id': guardian_id,
                    'face_images_count': len(saved_images),
                    'embedding_stored': embedding_stored,
                    'timestamp': datetime.now().isoformat(),
                    'message': f'New driver registered: {driver_name}'
                }
                
                print(f"🔔 [DRIVER REGISTRATION] Sending notification...")
                
                # Emit socket event
                for client_id, client_info in connected_clients.items():
                    if client_info.get('guardian_id') == guardian_id and client_info.get('authenticated'):
                        socketio.emit('guardian_notification', notification_data, room=client_id)
                
            except Exception as notify_error:
                print(f"⚠️ [DRIVER REGISTRATION] Error sending notification: {notify_error}")
            
            print(f"\n✅ [DRIVER REGISTRATION] ===== REGISTRATION SUCCESSFUL =====")
            return jsonify(response_data)
        
    except Exception as e:
        print(f"\n❌❌❌ [DRIVER REGISTRATION] UNEXPECTED ERROR: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ [DRIVER REGISTRATION] ===== REGISTRATION FAILED =====")
        
        return jsonify({
            'success': False,
            'error': f'Driver registration failed: {str(e)}'
        }), 500

@app.route('/api/test-driver-registration', methods=['POST'])
def test_driver_registration():
    """Test endpoint for driver registration debugging"""
    print("\n🧪🧪🧪 TEST DRIVER REGISTRATION ENDPOINT CALLED 🧪🧪🧪")
    
    data = request.json
    headers = dict(request.headers)
    
    print("📦 Request Data:")
    print(f"   Keys: {list(data.keys()) if data else 'No data'}")
    print(f"   Headers: {headers}")
    
    if data:
        for key, value in data.items():
            if key == 'face_images' and isinstance(value, list):
                print(f"   {key}: List with {len(value)} items")
                if len(value) > 0:
                    print(f"      First item length: {len(value[0])}")
                    print(f"      First item preview: {value[0][:50]}...")
            elif key == 'token' and value:
                print(f"   {key}: Present ({len(value)} chars) - {value[:20]}...")
            else:
                print(f"   {key}: {value}")
    
    # Validate guardian session if provided
    guardian_id = data.get('guardian_id') if data else None
    token = data.get('token') if data else None
    
    session_valid = False
    if guardian_id and token:
        print(f"\n🔐 Testing session validation...")
        print(f"   Guardian ID: {guardian_id}")
        print(f"   Token: {token[:20]}...")
        
        session_valid = validate_session(guardian_id, token)
        print(f"   Session valid: {session_valid}")
    
    # Check Cloudinary
    print(f"\n☁️ Cloudinary status:")
    print(f"   Enabled: {CLOUDINARY_ENABLED}")
    print(f"   Cloud name: {CLOUDINARY_CLOUD_NAME}")
    
    return jsonify({
        'success': True,
        'message': 'Test endpoint works',
        'test_data': {
            'received_keys': list(data.keys()) if data else [],
            'session_valid': session_valid,
            'cloudinary_enabled': CLOUDINARY_ENABLED,
            'cloudinary_configured': bool(CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET),
            'headers_received': list(headers.keys()),
            'request_method': request.method,
            'timestamp': datetime.now().isoformat()
        }
    })

@app.route('/api/driver/<driver_id>/face-images', methods=['GET'])
def get_driver_face_images(driver_id):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT image_id, image_path, capture_date
                FROM face_images
                WHERE driver_id = %s
                ORDER BY capture_date DESC
            ''', (driver_id,))
            
            images = cursor.fetchall()
            
            # Return Cloudinary URLs directly
            images_data = []
            for img in images:
                if isinstance(img, dict):
                    image_url = img['image_path']
                    capture_date = img['capture_date']
                    image_id = img['image_id']
                else:
                    image_id, image_url, capture_date = img
                
                images_data.append({
                    'image_id': image_id,
                    'url': image_url,
                    'capture_date': capture_date.isoformat() if hasattr(capture_date, 'isoformat') else str(capture_date),
                    'is_cloudinary': 'cloudinary' in image_url.lower()
                })
            
            return jsonify({
                'success': True,
                'driver_id': driver_id,
                'image_count': len(images_data),
                'images': images_data,
                'cloudinary_enabled': CLOUDINARY_ENABLED
            })
            
    except Exception as e:
        print(f"❌ Error in get_driver_face_images: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/guardian/<guardian_id>/drivers', methods=['GET'])
def get_guardian_drivers_endpoint(guardian_id):
    """Get all drivers for a specific guardian"""
    try:
        token = request.args.get('token')
        
        if not token:
            return jsonify({
                'success': False,
                'error': 'Authentication token required'
            }), 401
        
        # Validate session
        if not validate_session(guardian_id, token):
            return jsonify({
                'success': False,
                'error': 'Invalid or expired session'
            }), 401
        
        drivers = get_guardian_drivers(guardian_id)
        
        # Add face image count for each driver
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for driver in drivers:
                cursor.execute('SELECT COUNT(*) FROM face_images WHERE driver_id = %s', 
                             (driver['driver_id'],))
                result = cursor.fetchone()
                if isinstance(result, dict):
                    driver['face_image_count'] = result['count']
                else:
                    driver['face_image_count'] = result[0]
        
        return jsonify({
            'success': True,
            'session_valid': True,
            'guardian_id': guardian_id,
            'count': len(drivers),
            'drivers': drivers
        })
        
    except Exception as e:
        print(f"❌ Error in get_guardian_drivers_endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/driver/<driver_id>', methods=['GET'])
def get_driver_details(driver_id):
    """Get detailed information about a specific driver"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get driver details
            cursor.execute('''
                SELECT d.*, g.full_name as guardian_name, g.phone as guardian_phone, g.email as guardian_email
                FROM drivers d
                JOIN guardians g ON d.guardian_id = g.guardian_id
                WHERE d.driver_id = %s
            ''', (driver_id,))
            
            driver_result = cursor.fetchone()
            
            if not driver_result:
                return jsonify({
                    'success': False,
                    'error': 'Driver not found'
                }), 404
            
            driver = dict(driver_result)
            
            # Get alert count
            cursor.execute('SELECT COUNT(*) FROM alerts WHERE driver_id = %s', (driver_id,))
            alert_count_result = cursor.fetchone()
            driver['alert_count'] = alert_count_result['count'] if isinstance(alert_count_result, dict) else alert_count_result[0]
            
            # Get unacknowledged alert count
            cursor.execute('SELECT COUNT(*) FROM alerts WHERE driver_id = %s AND acknowledged = FALSE', 
                         (driver_id,))
            unread_alerts_result = cursor.fetchone()
            driver['unread_alerts'] = unread_alerts_result['count'] if isinstance(unread_alerts_result, dict) else unread_alerts_result[0]
            
            # Get face image count
            cursor.execute('SELECT COUNT(*) FROM face_images WHERE driver_id = %s', (driver_id,))
            face_count_result = cursor.fetchone()
            driver['face_image_count'] = face_count_result['count'] if isinstance(face_count_result, dict) else face_count_result[0]
            
            return jsonify({
                'success': True,
                'driver': driver
            })
            
    except Exception as e:
        print(f"❌ Error in get_driver_details: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== Driver Monitoring (WEBRTc) ==================

@app.route('/api/driver/<driver_id>/guardian', methods=['GET'])
def get_driver_guardian(driver_id):
    """Get guardian ID for a driver (for WebRTC connection)"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT guardian_id, name 
                FROM drivers 
                WHERE driver_id = %s AND is_active = TRUE
            ''', (driver_id,))
            
            result = cursor.fetchone()
            
            if not result:
                return jsonify({
                    'success': False,
                    'error': 'Driver not found'
                }), 404
            
            if isinstance(result, dict):
                guardian_id = result['guardian_id']
                driver_name = result['name']
            else:
                guardian_id = result[0]
                driver_name = result[1]
            
            return jsonify({
                'success': True,
                'guardian_id': guardian_id,
                'driver_name': driver_name,
                'driver_id': driver_id
            })
            
    except Exception as e:
        print(f"❌ Error getting driver guardian: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/driver/<driver_id>/verify', methods=['POST'])
def verify_driver_access(driver_id):
    """Verify driver has access to stream"""
    try:
        data = request.json
        provided_guardian_id = data.get('guardian_id')
        
        if not provided_guardian_id:
            return jsonify({
                'success': False,
                'error': 'Guardian ID required'
            }), 400
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT guardian_id, name 
                FROM drivers 
                WHERE driver_id = %s AND guardian_id = %s AND is_active = TRUE
            ''', (driver_id, provided_guardian_id))
            
            result = cursor.fetchone()
            
            if result:
                return jsonify({
                    'success': True,
                    'verified': True,
                    'message': 'Driver verified'
                })
            else:
                return jsonify({
                    'success': False,
                    'verified': False,
                    'error': 'Driver not associated with this guardian'
                }), 403
                
    except Exception as e:
        print(f"❌ Error verifying driver: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== UPDATE GUARDIAN ENDPOINT ====================

@app.route('/api/driver/update', methods=['POST'])
def update_driver():
    """Update driver information"""
    try:
        data = request.json

        guardian_id = data.get('guardian_id')
        token = data.get('token')
        driver_id = data.get('driver_id')

        if not all([guardian_id, token, driver_id]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400

        # Validate session
        if not validate_session(guardian_id, token):
            return jsonify({
                'success': False,
                'error': 'Invalid or expired session'
            }), 401

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Check if driver belongs to this guardian
            cursor.execute('''
                SELECT driver_id FROM drivers 
                WHERE driver_id = %s AND guardian_id = %s
            ''', (driver_id, guardian_id))

            if not cursor.fetchone():
                return jsonify({
                    'success': False,
                    'error': 'Driver not found or not authorized'
                }), 404

            # Build dynamic update fields
            update_fields = []
            update_values = []

            # --- PHONE UPDATE ---
            new_phone = data.get('phone')
            if new_phone:
                phone_clean = re.sub(r'[\s\-\(\)\+]', '', new_phone)

                cursor.execute('''
                    SELECT driver_id FROM drivers 
                    WHERE phone = %s AND driver_id != %s
                ''', (phone_clean, driver_id))

                if cursor.fetchone():
                    return jsonify({
                        'success': False,
                        'error': f'Phone number {phone_clean} is already registered'
                    }), 409

                update_fields.append('phone = %s')
                update_values.append(phone_clean)

            # --- OTHER FIELDS ---
            fields_to_update = ['name', 'email', 'address', 'license_number']

            for field in fields_to_update:
                if field in data:
                    update_fields.append(f'{field} = %s')
                    update_values.append(data[field])

            if not update_fields:
                return jsonify({
                    'success': False,
                    'error': 'No fields to update'
                }), 400

            # Build final query
            query = f'''
                UPDATE drivers 
                SET {', '.join(update_fields)}, updated_at = %s
                WHERE driver_id = %s
            '''

            # ✅ CORRECT PARAMETER ORDER
            cursor.execute(
                query,
                (*update_values, datetime.now(), driver_id)
            )

            conn.commit()

            # Get updated driver
            cursor.execute('''
                SELECT * FROM drivers WHERE driver_id = %s
            ''', (driver_id,))

            driver = dict(cursor.fetchone())

            # Log activity
            cursor.execute('''
                INSERT INTO activity_log (guardian_id, action, details)
                VALUES (%s, %s, %s)
            ''', (
                guardian_id,
                'DRIVER_UPDATED',
                f'Updated driver: {driver.get("name")} (ID: {driver_id})'
            ))

            conn.commit()

            return jsonify({
                'success': True,
                'driver': driver,
                'message': 'Driver updated successfully'
            })

    except Exception as e:
        print(f"❌ Error in update_driver: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/guardian/update', methods=['POST'])
def update_guardian():
    """Update guardian information"""
    try:
        data = request.json
        
        guardian_id = data.get('guardian_id')
        token = data.get('token')
        
        if not all([guardian_id, token]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400
        
        # Validate session
        if not validate_session(guardian_id, token):
            return jsonify({
                'success': False,
                'error': 'Invalid or expired session'
            }), 401
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build update query dynamically
            update_fields = []
            update_values = []
            
            # Check phone number uniqueness
            new_phone = data.get('phone')
            if new_phone:
                # Clean phone number
                phone_clean = re.sub(r'[\s\-\(\)\+]', '', new_phone)
                
                # Convert to 09 format if needed
                if len(phone_clean) == 12 and phone_clean.startswith('639'):
                    phone_clean = '09' + phone_clean[3:]
                elif len(phone_clean) == 11 and phone_clean.startswith('63'):
                    phone_clean = '09' + phone_clean[2:]
                elif len(phone_clean) == 10 and phone_clean.startswith('9'):
                    phone_clean = '0' + phone_clean
                elif len(phone_clean) == 10:
                    phone_clean = '09' + phone_clean
                
                # Check if phone already exists for another guardian
                cursor.execute('''
                    SELECT guardian_id FROM guardians 
                    WHERE phone = %s AND guardian_id != %s
                ''', (phone_clean, guardian_id))
                
                if cursor.fetchone():
                    return jsonify({
                        'success': False,
                        'error': f'Phone number {phone_clean} is already registered for another guardian'
                    }), 409
                
                update_fields.append('phone = %s')
                update_values.append(phone_clean)
            
            # Other fields that can be updated
            fields_to_update = ['full_name', 'email', 'address']
            for field in fields_to_update:
                if field in data:
                    update_fields.append(f'{field} = %s')
                    update_values.append(data[field])
            
            if not update_fields:
                return jsonify({
                    'success': False,
                    'error': 'No fields to update'
                }), 400
            
            # Add guardian_id to values
            update_values.append(guardian_id)
            
            # Execute update
            query = f'''
                UPDATE guardians 
                SET {', '.join(update_fields)}
                WHERE guardian_id = %s
            '''
            
            cursor.execute(query, (*update_values,))
            conn.commit()
            
            # Get updated guardian info
            cursor.execute('''
                SELECT * FROM guardians WHERE guardian_id = %s
            ''', (guardian_id,))
            
            guardian = dict(cursor.fetchone())
            
            # Update localStorage data if needed
            if 'full_name' in data:
                # This would be handled by frontend
            
                return jsonify({
                'success': True,
                'guardian': guardian,
                'message': 'Profile updated successfully'
            })
        
    except Exception as e:
        print(f"❌ Error in update_guardian: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/driver/identify', methods=['POST'])
def identify_driver():
    """Identify driver from a live face image using DeepFace"""
    try:
        data = request.json
        image_base64 = data.get('image')
        if not image_base64:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # Compute embedding from incoming image
        embedding = get_face_embedding_from_base64(image_base64)
        if embedding is None:
            return jsonify({'success': False, 'error': 'No face detected in image'}), 400
        
        # Fetch all drivers with stored embeddings
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT driver_id, name, face_embedding FROM drivers 
                WHERE face_embedding IS NOT NULL
            ''')
            drivers = cursor.fetchall()
        
        if not drivers:
            return jsonify({'success': False, 'error': 'No enrolled drivers found'}), 404
        
        # Find best match by cosine distance (DeepFace uses cosine)
        best_match = None
        best_distance = float('inf')
        threshold = 0.4  # Facenet threshold
        
        for driver in drivers:
            if isinstance(driver, dict):
                stored_emb = json.loads(driver['face_embedding'])
                driver_id = driver['driver_id']
                driver_name = driver['name']
            else:
                stored_emb = json.loads(driver[2])
                driver_id = driver[0]
                driver_name = driver[1]
            
            # Cosine distance
            dot_product = np.dot(embedding, stored_emb)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(stored_emb)
            cosine_similarity = dot_product / (norm_a * norm_b)
            distance = 1 - cosine_similarity  # Convert to distance
            
            if distance < best_distance:
                best_distance = distance
                best_match = (driver_id, driver_name)
        
        if best_distance < threshold:
            return jsonify({
                'success': True,
                'driver_id': best_match[0],
                'driver_name': best_match[1],
                'confidence': 1 - best_distance,
                'distance': best_distance
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No matching driver found',
                'distance': best_distance
            }), 404
            
    except Exception as e:
        print(f"❌ Error in identify_driver: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
# ==================== GET DRIVER DETAILS ENDPOINT ====================

@app.route('/api/driver/<driver_id>/details', methods=['GET'])
def get_driver_details_full(driver_id):
    """Get detailed information about a specific driver with validation"""
    try:
        guardian_id = request.args.get('guardian_id')
        token = request.args.get('token')
        
        if not guardian_id or not token:
            return jsonify({
                'success': False,
                'error': 'Authentication required'
            }), 401
        
        # Validate session
        if not validate_session(guardian_id, token):
            return jsonify({
                'success': False,
                'error': 'Invalid or expired session'
            }), 401
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get driver details with validation
            cursor.execute('''
                SELECT d.*, 
                       (SELECT COUNT(*) FROM alerts a WHERE a.driver_id = d.driver_id AND a.acknowledged = FALSE) as unread_alerts,
                       (SELECT COUNT(*) FROM face_images f WHERE f.driver_id = d.driver_id) as face_image_count
                FROM drivers d
                WHERE d.driver_id = %s AND d.guardian_id = %s
            ''', (driver_id, guardian_id))
            
            driver_result = cursor.fetchone()
            
            if not driver_result:
                return jsonify({
                    'success': False,
                    'error': 'Driver not found or not authorized'
                }), 404
            
            driver = dict(driver_result)
            
            # Get face images
            cursor.execute('''
                SELECT image_id, image_path, capture_date
                FROM face_images
                WHERE driver_id = %s
                ORDER BY capture_date DESC
            ''', (driver_id,))
            
            face_images = cursor.fetchall()
            driver['face_images'] = [dict(img) for img in face_images]
            
            return jsonify({
                'success': True,
                'driver': driver
            })
            
    except Exception as e:
        print(f"❌ Error in get_driver_details_full: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== ALERT ENDPOINTS ====================
@app.route('/api/send-alert', methods=['POST'])
def send_alert():
    """Send drowsiness alert"""
    try:
        data = request.json
        driver_id = data.get('driver_id')
        driver_name = data.get('driver_name', 'Unknown Driver')
        severity = data.get('severity', 'high')
        message = data.get('message', 'Drowsiness detected!')
        confidence = data.get('confidence', 0.0)
        detection_details = data.get('detection_details', {})
        
        if not driver_id:
            return jsonify({
                'success': False,
                'error': 'Driver ID required'
            }), 400
        
        # Try to find the driver in the database
        driver_info = get_driver_by_name_or_id(driver_id)
        
        if driver_info:
            driver_id = driver_info['driver_id']
            driver_name = driver_info['name']
            guardian_id = driver_info['guardian_id']
            guardian_name = driver_info['guardian_name']
            guardian_phone = driver_info['guardian_phone']
        else:
            # Driver not found - create a temporary record
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT guardian_id, full_name, phone FROM guardians LIMIT 1')
                guardian_result = cursor.fetchone()
                
                if guardian_result:
                    guardian_id = guardian_result['guardian_id']
                    guardian_name = guardian_result['full_name']
                    guardian_phone = guardian_result['phone']
                    
                    # Create a temporary driver entry
                    temp_driver_id = f"TEMP{int(time.time())}"
                    try:
                        cursor.execute('''
                            INSERT INTO drivers (driver_id, name, phone, guardian_id)
                            VALUES (%s, %s, %s, %s)
                        ''', (temp_driver_id, driver_name, '00000000000', guardian_id))
                        conn.commit()
                    except Exception as e:
                        print(f"⚠️ Error creating temp driver: {e}")
                    
                    driver_id = temp_driver_id
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No guardian found for this alert'
                    }), 404
        
        # Convert detection_details to JSON string for storage
        detection_details_json = json.dumps(detection_details) if detection_details else None
        
        with get_db_cursor() as cursor:
            # Create alert
            cursor.execute('''
                INSERT INTO alerts (driver_id, guardian_id, severity, message, detection_details, source)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (driver_id, guardian_id, severity, message, detection_details_json, 'drowsiness_detection'))
            
            cursor.execute('SELECT LASTVAL()')
            alert_id_result = cursor.fetchone()
            alert_id = alert_id_result[0] if alert_id_result else None
            
            # Log drowsiness event
            cursor.execute('''
                INSERT INTO drowsiness_events (driver_id, guardian_id, confidence, state, ear, mar, perclos)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                driver_id, 
                guardian_id, 
                confidence,
                detection_details.get('state', 'unknown'),
                detection_details.get('ear', 0.0),
                detection_details.get('mar', 0.0),
                detection_details.get('perclos', 0.0)
            ))
            
            # Log activity
            cursor.execute('''
                INSERT INTO activity_log (guardian_id, action, details)
                VALUES (%s, %s, %s)
            ''', (guardian_id, 'ALERT_GENERATED', 
                f'Alert for driver {driver_name}: {message} (Confidence: {confidence:.1%})'))
            
            # Prepare alert data for WebSocket
            alert_data = {
                'alert_id': alert_id,
                'driver_id': driver_id,
                'driver_name': driver_name,
                'guardian_id': guardian_id,
                'guardian_name': guardian_name,
                'severity': severity,
                'message': message,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'detection_details': detection_details,
                'acknowledged': False
            }
            
            # Emit socket events
            socketio.emit('new_alert', alert_data)
            
            # Send to specific guardian clients
            for client_id, client_info in connected_clients.items():
                if client_info.get('guardian_id') == guardian_id and client_info.get('authenticated'):
                    socketio.emit('guardian_alert', alert_data, room=client_id)
            
            return jsonify({
                'success': True,
                'alert_id': alert_id,
                'data': alert_data,
                'message': f'Alert sent to guardian {guardian_name}'
            })
        
    except Exception as e:
        print(f"❌ Error in send_alert: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/guardian/alerts', methods=['GET'])
def get_guardian_alerts():
    """Get alerts for a guardian"""
    try:
        guardian_id = request.args.get('guardian_id')
        token = request.args.get('token')
        limit = request.args.get('limit', 20, type=int)
        acknowledged = request.args.get('acknowledged', type=str)
        
        if not guardian_id or not token:
            return jsonify({
                'success': False,
                'error': 'Authentication required'
            }), 401
        
        # Validate session
        if not validate_session(guardian_id, token):
            return jsonify({
                'success': False,
                'error': 'Session expired or invalid'
            }), 401
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build query based on acknowledged filter
            base_query = '''
                SELECT a.*, d.name as driver_name
                FROM alerts a
                JOIN drivers d ON a.driver_id = d.driver_id
                WHERE a.guardian_id = %s
            '''
            params = [guardian_id]
            
            if acknowledged is not None:
                if acknowledged.lower() == 'true':
                    base_query += ' AND a.acknowledged = TRUE'
                elif acknowledged.lower() == 'false':
                    base_query += ' AND a.acknowledged = FALSE'
            
            base_query += ' ORDER BY a.timestamp DESC LIMIT %s'
            params.append(limit)
            
            cursor.execute(base_query, tuple(params))
            alerts = cursor.fetchall()
            
            result_alerts = []
            for alert in alerts:
                alert_dict = dict(alert)
                if alert_dict.get('detection_details'):
                    try:
                        alert_dict['detection_details'] = json.loads(alert_dict['detection_details'])
                    except:
                        pass
                result_alerts.append(alert_dict)
            
            return jsonify({
                'success': True,
                'session_valid': True,
                'count': len(alerts),
                'alerts': result_alerts
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/guardian/active-drivers', methods=['GET'])
def get_active_drivers():
    """Return list of drivers that should be monitored."""
    # Optionally require an API key for the AI service
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f'Bearer {os.getenv("AI_SERVICE_TOKEN")}':
        return jsonify({'error': 'Unauthorized'}), 401

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT driver_id, name as driver_name, guardian_id
            FROM drivers
            WHERE is_active = TRUE
            ORDER BY registration_date DESC
        ''')
        drivers = cursor.fetchall()
    return jsonify({'success': True, 'drivers': [dict(d) for d in drivers]})

@app.route('/api/guardian/acknowledge-alert', methods=['POST'])
def acknowledge_alert():
    """Acknowledge an alert"""
    try:
        data = request.json
        guardian_id = data.get('guardian_id')
        token = data.get('token')
        alert_id = data.get('alert_id')
        
        if not guardian_id or not token or not alert_id:
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400
        
        # Validate session
        if not validate_session(guardian_id, token):
            return jsonify({
                'success': False,
                'error': 'Session expired or invalid'
            }), 401
        
        with get_db_cursor() as cursor:
            # Check if alert belongs to this guardian
            cursor.execute('''
                SELECT a.*, d.name as driver_name 
                FROM alerts a
                JOIN drivers d ON a.driver_id = d.driver_id
                WHERE a.alert_id = %s AND a.guardian_id = %s
            ''', (alert_id, guardian_id))
            
            alert_result = cursor.fetchone()
            if not alert_result:
                return jsonify({
                    'success': False,
                    'error': 'Alert not found or not authorized'
                }), 404
            
            alert = dict(alert_result)
            
            # Acknowledge the alert
            cursor.execute('''
                UPDATE alerts SET acknowledged = TRUE 
                WHERE alert_id = %s AND guardian_id = %s
            ''', (alert_id, guardian_id))
            
            # Log activity
            cursor.execute('''
                INSERT INTO activity_log (guardian_id, action, details)
                VALUES (%s, %s, %s)
            ''', (guardian_id, 'ALERT_ACKNOWLEDGED', 
                f'Acknowledged alert #{alert_id} for driver {alert["driver_name"]}'))
            
            # Emit socket event for real-time update
            socketio.emit('alert_acknowledged', {
                'alert_id': alert_id,
                'guardian_id': guardian_id,
                'timestamp': datetime.now().isoformat()
            })
            
            return jsonify({
                'success': True,
                'alert_id': alert_id,
                'acknowledged': True,
                'message': 'Alert acknowledged successfully'
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== ADMIN AUTHENTICATION ====================
@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Admin login endpoint"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                'success': False,
                'error': 'Username and password required'
            }), 400
        
        ip = request.remote_addr
        if rate_limit_exceeded(ip, 'admin_login', limit=5):
            return jsonify({
                'success': False,
                'error': 'Too many login attempts'
            }), 429
        
        admin = verify_admin_credentials(username, password)
        
        if admin:
            token, expires_at = create_admin_session(username)
            cleanup_admin_sessions()
            
            log_activity(admin_username=username, action='ADMIN_LOGIN', 
                        details=f'Admin logged in from {request.remote_addr}')
            
            return jsonify({
                'success': True,
                'username': username,
                'full_name': admin['full_name'],
                'role': admin['role'],
                'email': admin['email'],
                'token': token,
                'expires': expires_at.isoformat(),
                'message': 'Admin login successful',
                'redirect_url': f'https://guardian-drive-app.web.app/admin-dashboard.html?username={username}&token={token}'
            })
        
        return jsonify({
            'success': False,
            'error': 'Invalid admin credentials'
        }), 401
        
    except Exception as e:
        print(f"❌ Admin login error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== ADMIN MANAGEMENT ====================
@app.route('/api/admin/db-drivers', methods=['GET'])
@require_admin_auth
def admin_get_drivers():
    """Get all drivers from database - ADMIN ONLY"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT d.*, g.full_name as guardian_name, g.phone as guardian_phone
                FROM drivers d
                LEFT JOIN guardians g ON d.guardian_id = g.guardian_id
                ORDER BY d.registration_date DESC
            ''')
            
            drivers = cursor.fetchall()
            drivers_list = [dict(driver) for driver in drivers]
            
        return jsonify({
            'success': True,
            'count': len(drivers_list),
            'drivers': drivers_list
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/db-alerts', methods=['GET'])
@require_admin_auth
def admin_get_alerts():
    """Get all alerts from database - ADMIN ONLY"""
    try:
        limit = request.args.get('limit', 100, type=int)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT a.*, d.name as driver_name, g.full_name as guardian_name
                FROM alerts a
                JOIN drivers d ON a.driver_id = d.driver_id
                JOIN guardians g ON a.guardian_id = g.guardian_id
                ORDER BY a.timestamp DESC
                LIMIT %s
            ''', (limit,))
            
            alerts = cursor.fetchall()
            
            result = []
            for alert in alerts:
                alert_dict = dict(alert)
                if alert_dict.get('detection_details'):
                    try:
                        alert_dict['detection_details'] = json.loads(alert_dict['detection_details'])
                    except:
                        pass
                result.append(alert_dict)
            
        return jsonify({
            'success': True,
            'count': len(result),
            'alerts': result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/db-guardians', methods=['GET'])
@require_admin_auth
def admin_get_guardians():
    """Get all guardians from database - ADMIN ONLY"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT g.*, 
                       (SELECT COUNT(*) FROM drivers d WHERE d.guardian_id = g.guardian_id) as driver_count,
                       (SELECT COUNT(*) FROM alerts a WHERE a.guardian_id = g.guardian_id) as alert_count
                FROM guardians g
                ORDER BY g.registration_date DESC
            ''')
            
            guardians = cursor.fetchall()
            guardians_list = [dict(guardian) for guardian in guardians]
            
        return jsonify({
            'success': True,
            'count': len(guardians_list),
            'guardians': guardians_list
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
@require_admin_auth
def admin_stats():
    """Admin statistics endpoint - ADMIN ONLY"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            cursor.execute('SELECT COUNT(*) as total_alerts FROM alerts')
            result = cursor.fetchone()
            stats['total_alerts'] = result['total_alerts'] if isinstance(result, dict) else result[0]
            
            cursor.execute('SELECT COUNT(*) as total_drivers FROM drivers')
            result = cursor.fetchone()
            stats['total_drivers'] = result['total_drivers'] if isinstance(result, dict) else result[0]
            
            cursor.execute('SELECT COUNT(*) as total_guardians FROM guardians')
            result = cursor.fetchone()
            stats['total_guardians'] = result['total_guardians'] if isinstance(result, dict) else result[0]
            
        return jsonify({
            'success': True,
            'statistics': stats,
            'system_status': {
                'database': 'postgresql',
                'connected_clients': len(connected_clients),
                'admin_sessions': len(admin_sessions),
                'server_time': datetime.now().isoformat(),
                'firebase_integration': True,
                'cloudinary_enabled': CLOUDINARY_ENABLED,
                'allowed_origins': ALLOWED_ORIGINS,
                'frontend_url': 'https://guardian-drive-app.web.app'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== DEBUG ENDPOINTS ====================
@app.route('/api/debug/login-flow', methods=['POST'])
def debug_login_flow():
    """Debug the exact login flow step by step"""
    try:
        data = request.json
        phone = data.get('phone', '').strip()
        password = data.get('password', '')
        
        result = {
            'success': True,
            'steps': [],
            'input': {
                'phone': phone,
                'password': password,
                'password_length': len(password)
            }
        }
        
        # Step 1: Initial input
        result['steps'].append({
            'step': 1,
            'action': 'Input received',
            'phone': phone,
            'password': '***' + password[-3:] if len(password) > 3 else '***'
        })
        
        # Step 2: Clean phone
        phone_clean = str(phone).strip()
        phone_clean = re.sub(r'[\s\-\(\)\+]', '', phone_clean)
        result['steps'].append({
            'step': 2,
            'action': 'Clean phone',
            'phone_clean': phone_clean,
            'is_digits': phone_clean.isdigit()
        })
        
        if not phone_clean.isdigit():
            result['error'] = 'Phone contains non-digits'
            result['success'] = False
            return jsonify(result)
        
        # Step 3: Convert to 09 format
        lookup_phone = phone_clean
        conversion_note = 'No conversion needed'
        
        if len(lookup_phone) == 12 and lookup_phone.startswith('639'):
            lookup_phone = '09' + lookup_phone[3:]
            conversion_note = '639XXXXXXXXX -> 09XXXXXXXXX'
        elif len(lookup_phone) == 11 and lookup_phone.startswith('63'):
            lookup_phone = '09' + lookup_phone[2:]
            conversion_note = '63XXXXXXXXX -> 09XXXXXXXXX'
        elif len(lookup_phone) == 10 and lookup_phone.startswith('9'):
            lookup_phone = '0' + lookup_phone
            conversion_note = '9XXXXXXXXX -> 09XXXXXXXXX'
        elif len(lookup_phone) >= 10:
            last_10_digits = lookup_phone[-10:]
            lookup_phone = '09' + last_10_digits
            conversion_note = 'Last 10 digits -> 09XXXXXXXXX'
        
        result['steps'].append({
            'step': 3,
            'action': 'Convert phone format',
            'lookup_phone': lookup_phone,
            'conversion_note': conversion_note,
            'is_valid_format': lookup_phone.startswith('09') and len(lookup_phone) == 11
        })
        
        if not lookup_phone.startswith('09') or len(lookup_phone) != 11:
            result['error'] = f'Invalid phone format: {lookup_phone}'
            result['success'] = False
            return jsonify(result)
        
        # Step 4: Check database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # First check what's in the database
            cursor.execute('SELECT phone, guardian_id FROM guardians ORDER BY guardian_id DESC LIMIT 5')
            all_users = cursor.fetchall()
            
            result['steps'].append({
                'step': 4,
                'action': 'Database check - all users',
                'all_users_in_db': [dict(user) for user in all_users]
            })
            
            # Now search for our phone
            cursor.execute('''
                SELECT guardian_id, full_name, password_hash, is_active
                FROM guardians 
                WHERE phone = %s
            ''', (lookup_phone,))
            
            db_result = cursor.fetchone()
            
            if not db_result:
                result['steps'].append({
                    'step': 5,
                    'action': 'User not found in database',
                    'lookup_phone_used': lookup_phone,
                    'user_found': False
                })
                result['error'] = f'No user found with phone: {lookup_phone}'
                result['success'] = False
                return jsonify(result)
            
            # Handle both dict and tuple
            if isinstance(db_result, dict):
                guardian_id = db_result['guardian_id']
                full_name = db_result['full_name']
                stored_hash = db_result['password_hash']
                is_active = db_result['is_active']
            else:
                guardian_id, full_name, stored_hash, is_active = db_result
            
            result['steps'].append({
                'step': 5,
                'action': 'User found in database',
                'guardian_id': guardian_id,
                'full_name': full_name,
                'is_active': is_active,
                'stored_hash_preview': stored_hash[:30] + '...' if stored_hash else 'None',
                'hash_length': len(stored_hash) if stored_hash else 0,
                'is_bcrypt': stored_hash.startswith('$2') if stored_hash else False
            })
            
            if not is_active:
                result['error'] = 'Account is not active'
                result['success'] = False
                return jsonify(result)
            
            # Step 6: Clean password
            cleaned_password = clean_password(password)
            result['steps'].append({
                'step': 6,
                'action': 'Clean password',
                'cleaned_password': '***' + cleaned_password[-3:] if len(cleaned_password) > 3 else '***',
                'cleaned_length': len(cleaned_password)
            })
            
            # Step 7: Manual verification
            try:
                password_bytes = cleaned_password.encode('utf-8')
                hash_bytes = stored_hash.encode('utf-8')
                
                manual_verify = bcrypt.checkpw(password_bytes, hash_bytes)
                
                result['steps'].append({
                    'step': 7,
                    'action': 'Manual bcrypt.checkpw verification',
                    'result': manual_verify,
                    'method': 'bcrypt.checkpw(password_bytes, hash_bytes)'
                })
                
                # Step 8: Use verify_password function
                func_verify = verify_password(password, stored_hash)
                
                result['steps'].append({
                    'step': 8,
                    'action': 'verify_password() function',
                    'result': func_verify,
                    'method': 'verify_password(password, stored_hash)'
                })
                
                result['final_verification'] = {
                    'manual_bcrypt': manual_verify,
                    'verify_password_func': func_verify,
                    'should_login_succeed': manual_verify and func_verify
                }
                
                if manual_verify and func_verify:
                    result['login_possible'] = True
                    result['message'] = 'Password verification successful - login should work'
                else:
                    result['login_possible'] = False
                    result['message'] = 'Password verification failed'
                
            except Exception as verify_error:
                result['steps'].append({
                    'step': 7,
                    'action': 'Verification error',
                    'error': str(verify_error)
                })
                result['error'] = f'Verification error: {verify_error}'
                result['success'] = False
            
        return jsonify(result)
                
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/fix-db-schema', methods=['POST'])
def fix_db_schema():
    """Fix database schema by adding missing columns"""
    try:
        with get_db_cursor() as cursor:
            # Check current schema
            cursor.execute('''
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'guardians'
            ''')
            columns = [row[0] for row in cursor.fetchall()]
            
            fixes = []
            
            # Add google_id if missing
            if 'google_id' not in columns:
                cursor.execute('ALTER TABLE guardians ADD COLUMN google_id TEXT UNIQUE')
                fixes.append('Added google_id column')
            
            # Add auth_provider if missing
            if 'auth_provider' not in columns:
                cursor.execute('ALTER TABLE guardians ADD COLUMN auth_provider TEXT DEFAULT \'phone\'')
                fixes.append('Added auth_provider column')
            
            # Check again
            cursor.execute('''
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'guardians'
                ORDER BY ordinal_position
            ''')
            final_schema = cursor.fetchall()
            
            return jsonify({
                'success': True,
                'fixes_applied': fixes,
                'final_schema': [dict(zip(['column_name', 'data_type', 'is_nullable'], row)) for row in final_schema],
                'message': 'Schema fixes applied successfully'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# ==================== CLOUDINARY TEST ENDPOINT ====================
@app.route('/api/test-cloudinary', methods=['POST'])
def test_cloudinary():
    """Test Cloudinary upload functionality"""
    try:
        if not CLOUDINARY_ENABLED:
            return jsonify({
                'success': False,
                'error': 'Cloudinary not configured. Check environment variables.'
            }), 500
        
        data = request.json
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Clean base64 data
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Generate test public ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        public_id = f"test_uploads/test_image_{timestamp}"
        
        print(f"🧪 Testing Cloudinary upload with public_id: {public_id}")
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            f"data:image/jpeg;base64,{image_base64}",
            public_id=public_id,
            folder="test_uploads",
            resource_type="image",
            overwrite=False,
            quality="auto:good",
            format="jpg"
        )
        
        cloudinary_url = upload_result.get('secure_url')
        
        return jsonify({
            'success': True,
            'message': 'Cloudinary upload successful!',
            'url': cloudinary_url,
            'public_id': upload_result.get('public_id'),
            'details': {
                'width': upload_result.get('width'),
                'height': upload_result.get('height'),
                'format': upload_result.get('format'),
                'resource_type': upload_result.get('resource_type'),
                'bytes': upload_result.get('bytes')
            }
        })
        
    except Exception as e:
        print(f"❌ Cloudinary test error: {e}")
        return jsonify({
            'success': False,
            'error': f'Cloudinary test failed: {str(e)}'
        }), 500

# ==================== APPLICATION STARTUP ====================
def startup_tasks():
    """Run startup tasks"""
    print(f"\n{'='*70}")
    print("🚗 DRIVER DROWSINESS ALERT SYSTEM - CAPSTONE PROJECT")
    print(f"{'='*70}")
    
    print("🌐 DEPLOYMENT: Firebase Hosting + Render Backend")
    print("📊 Database: PostgreSQL (Persistent)")
    print("🔒 Security: bcrypt password hashing enabled")
    print("☁️  Cloudinary: Image storage enabled" if CLOUDINARY_ENABLED else "⚠️  Cloudinary: Not configured")
    print("🔥 Firebase: Hosting integration enabled")
    
    # Check environment variables
    print("\n🔧 Environment Check:")
    
    # Get and mask database URL for security
    database_url = os.environ.get('DATABASE_URL')
    if database_url:
        try:
            # Parse URL to mask password
            parsed_url = urllib.parse.urlparse(database_url)
            if parsed_url.password:
                masked_url = database_url.replace(parsed_url.password, '******')
                print(f"   ✅ DATABASE_URL: {masked_url}")
            else:
                print(f"   ✅ DATABASE_URL: [configured]")
        except:
            print(f"   ✅ DATABASE_URL: [configured]")
    else:
        print("❌ DATABASE_URL not found")
    
    # Check Google OAuth config
    google_client_id = os.environ.get('GOOGLE_CLIENT_ID')
    if google_client_id and google_client_id != 'your-google-client-id-here':
        print(f"   ✅ GOOGLE_CLIENT_ID: [configured]")
    else:
        print(f"   ⚠️  GOOGLE_CLIENT_ID: not configured")
    
    # Check Cloudinary config
    if CLOUDINARY_ENABLED:
        print(f"   ✅ CLOUDINARY: Configured for cloud: {CLOUDINARY_CLOUD_NAME}")
    else:
        print(f"   ⚠️  CLOUDINARY: Not configured")
    
    # Check secret key
    secret_key = os.environ.get('SECRET_KEY')
    if secret_key:
        print(f"   ✅ SECRET_KEY: [configured]")
    else:
        print(f"   ⚠️  SECRET_KEY: using generated key")
    
    # Firebase integration
    print(f"   ✅ Firebase Integration: Enabled")
    print(f"   ✅ Frontend URL: https://guardian-drive-app.web.app")
    print(f"   ✅ Allowed Origins: {len(ALLOWED_ORIGINS)} domains configured")
    
    # Database initialization
    print("\n🗄️  Database Initialization:")
    try:
        if init_db():
            print("✅ Database initialized successfully")
        else:
            print("⚠️ Database initialization had issues")
        
        # Update schema for google_id column
        print("\n🔧 Schema Updates:")
        update_db_schema()
    except Exception as e:
        print(f"⚠️ Database initialization error: {e}")
        import traceback
        traceback.print_exc()
    
    # Security check
    print("\n🔐 Security Status:")
    print(f"   Admin password hash: {ADMIN_PASSWORD_HASH[:20]}...")
    print(f"   Hash algorithm: bcrypt (12 rounds)")
    
    # Network and server info
    print("\n🌐 Network Configuration:")
    print(f"   Host: 0.0.0.0 (all interfaces)")
    port = int(os.environ.get('PORT', 5000))
    print(f"   Port: {port}")
    
    # Check if running in Render
    render_env = os.environ.get('RENDER', '')
    if render_env:
        print(f"   ✅ Running on Render.com")
        print(f"   Firebase Frontend: https://guardian-drive-app.web.app")
    else:
        print(f"   ⚠️  Not running on Render (local development)")
    
    # API endpoints summary
    print(f"\n🔗 Available API Endpoints:")
    endpoints = [
        ("GET", "/api/health", "Health check"),
        ("POST", "/api/login", "Guardian login"),
        ("POST", "/api/google-login", "Google OAuth login"),
        ("POST", "/api/register-guardian", "Guardian registration"),
        ("POST", "/api/register-driver", "Register driver with Cloudinary"),
        ("POST", "/api/send-alert", "Send drowsiness alert"),
        ("GET", "/api/guardian/dashboard", "Guardian dashboard"),
        ("POST", "/api/admin/login", "Admin login"),
        ("POST", "/api/test-cloudinary", "Test Cloudinary upload"),
    ]
    
    for method, path, desc in endpoints:
        print(f"   • {method:6} {path:30} - {desc}")
    
    # Authentication info
    print(f"\n🔐 Authentication Info:")
    print("  • Admin Username: admin")
    print("  • Admin Password: admin123")
    print("  • Guardian Registration: Open to public")
    print("  • Google OAuth: " + ("Enabled" if google_client_id and google_client_id != 'your-google-client-id-here' else "Disabled"))
    print("  • Firebase Domain: guardian-drive-app.web.app")
    print("  • Cloudinary: " + ("Enabled" if CLOUDINARY_ENABLED else "Disabled"))
    
    # Real-time features
    print(f"\n⚡ Real-time Features:")
    print(f"  • WebSocket Alerts: Enabled")
    print(f"  • Active Clients: {len(connected_clients)}")
    print(f"  • Active Sessions: {len(active_sessions)}")
    print(f"  • Firebase CORS: Configured")
    
    # Startup completion
    print(f"\n✅ Startup Tasks Complete")
    print(f"   Server Time: {datetime.now().strftime('%Y-%m-d %H:%M:%S')}")
    
    print(f"\n🚀 Ready to accept connections")
    print(f"   Render Backend: https://driver-drowsiness-with-alert.onrender.com")
    print(f"   Firebase Frontend: https://guardian-drive-app.web.app")
    print(f"   Health Check: https://driver-drowsiness-with-alert.onrender.com/api/health")
    print(f"   WebSocket: wss://driver-drowsiness-with-alert.onrender.com/socket.io/")
    print(f"{'='*70}\n")

# ==================== MAIN ENTRY POINT ====================
if __name__ == '__main__':
    # Run startup tasks
    startup_tasks()
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    
    print(f"\n{'='*70}")
    print("🚀 STARTING DRIVER ALERT SYSTEM BACKEND")
    print(f"{'='*70}")
    print(f"📡 REST API: https://driver-drowsiness-with-alert.onrender.com/api/")
    print(f"🔌 WebSocket: wss://driver-drowsiness-with-alert.onrender.com/socket.io/")
    print(f"🔥 Firebase Frontend: https://guardian-drive-app.web.app")
    print(f"🐍 Python Version: 3.10.11")
    print(f"📦 Flask Version: 2.3.3")
    print(f"📦 Flask-SocketIO: 5.3.4")
    print(f"📦 Eventlet: 0.33.3")
    print(f"📦 dnspython: 2.3.0 (Python 3.10 compatible)")
    print(f"🌐 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔐 Google OAuth: {'Enabled' if GOOGLE_CLIENT_ID else 'Disabled'}")
    print(f"☁️  Cloudinary: {'Enabled' if CLOUDINARY_ENABLED else 'Disabled'}")
    print(f"📊 Database: PostgreSQL")
    print(f"⚡ Async Mode: eventlet")
    print(f"📡 WebSocket Transports: polling → websocket")
    print(f"📡 WebSocket Path: /socket.io/")
    print(f"🔌 Allowed Origins: {len(ALLOWED_ORIGINS)} domains")
    print(f"{'='*70}\n")
    
    # Force DNS resolution for Render - FIX for Python 3.10
    try:
        import dns.resolver
        dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
        dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']  # Google DNS
        print("✅ DNS resolver configured for Render")
    except Exception as e:
        print(f"⚠️ DNS resolver configuration warning: {e}")
    
    # IMPORTANT: Use eventlet WSGI server for WebSocket support
    try:
        import eventlet
        import eventlet.wsgi
        from eventlet import listen
        
        print("✅ Eventlet imported successfully")
        print(f"   Eventlet version: {eventlet.__version__}")
        
        # Monkey patch for Python 3.10 compatibility
        eventlet.monkey_patch(
            socket=True,
            select=True,
            time=True,
            os=True,
            thread=True,
            subprocess=True
        )
        print("✅ Eventlet monkey patching applied")
        
        # Create socket with proper configuration
        listen_socket = eventlet.listen((host, port))
        
        # Set socket options for better performance
        listen_socket.setsockopt(eventlet.socket.SOL_SOCKET, eventlet.socket.SO_REUSEADDR, 1)
        
        # Configure connection pooling for Python 3.10
        eventlet.wsgi.MAX_HEADER_LINE = 16384
        eventlet.wsgi.MAX_REQUEST_LINE = 32768
        eventlet.wsgi.MAX_READ_BYTES = 65536
        eventlet.wsgi.DEFAULT_MAX_SIMULTANEOUS_REQUESTS = 1000
        
        print(f"✅ Server socket created on {host}:{port}")
        print(f"✅ WebSocket support enabled with eventlet")
        print(f"✅ Ready to accept connections...\n")
        
        # Add connection tracking middleware
        def connection_count_middleware(environ, start_response):
            """Middleware to track connections"""
            client_id = environ.get('REMOTE_ADDR', 'unknown')
            print(f"📡 New connection from {client_id} at {datetime.now().strftime('%H:%M:%S')}")
            return app(environ, start_response)
        
        # Start the server with proper configuration
        eventlet.wsgi.server(
            listen_socket,
            connection_count_middleware,  # Wrap app with middleware
            log_output=True,               # Enable logging
            keepalive=True,                 # Enable keepalive
            max_size=8096,                   # Maximum number of concurrent connections
            protocol=eventlet.wsgi.HttpProtocol,
            timeout=60,                      # Socket timeout
            max_http_version="HTTP/1.1",
            debug=False,                      # Disable debug mode in production
            log_format='%(client_ip)s - "%(request_line)s" %(status_code)s %(body_length)s'
        )
        
    except ImportError as e:
        print(f"❌ Error importing eventlet: {e}")
        print("⚠️ Falling back to gunicorn with eventlet worker")
        
        # Instructions for gunicorn fallback
        print("\n📋 To run with gunicorn instead, use:")
        print("   gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT server:app")
        
        # Try socketio.run as last resort
        print("\n⚠️ Attempting fallback with socketio.run()")
        print(f"   Starting server on {host}:{port}...\n")
        
        try:
            socketio.run(
                app,
                host=host,
                port=port,
                debug=False,
                use_reloader=False,
                log_output=True,
                allow_unsafe_werkzeug=False
            )
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            print("\n💡 SUGGESTION: Try running with gunicorn:")
            print("   gunicorn --worker-class eventlet -w 1 -k eventlet --bind 0.0.0.0:$PORT server:app")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Fatal error starting server: {e}")
        import traceback
        traceback.print_exc()
        
        # Try socketio.run as last resort
        print("\n⚠️ Attempting emergency fallback with socketio.run()")
        print(f"   Starting server on {host}:{port}...\n")
        
        try:
            socketio.run(
                app,
                host=host,
                port=port,
                debug=False,
                use_reloader=False,
                log_output=True
            )
        except Exception as emergency_error:
            print(f"❌ Emergency fallback also failed: {emergency_error}")
            print("\n💡 CRITICAL: Check your deployment configuration on Render")
            print("   Make sure the following environment variables are set:")
            print("   - PYTHON_VERSION=3.10.11")
            print("   - PORT=5000")
            print("   - WEBSOCKET_ENABLED=true")
            sys.exit(1)