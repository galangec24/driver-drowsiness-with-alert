import os
from pathlib import Path
import sys
import dns.resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4'] 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
import requests
import io

# Supabase imports
from supabase import create_client, Client

# Cloudinary imports
import cloudinary
import cloudinary.uploader
import cloudinary.api
import cloudinary.exceptions  
from cloudinary.utils import cloudinary_url
# Google OAuth imports
import jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import google.auth.transport.requests
import io

import logging

# Setup logger (if not already done)
logger = logging.getLogger(__name__)


#region Backend Setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Patch for async
eventlet.monkey_patch()

GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')
JITSI_DOMAIN = os.getenv('JITSI_DOMAIN', 'meet.jit.si')
FACEBOOK_APP_ID = os.environ.get('FACEBOOK_APP_ID', '')
FACEBOOK_APP_SECRET = os.environ.get('FACEBOOK_APP_SECRET', '')
SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')
SUPABASE_JWT_SECRET = os.environ.get('SUPABASE_JWT_SECRET', '')

# Initialize Supabase client
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"✅ Supabase client initialized")
else:
    print("⚠️ Supabase not configured. Check SUPABASE_URL and SUPABASE_KEY environment variables.")

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
#end Region

#region Firebase CORS
ALLOWED_ORIGINS = [
    'https://driver-drowsiness-with-alert.onrender.com/api/facebook-login',
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
    'http://localhost:8080',
    'https://localhost:8080'
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
    cors_allowed_origins=[
        'https://guardian-drive-app.web.app',
        'https://guardian-drive-app.firebaseapp.com',
        'http://localhost:5000',
        'http://localhost:8080',
        'http://localhost:3000'
    ],
    async_mode='eventlet',
    ping_timeout=30,
    ping_interval=15,
    max_http_buffer_size=1e6,
    transports=['polling', 'websocket'],  
    allow_upgrades=True,
    manage_session=False,
    cookie=None,
    always_connect=True,
    logger=True,
    engineio_logger=True,
    log_output=True,
    path='socket.io',
    cors_credentials=True,
    max_guest_sessions=1000,
    preserve_context=True,
    websocket_ping_interval=25,
    websocket_ping_timeout=60
)

def monitor_webrtc_connections():
    """Background task to monitor WebRTC connections and clean up stale ones"""
    while True:
        time.sleep(30)  # Check every 30 seconds
        try:
            current_time = datetime.now()
            stale_connections = []
            
            for client_id, info in connected_clients.items():
                # Check if connection is stale (no ping for 60 seconds)
                last_ping = info.get('last_ping')
                if last_ping and (current_time - last_ping) > timedelta(seconds=60):
                    if info.get('type') in ['driver', 'guardian']:
                        print(f"🧹 Found stale {info.get('type')} connection: {client_id}")
                        stale_connections.append(client_id)
            
            # Remove stale connections
            for client_id in stale_connections:
                try:
                    socketio.disconnect(client_id, silent=True)
                    if client_id in connected_clients:
                        del connected_clients[client_id]
                except:
                    pass
            
            if stale_connections:
                print(f"   Cleaned up {len(stale_connections)} stale connections")
                
        except Exception as e:
            print(f"⚠️ Error in connection monitor: {e}")

# Start the monitor thread
threading.Thread(target=monitor_webrtc_connections, daemon=True).start()
print("✅ WebRTC connection monitor started")

#end Region

@app.route('/socket.io/', methods=['OPTIONS'])
@app.route('/socket.io/<path:path>', methods=['OPTIONS'])
def handle_socketio_preflight(path=None):
    """Handle CORS preflight for socket.io paths"""
    response = jsonify({'status': 'ok'})
    origin = request.headers.get('Origin')
    
    if origin and any(allowed in origin for allowed in ['guardian-drive-app.web.app', 'localhost']):
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
    
    return response, 200

#region Admin (Initial)
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
#end Region

#region Clean Password
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

#end Region

#region Admin Data
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
#end Region

#region Database - Supabase
@contextmanager
def get_db_connection():
    """Context manager for Supabase connection"""
    if not supabase:
        raise Exception("Supabase client not initialized")
    yield supabase

@contextmanager
def get_db_cursor():
    """
    Context manager for Supabase operations.
    Note: Supabase doesn't use traditional cursors, so we yield the client
    and handle transactions differently.
    """
    if not supabase:
        raise Exception("Supabase client not initialized")
    yield supabase

# Store connected clients and active sessions
connected_clients = {}
active_sessions = {}

def cleanup_stale_clients():
    """Remove clients that haven't sent a ping in 2 minutes."""
    while True:
        time.sleep(300)  # run every 5 minutes
        now = datetime.now()
        stale_sids = []
        for sid, info in list(connected_clients.items()):
            last_ping = info.get('last_ping')
            if last_ping and (now - last_ping) > timedelta(minutes=2):
                print(f"🧹 Cleaning up stale client {sid} (last ping {last_ping})")
                stale_sids.append(sid)
        for sid in stale_sids:
            connected_clients.pop(sid, None)
        if stale_sids:
            print(f"   Removed {len(stale_sids)} stale clients")

# Start the thread (daemon so it exits when main process ends)
threading.Thread(target=cleanup_stale_clients, daemon=True).start()

# ==================== SUPABASE INITIALIZATION ====================
def init_db():
    """Initialize database with all required tables in Supabase"""
    print("🗄️  Initializing Supabase database...")
    
    try:
        if not supabase:
            print("❌ Supabase client not initialized")
            return False

        result = supabase.table('guardians').select('*').limit(1).execute()
        print("✅ Successfully connected to Supabase")
        
        # Check if we can access the tables
        tables_to_check = ['guardians', 'drivers', 'alerts', 'face_images', 
                          'drowsiness_events', 'activity_log', 'session_tokens', 
                          'admin_activity_log']
        
        for table in tables_to_check:
            try:
                result = supabase.table(table).select('*').limit(1).execute()
                print(f"   ✅ Table '{table}' exists and is accessible")
            except Exception as e:
                print(f"   ⚠️ Table '{table}' might not exist: {e}")
                print(f"   Please create this table in Supabase SQL editor")
        
        print("✅ Supabase connection verified successfully")
        return True
        
    except Exception as e:
        print(f"❌ Supabase initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_location_tables():
    """Create location tracking tables if they don't exist"""
    if not supabase:
        print("⚠️ [DB] Supabase not initialized, skipping table creation")
        return
    
    try:
        # Note: In Supabase, tables should be created via SQL editor
        # This function just checks if they exist
        print("\n🗄️ [DB] Checking location tables...")
        
        # Check guardian_locations table
        try:
            result = supabase.table('guardian_locations').select('*').limit(1).execute()
            print("   ✅ guardian_locations table exists")
        except Exception as e:
            print("   ⚠️ guardian_locations table may not exist")
            print("   Please create it in Supabase SQL editor with:")
            print("""
            CREATE TABLE guardian_locations (
                id BIGSERIAL PRIMARY KEY,
                guardian_id TEXT NOT NULL,
                latitude FLOAT NOT NULL,
                longitude FLOAT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                accuracy FLOAT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                INDEX idx_guardian_locations_guardian (guardian_id),
                INDEX idx_guardian_locations_timestamp (timestamp)
            );
            """)
        
        # Check driver_locations table
        try:
            result = supabase.table('driver_locations').select('*').limit(1).execute()
            print("   ✅ driver_locations table exists")
        except Exception as e:
            print("   ⚠️ driver_locations table may not exist")
            print("   Please create it in Supabase SQL editor with:")
            print("""
            CREATE TABLE driver_locations (
                id BIGSERIAL PRIMARY KEY,
                driver_id TEXT NOT NULL,
                guardian_id TEXT NOT NULL,
                latitude FLOAT NOT NULL,
                longitude FLOAT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                drowsiness FLOAT DEFAULT 0,
                drowsiness_level TEXT DEFAULT 'ALERT',
                accuracy FLOAT,
                method TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                INDEX idx_driver_locations_driver (driver_id),
                INDEX idx_driver_locations_guardian (guardian_id),
                INDEX idx_driver_locations_timestamp (timestamp)
            );
            """)
        
    except Exception as e:
        print(f"⚠️ [DB] Error checking location tables: {e}")

def update_db_schema():
    """Check Supabase schema - migrations should be done via SQL editor"""
    print("🔧 Checking Supabase schema...")
    
    try:
        if not supabase:
            print("❌ Supabase client not initialized")
            return False
        
        # Check if google_id column exists in guardians table by trying to query it
        try:
            result = supabase.table('guardians').select('google_id').limit(1).execute()
            print("   ✅ google_id column exists in guardians table")
        except Exception as e:
            print("   ⚠️ google_id column might not exist in guardians table")
            print("   Please add it via Supabase SQL editor:")
            print("   ALTER TABLE guardians ADD COLUMN google_id TEXT UNIQUE;")
        
        # Check auth_provider column
        try:
            result = supabase.table('guardians').select('auth_provider').limit(1).execute()
            print("   ✅ auth_provider column exists in guardians table")
        except Exception as e:
            print("   ⚠️ auth_provider column might not exist in guardians table")
            print("   Please add it via Supabase SQL editor:")
            print("   ALTER TABLE guardians ADD COLUMN auth_provider TEXT DEFAULT 'phone';")
        
        # Check face_embedding column in drivers
        try:
            result = supabase.table('drivers').select('face_embedding').limit(1).execute()
            print("   ✅ face_embedding column exists in drivers table")
        except Exception as e:
            print("   ⚠️ face_embedding column might not exist in drivers table")
            print("   Please add it via Supabase SQL editor:")
            print("   ALTER TABLE drivers ADD COLUMN face_embedding JSONB;")
        
        print("✅ Supabase schema check completed")
        return True
        
    except Exception as e:
        print(f"❌ Supabase schema check failed: {e}")
        return False

def get_guardian_for_driver(driver_id):
    """Get guardian ID for a driver with debug logging"""
    try:
        if not supabase:
            print("❌ Supabase not initialized in get_guardian_for_driver")
            return None
        
        print(f"🔍 Looking up guardian for driver: {driver_id}")
        
        # Try by driver_id first
        result = supabase.table('drivers') \
            .select('guardian_id, name, is_active') \
            .eq('driver_id', driver_id) \
            .eq('is_active', True) \
            .execute()
        
        if result.data and len(result.data) > 0:
            guardian_id = result.data[0]['guardian_id']
            driver_name = result.data[0].get('name', 'Unknown')
            print(f"✅ Found guardian {guardian_id} for driver {driver_id} ({driver_name})")
            return guardian_id
        
        # If not found by driver_id, try by reference_number
        result = supabase.table('drivers') \
            .select('guardian_id, name, is_active') \
            .eq('reference_number', driver_id) \
            .eq('is_active', True) \
            .execute()
        
        if result.data and len(result.data) > 0:
            guardian_id = result.data[0]['guardian_id']
            driver_name = result.data[0].get('name', 'Unknown')
            print(f"✅ Found guardian {guardian_id} for reference {driver_id} ({driver_name})")
            return guardian_id
        
        print(f"❌ No driver found with ID or reference: {driver_id}")
        return None
        
    except Exception as e:
        print(f"❌ Error getting guardian for driver {driver_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


#region Session AUTH
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

google_session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
google_session.mount("https://", adapter)
google_session.mount("http://", adapter)

# Custom request class for google-auth library
class CustomRequest(google.auth.transport.requests.Request):
    def __init__(self, session=None):
        super().__init__(session=session or google_session)

_certs_cache = None
_certs_cache_time = 0
CERTS_REFRESH_INTERVAL = 6 * 3600  # refresh every 6 hours

def verify_google_token_hybrid(token, client_id):
    """
    Verify Google token using Google's library but with custom HTTP transport.
    This avoids DNS issues while still using Google's reliable verification.
    """
    try:
        # Use Google's library with our custom session
        request = CustomRequest()
        
        # This will use our custom session with retries and proper DNS
        idinfo = id_token.verify_oauth2_token(
            token, 
            request,
            client_id,
            clock_skew_in_seconds=10  # Allow 10 seconds clock skew
        )
        
        return idinfo
        
    except Exception as e:
        print(f"❌ Google token verification failed with hybrid approach: {e}")
        # Fall back to manual verification with cached certs
        return verify_google_token_manual_fallback(token, client_id)

def verify_google_token_manual_fallback(token, client_id):
    """
    Manual verification with cached certificates as fallback.
    """
    try:
        # Get unverified header to find the key ID (kid)
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header['kid']
        
        # Fetch current certs (from cache)
        certs = get_certs_with_retry()
        if kid not in certs:
            raise Exception(f"Key ID {kid} not found in Google certs")
        
        public_key = certs[kid]
        
        # Verify the token
        decoded = jwt.decode(
            token,
            public_key,
            algorithms=['RS256'],
            audience=client_id,
            issuer=['accounts.google.com', 'https://accounts.google.com'],
            options={
                'verify_signature': True,
                'verify_exp': True,
                'verify_aud': True,
                'verify_iss': True
            }
        )
        return decoded
        
    except jwt.ExpiredSignatureError:
        raise Exception("Token has expired")
    except jwt.InvalidAudienceError:
        raise Exception("Token audience doesn't match")
    except jwt.InvalidIssuerError:
        raise Exception("Token issuer doesn't match")
    except Exception as e:
        raise Exception(f"Manual verification failed: {str(e)}")

def get_certs_with_retry(force_refresh=False):
    """Fetch Google certs with retry logic from multiple endpoints"""
    global _certs_cache, _certs_cache_time
    now = time.time()
    
    if force_refresh or _certs_cache is None or (now - _certs_cache_time) > CERTS_REFRESH_INTERVAL:
        # Try multiple cert endpoints
        cert_urls = [
            "https://www.googleapis.com/oauth2/v1/certs",
            "https://www.googleapis.com/oauth2/v3/certs",
            "https://oauth2.googleapis.com/certs"
        ]
        
        for url in cert_urls:
            try:
                print(f"🔄 Fetching Google certs from {url}")
                response = google_session.get(url, timeout=10)
                response.raise_for_status()
                _certs_cache = response.json()
                _certs_cache_time = now
                print(f"✅ Google certificates fetched successfully from {url}")
                return _certs_cache
            except Exception as e:
                print(f"⚠️ Failed to fetch from {url}: {e}")
                continue
        
        if _certs_cache is None:
            raise Exception("Unable to fetch Google certificates from any endpoint")
    
    return _certs_cache

# Pre-fetch certificates at startup
try:
    get_certs_with_retry()
    print("✅ Google certificates pre-fetched at startup")
except Exception as e:
    print(f"⚠️ Could not pre-fetch Google certificates: {e}")

# ==================== SESSION FUNCTIONS ====================
def generate_session_token():
    return secrets.token_urlsafe(32)

def create_session(guardian_id, ip_address=None, user_agent=None):
    token = generate_session_token()
    expires_at = datetime.now() + timedelta(hours=24)
    
    try:
        if not supabase:
            return None
            
        # Invalidate existing sessions
        supabase.table('session_tokens') \
            .update({'is_valid': False}) \
            .eq('guardian_id', guardian_id) \
            .eq('is_valid', True) \
            .execute()
        
        # Create new session
        data = {
            'guardian_id': guardian_id,
            'token': token,
            'expires_at': expires_at.isoformat(),
            'ip_address': ip_address,
            'user_agent': user_agent,
            'is_valid': True,
            'created_at': datetime.now().isoformat()
        }
        
        result = supabase.table('session_tokens').insert(data).execute()
        
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
    
    # Check Supabase
    try:
        if not supabase:
            return False
            
        result = supabase.table('session_tokens') \
            .select('*') \
            .eq('guardian_id', guardian_id) \
            .eq('token', token) \
            .eq('is_valid', True) \
            .gte('expires_at', datetime.now().isoformat()) \
            .execute()
        
        exists = len(result.data) > 0
        
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
        if not supabase:
            return False
            
        query = supabase.table('session_tokens') \
            .update({'is_valid': False}) \
            .eq('guardian_id', guardian_id)
        
        if token:
            query = query.eq('token', token)
        
        query.execute()
        
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

#End Region

#region Guardian Auth
def verify_guardian_credentials(identifier, password):
    """Verify guardian login credentials using bcrypt - Supports email or phone"""
    try:
        print(f"\n🔍 [LOGIN VERIFY] Starting verification")
        print(f"   Identifier received: '{identifier}'")
        print(f"   Password received: '{password}' (length: {len(password)})")
        
        if not supabase:
            print("❌ [LOGIN VERIFY] Supabase client not initialized")
            return None
        
        # Check if identifier is email or phone
        is_email = '@' in identifier and '.' in identifier
        
        if is_email:
            # Email login - use as-is
            email = identifier.strip().lower()
            print(f"   Using email login: '{email}'")
            
            # Query by email
            result = supabase.table('guardians') \
                .select('guardian_id, full_name, password_hash, is_active, phone') \
                .eq('email', email) \
                .execute()
            
            if not result.data or len(result.data) == 0:
                print(f"❌ [LOGIN VERIFY] No user found with email: '{email}'")
                
                # Debug: Show some emails in DB
                sample = supabase.table('guardians') \
                    .select('email, full_name') \
                    .not_.is_('email', 'null') \
                    .limit(5) \
                    .execute()
                print(f"   First 5 emails in DB: {[item['email'] for item in sample.data]}")
                
                return None
            
            guardian_data = result.data[0]
                
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
            
            # Query by phone
            result = supabase.table('guardians') \
                .select('guardian_id, full_name, password_hash, is_active, email') \
                .eq('phone', lookup_phone) \
                .execute()
            
            if not result.data or len(result.data) == 0:
                print(f"❌ [LOGIN VERIFY] No user found with phone: '{lookup_phone}'")
                
                # Debug: Show what's in the database
                sample = supabase.table('guardians') \
                    .select('phone, full_name') \
                    .not_.is_('phone', 'null') \
                    .limit(5) \
                    .execute()
                print(f"   First 5 phones in DB: {[item['phone'] for item in sample.data]}")
                
                return None
            
            guardian_data = result.data[0]
        
        print(f"   Result data: {guardian_data}")
        
        guardian_id = guardian_data['guardian_id']
        full_name = guardian_data['full_name']
        stored_hash = guardian_data['password_hash']
        is_active = guardian_data['is_active']
        
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
                supabase.table('guardians') \
                    .update({'last_login': datetime.now().isoformat()}) \
                    .eq('guardian_id', guardian_id) \
                    .execute()
            except Exception as update_error:
                print(f"⚠️ [LOGIN VERIFY] Error updating last login: {update_error}")
            
            return {
                'guardian_id': guardian_id, 
                'full_name': full_name,
                'identifier': email if is_email else lookup_phone,
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
        if not supabase:
            return None
            
        result = supabase.table('guardians') \
            .select('guardian_id, full_name, phone, email, address, registration_date, last_login, auth_provider') \
            .eq('guardian_id', guardian_id) \
            .execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        print(f"❌ Error getting guardian: {e}")
        return None

def log_activity(guardian_id=None, admin_username=None, action=None, details=None):
    """Log guardian or admin activity with debug output"""
    try:
        if not supabase:
            print("⚠️ [LOG_ACTIVITY] Supabase not initialized")
            return
            
        ip_address = request.remote_addr if request else None
        user_agent = request.headers.get('User-Agent') if request else None
        
        print(f"🔍 [LOG_ACTIVITY] Attempting to log: action={action}, guardian={guardian_id}, admin={admin_username}")
        
        if admin_username:
            data = {
                'admin_username': admin_username,
                'action': action,
                'details': details,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'timestamp': datetime.now().isoformat()
            }
            supabase.table('admin_activity_log').insert(data).execute()
            print(f"✅ [LOG_ACTIVITY] Admin log inserted for {admin_username}")
        else:
            data = {
                'guardian_id': guardian_id,
                'action': action,
                'details': details,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'timestamp': datetime.now().isoformat()
            }
            supabase.table('activity_log').insert(data).execute()
            print(f"✅ [LOG_ACTIVITY] Guardian log inserted for {guardian_id}")
    except Exception as e:
        print(f"⚠️ [LOG_ACTIVITY] Error: {e}")
        traceback.print_exc()

#end Region

#region Global Funct
def get_guardian_drivers(guardian_id):
    """Get all drivers registered by a guardian"""
    try:
        if not supabase:
            return []
        
        result = supabase.table('drivers') \
            .select('*') \
            .eq('guardian_id', guardian_id) \
            .eq('is_active', True) \
            .order('registration_date', desc=True) \
            .execute()
        
        drivers = result.data if result.data else []
        
        # Add alert counts and face counts
        for driver in drivers:
            # Alert count
            alert_result = supabase.table('alerts') \
                .select('count', count='exact') \
                .eq('driver_id', driver['driver_id']) \
                .eq('acknowledged', False) \
                .execute()
            driver['alert_count'] = alert_result.count if hasattr(alert_result, 'count') else 0
            
            # Face count
            face_result = supabase.table('face_images') \
                .select('count', count='exact') \
                .eq('driver_id', driver['driver_id']) \
                .execute()
            driver['face_count'] = face_result.count if hasattr(face_result, 'count') else 0
        
        return drivers
    except Exception as e:
        print(f"❌ Error in get_guardian_drivers: {e}")
        return []

def get_recent_alerts(guardian_id, limit=10):
    """Get recent alerts for a guardian"""
    try:
        if not supabase:
            return []
        
        result = supabase.table('alerts') \
            .select('*, drivers!inner(name)') \
            .eq('guardian_id', guardian_id) \
            .order('timestamp', desc=True) \
            .limit(limit) \
            .execute()
        
        alerts = result.data if result.data else []
        
        # Process detection details
        for alert in alerts:
            if alert.get('detection_details'):
                try:
                    alert['detection_details'] = json.loads(alert['detection_details'])
                except:
                    pass
            # Rename driver name field
            if 'drivers' in alert and alert['drivers']:
                alert['driver_name'] = alert['drivers']['name']
                del alert['drivers']
        
        return alerts
    except Exception as e:
        print(f"❌ Error in get_recent_alerts: {e}")
        return []

def get_driver_by_name_or_id(identifier):
    """Get driver by name or ID"""
    try:
        if not supabase:
            return None
        
        # Try by ID first
        result = supabase.table('drivers') \
            .select('*, guardians!inner(full_name, phone)') \
            .eq('driver_id', identifier) \
            .eq('is_active', True) \
            .execute()
        
        if result.data and len(result.data) > 0:
            driver = result.data[0]
            if 'guardians' in driver:
                driver['guardian_name'] = driver['guardians']['full_name']
                driver['guardian_phone'] = driver['guardians']['phone']
                del driver['guardians']
            return driver
        
        # If not found by ID, try by name
        result = supabase.table('drivers') \
            .select('*, guardians!inner(full_name, phone)') \
            .ilike('name', f'%{identifier}%') \
            .eq('is_active', True) \
            .limit(1) \
            .execute()
        
        if result.data and len(result.data) > 0:
            driver = result.data[0]
            if 'guardians' in driver:
                driver['guardian_name'] = driver['guardians']['full_name']
                driver['guardian_phone'] = driver['guardians']['phone']
                del driver['guardians']
            return driver
        
        return None
    except Exception as e:
        print(f"❌ Error getting driver: {e}")
        return None
#end Region

#region Security
@app.after_request

def add_cors_headers_for_socketio(response):
    """Ensure CORS headers are set for all responses, especially socket.io"""
    origin = request.headers.get('Origin')
    
    # Allow all your Firebase origins
    if origin and any(allowed in origin for allowed in ['guardian-drive-app.web.app', 'localhost']):
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    
    # Special handling for socket.io paths
    if request.path.startswith('/socket.io/'):
        # Ensure these headers are always present for socket.io
        if 'Access-Control-Allow-Origin' not in response.headers:
            response.headers['Access-Control-Allow-Origin'] = 'https://guardian-drive-app.web.app'
            response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response

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

@app.errorhandler(400)
def handle_bad_request(e):
   
    return jsonify({'error': 'Bad request', 'details': str(e)}), 400

#region Socket IO
@socketio.on('connect')
def handle_connect():
    """Handle client connection with better error tracking"""
    client_id = request.sid
    
    # Extract query parameters
    query_string = request.args.to_dict() if hasattr(request, 'args') else {}
    driver_id = query_string.get('driver_id')
    guardian_id = query_string.get('guardian_id')
    
    connected_clients[client_id] = {
        'connected_at': datetime.now(),
        'ip': request.remote_addr,
        'type': None,
        'guardian_id': guardian_id,
        'driver_id': driver_id,
        'authenticated': False,
        'last_ping': datetime.now(),
        'last_message': datetime.now(), 
        'last_location': None,
        'last_location_time': None,
        'location_count': 0, 
        'user_agent': request.headers.get('User-Agent', 'Unknown'),
        'origin': request.headers.get('Origin', 'Unknown'),
        'transport': request.environ.get('HTTP_UPGRADE', 'polling'),
        'query_params': query_string,
        'reconnect_count': 0  
    }
    
    print(f"\n{'='*60}")
    print(f"✅ WebSocket client connected: {client_id}")
    print(f"   IP: {request.remote_addr}")
    print(f"   Query params: {query_string}")
    if driver_id:
        print(f"   Driver ID from URL: {driver_id}")
    if guardian_id:
        print(f"   Guardian ID from URL: {guardian_id}")
    print(f"{'='*60}\n")
    
    emit('connected', {
        'status': 'connected',
        'client_id': client_id,
        'timestamp': datetime.now().isoformat(),
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    if client_id in connected_clients:
        client_info = connected_clients[client_id]
        
        print(f"\n⚠️ Client disconnected: {client_id}")
        print(f"   Type: {client_info.get('type')}")
        print(f"   Guardian ID: {client_info.get('guardian_id')}")
        
        # If it was a driver, notify its guardian
        if client_info.get('type') == 'driver':
            guardian_id = client_info.get('guardian_id')
            driver_id = client_info.get('driver_id')
            driver_name = client_info.get('driver_name')
            
            if guardian_id:
                socketio.emit('driver_disconnected', {
                    'driver_id': driver_id,
                    'driver_name': driver_name
                }, room=f'guardian_{guardian_id}')
                print(f"   Notified guardian {guardian_id}")
        
        # Clean up
        del connected_clients[client_id]
        
        # Leave rooms
        try:
            socketio.server.close_room(client_id)
        except:
            pass

@socketio.on('guardian_authenticate')
def handle_guardian_auth(data):
    """Guardian authentication via WebSocket - FIXED ROOM JOINING"""
    client_id = request.sid
    guardian_id = data.get('guardian_id')
    token = data.get('token')
    auth_provider = data.get('auth_provider', 'unknown')
    
    print(f"\n{'='*60}")
    print(f"🔐 GUARDIAN AUTHENTICATION ATTEMPT")
    print(f"{'='*60}")
    print(f"   Client ID: {client_id}")
    print(f"   Guardian ID: {guardian_id}")
    print(f"   Auth Provider: {auth_provider}")
    print(f"   Token present: {bool(token)}")
    
    if not guardian_id or not token:
        print(f"❌ Missing credentials from client {client_id}")
        emit('auth_failed', {'error': 'Missing guardian_id or token'})
        return
    
    # Validate session
    is_valid = validate_session(guardian_id, token)
    
    if is_valid:
        print(f"   ✅ Session validated successfully")
        
        # CRITICAL: Store in connected_clients FIRST
        connected_clients[client_id] = {
            'connected_at': datetime.now(),
            'ip': request.remote_addr,
            'type': 'guardian',
            'guardian_id': guardian_id,
            'authenticated': True,
            'auth_time': datetime.now(),
            'auth_provider': auth_provider,
            'last_ping': datetime.now(),
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'origin': request.headers.get('Origin', 'Unknown'),
            'transport': request.environ.get('HTTP_UPGRADE', 'polling')
        }
        
        print(f"   ✅ Client added to connected_clients: {client_id}")
        
        # CRITICAL: Join guardian room - MULTIPLE METHODS to ensure it works
        room_name = f"guardian_{guardian_id}"
        room_joined = False
        
        # Method 1: Using server.enter_room
        try:
            socketio.server.enter_room(client_id, room_name)
            print(f"   ✅ Method 1 - Joined room via server.enter_room: {room_name}")
            room_joined = True
        except Exception as e:
            print(f"   ⚠️ Method 1 failed: {e}")
        
        # Method 2: Using socketio.enter_room
        if not room_joined:
            try:
                socketio.enter_room(client_id, room_name)
                print(f"   ✅ Method 2 - Joined room via socketio.enter_room: {room_name}")
                room_joined = True
            except Exception as e2:
                print(f"   ⚠️ Method 2 failed: {e2}")
        
        # Method 3: Manual room management via emit with room parameter
        if not room_joined:
            try:
                # Force room creation by emitting to it
                socketio.emit('room_test', {'msg': 'testing'}, room=room_name)
                print(f"   ✅ Method 3 - Force-created room via emit: {room_name}")
                room_joined = True
            except Exception as e3:
                print(f"   ⚠️ Method 3 failed: {e3}")
        
        # Verify room membership
        try:
            rooms = list(socketio.rooms(client_id))
            print(f"   Client rooms: {rooms}")
            if room_name in rooms:
                print(f"   ✅ VERIFIED: Client is in room {room_name}")
            else:
                print(f"   ⚠️ Client NOT in room {room_name} - WILL RETRY")
                # One more attempt with a different approach
                socketio.server.manager.add_room(client_id, room_name, socketio.server.namespace)
                print(f"   ✅ Added room via manager: {room_name}")
        except Exception as e:
            print(f"   ⚠️ Could not verify rooms: {e}")
        
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
        else:
            print(f"✅ Guardian authenticated successfully (no details)")
            emit('auth_confirmed', {
                'success': True,
                'guardian_id': guardian_id,
                'timestamp': datetime.now().isoformat(),
                'message': 'WebSocket authentication successful'
            })
        
        print(f"   Current connected clients: {len(connected_clients)}")
        for cid, info in connected_clients.items():
            print(f"      {cid}: type={info.get('type')}, guardian={info.get('guardian_id')}")
        print(f"{'='*60}\n")
        return

    # Authentication failed
    print(f"❌ Session validation failed for guardian {guardian_id}")
    emit('auth_failed', {
        'error': 'Authentication failed',
        'timestamp': datetime.now().isoformat()
    })
    print(f"{'='*60}\n")
        
@socketio.on('driver_authenticate')
def handle_driver_auth(data):
    """Driver authentication via WebSocket - with reconnection tracking"""
    client_id = request.sid
    
    driver_id = data.get('driver_id')
    driver_name = data.get('driver_name', 'Unknown')
    guardian_id = data.get('guardian_id')
    
    print(f"\n{'='*60}")
    print(f"🔐 DRIVER AUTHENTICATION ATTEMPT")
    print(f"{'='*60}")
    print(f"   Client ID: {client_id}")
    print(f"   Driver ID: {driver_id}")
    print(f"   Driver Name: {driver_name}")

    if not driver_id:
        print("❌ Missing driver_id")
        emit('auth_failed', {'error': 'Missing driver_id'})
        return

    # Verify driver exists and get guardian_id
    if not guardian_id:
        guardian_id = get_guardian_for_driver(driver_id)
    
    if not guardian_id:
        print(f"❌ No guardian found for driver {driver_id}")
        emit('auth_failed', {'error': 'Driver not found or inactive'})
        return

    print(f"✅ Found guardian_id: {guardian_id} for driver {driver_id}")

    # Fetch driver name from database if needed
    if driver_name == 'Unknown' or not driver_name:
        try:
            if supabase:
                result = supabase.table('drivers') \
                    .select('name') \
                    .eq('driver_id', driver_id) \
                    .execute()
                if result.data and len(result.data) > 0:
                    driver_name = result.data[0]['name']
                    print(f"   Fetched driver name from DB: {driver_name}")
        except Exception as e:
            print(f"⚠️ Could not fetch driver name: {e}")

    # Check if this driver was previously connected (for reconnection tracking)
    old_client_id = None
    reconnect_count = 0
    for sid, info in connected_clients.items():
        if info.get('driver_id') == driver_id and sid != client_id:
            old_client_id = sid
            reconnect_count = info.get('reconnect_count', 0) + 1
            print(f"🔄 Driver {driver_name} reconnecting (was {old_client_id})")
            # Remove old connection
            try:
                socketio.server.disconnect(old_client_id, silent=True)
            except:
                pass
            del connected_clients[old_client_id]
            break

    # Update or create client info
    if client_id in connected_clients:
        connected_clients[client_id].update({
            'type': 'driver',
            'driver_id': driver_id,
            'guardian_id': guardian_id,
            'driver_name': driver_name,
            'authenticated': True,
            'auth_time': datetime.now(),
            'last_ping': datetime.now(),
            'last_message': datetime.now(),
            'reconnect_count': reconnect_count,
            'location_count': connected_clients[client_id].get('location_count', 0)
        })
        print(f"✅ Updated existing client info for {client_id}")
    else:
        connected_clients[client_id] = {
            'connected_at': datetime.now(),
            'ip': request.remote_addr,
            'type': 'driver',
            'guardian_id': guardian_id,
            'driver_id': driver_id,
            'driver_name': driver_name,
            'authenticated': True,
            'auth_time': datetime.now(),
            'last_ping': datetime.now(),
            'last_message': datetime.now(),
            'last_location': None,
            'last_location_time': None,
            'location_count': 0,
            'reconnect_count': reconnect_count,
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'origin': request.headers.get('Origin', 'Unknown'),
            'transport': request.environ.get('HTTP_UPGRADE', 'polling')
        }
        print(f"✅ Created new client entry for {client_id}")

    # Join rooms
    try:
        socketio.server.enter_room(client_id, f'driver_{driver_id}')
        socketio.server.enter_room(client_id, f'guardian_{guardian_id}')
        print(f"   ✅ Joined rooms: driver_{driver_id}, guardian_{guardian_id}")
    except Exception as e:
        print(f"   ⚠️ Error joining rooms: {e}")

    # Send confirmation
    response = {
        'success': True,
        'driver_id': driver_id,
        'guardian_id': guardian_id,
        'driver_name': driver_name,
        'reconnected': reconnect_count > 0,
        'reconnect_count': reconnect_count,
        'message': 'Driver authenticated successfully'
    }
    
    print(f"✅ Driver {driver_name} authenticated (reconnect #{reconnect_count})")
    emit('auth_confirmed', response)
    
    # Broadcast driver online
    try:
        socketio.emit('driver_online', {
            'driver_id': driver_id,
            'driver_name': driver_name,
            'guardian_id': guardian_id,
            'reconnected': reconnect_count > 0
        }, room=f'guardian_{guardian_id}')
        print(f"📢 Broadcast driver_online to guardian_{guardian_id}")
    except Exception as e:
        print(f"⚠️ Could not broadcast driver_online: {e}")
    
    print(f"{'='*60}\n")
    
@socketio.on('webrtc_offer')
def handle_webrtc_offer(data):
    """Forward WebRTC offer from guardian to driver"""
    client_id = request.sid
    client_info = connected_clients.get(client_id, {})
    
    # Guardian sending offer to driver
    driver_id = data.get('driver_id')
    if not driver_id:
        return
    
    # Find the driver's socket ID
    for cid, info in connected_clients.items():
        if info.get('type') == 'driver' and info.get('driver_id') == driver_id:
            socketio.emit('webrtc_offer', data, room=cid)
            return
    
@socketio.on('webrtc_answer')
def handle_webrtc_answer(data):
    """Forward WebRTC answer from driver to guardian"""
    client_id = request.sid
    client_info = connected_clients.get(client_id, {})
    
    # Driver answering guardian's offer
    guardian_id = data.get('guardian_id')
    if not guardian_id:
        return
    
    # Find the guardian's socket ID
    for cid, info in connected_clients.items():
        if info.get('type') == 'guardian' and str(info.get('guardian_id')) == str(guardian_id):
            socketio.emit('webrtc_answer', data, room=cid)
            return

@socketio.on('webrtc_ice_candidate')
def handle_webrtc_ice(data):
    """Forward ICE candidates between driver and guardian"""
    client_id = request.sid
    client_info = connected_clients.get(client_id, {})
    
    target = data.get('target')
    if target == 'guardian':
        # Driver sending ICE to guardian
        guardian_id = data.get('guardian_id') or client_info.get('guardian_id')
        if not guardian_id:
            return
        
        for cid, info in connected_clients.items():
            if info.get('type') == 'guardian' and str(info.get('guardian_id')) == str(guardian_id):
                socketio.emit('webrtc_ice_candidate', data, room=cid)
                return
                
    elif target == 'driver':
        # Guardian sending ICE to driver
        driver_id = data.get('driver_id')
        if not driver_id:
            return
            
        for cid, info in connected_clients.items():
            if info.get('type') == 'driver' and info.get('driver_id') == driver_id:
                socketio.emit('webrtc_ice_candidate', data, room=cid)
                return

@socketio.on('webrtc_ready')
def handle_webrtc_ready(data):
    """Handle driver ready signal and notify guardian"""
    client_id = request.sid
    client_info = connected_clients.get(client_id, {})
    
    print(f"\n📢 WEBRTC READY SIGNAL RECEIVED")
    print(f"{'='*60}")
    print(f"   From client: {client_id}")
    print(f"   Client type: {client_info.get('type')}")
    print(f"   Full data: {data}")
    
    if client_info.get('type') == 'driver':
        guardian_id = client_info.get('guardian_id')
        driver_id = client_info.get('driver_id')
        driver_name = client_info.get('driver_name', 'Unknown')
        
        print(f"   Driver: {driver_name} (ID: {driver_id})")
        print(f"   Notifying guardian: {guardian_id}")
        
        # Emit to guardian's room
        try:
            room_name = f"guardian_{guardian_id}"
            
            # First try room
            socketio.emit('webrtc_ready', {
                'driver_id': driver_id,
                'driver_name': driver_name,
                'guardian_id': guardian_id
            }, room=room_name)
            print(f"✅ Notification sent to room {room_name}")
            
            # Also send directly to any guardian connections
            for cid, info in connected_clients.items():
                if info.get('type') == 'guardian' and str(info.get('guardian_id')) == str(guardian_id):
                    print(f"   Also sending directly to guardian client: {cid}")
                    socketio.emit('webrtc_ready', {
                        'driver_id': driver_id,
                        'driver_name': driver_name,
                        'guardian_id': guardian_id
                    }, room=cid)
                    
        except Exception as e:
            print(f"⚠️ Error sending to guardian: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Not a driver client (type: {client_info.get('type')})")
    print(f"{'='*60}\n")

@socketio.on('webrtc_stop')
def handle_webrtc_stop(data):
    """Handle WebRTC stop signal from either side"""
    client_id = request.sid
    client_info = connected_clients.get(client_id, {})
    client_type = client_info.get('type')
    
    print(f"\n🛑 WebRTC STOP received from {client_type}: {client_id}")
    
    if client_type == 'guardian':
        driver_id = data.get('driver_id')
        if driver_id:
            socketio.emit('webrtc_stop', {
                'guardian_id': client_info.get('guardian_id')
            }, room=f'driver_{driver_id}')
            print(f"   Sent stop signal to driver {driver_id}")
            
    elif client_type == 'driver':
        guardian_id = client_info.get('guardian_id')
        if guardian_id:
            socketio.emit('webrtc_stop', {
                'driver_id': client_info.get('driver_id')
            }, room=f'guardian_{guardian_id}')
            print(f"   Sent stop signal to guardian {guardian_id}")

@socketio.on('driver_location')
def handle_driver_location(data):
    """Handle real-time location updates from drivers with better tracking"""
    client_id = request.sid
    client_info = connected_clients.get(client_id, {})
    
    try:
        driver_id = data.get('driver_id') or client_info.get('driver_id')
        driver_name = data.get('driver_name') or client_info.get('driver_name', 'Unknown')
        guardian_id = data.get('guardian_id') or client_info.get('guardian_id')
        location = data.get('location')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        drowsiness = data.get('drowsiness', 0)
        drowsiness_level = data.get('drowsiness_level', 'ALERT')
        
        if not driver_id or not guardian_id or not location:
            print(f"⚠️ [SOCKET] Missing location data from driver {driver_id}")
            return
        
        # Update client info with location and timestamp
        if client_id in connected_clients:
            connected_clients[client_id]['last_location'] = location
            connected_clients[client_id]['last_location_time'] = timestamp
            connected_clients[client_id]['drowsiness'] = drowsiness
            connected_clients[client_id]['drowsiness_level'] = drowsiness_level
            connected_clients[client_id]['last_message'] = datetime.now()
            connected_clients[client_id]['location_count'] = connected_clients[client_id].get('location_count', 0) + 1
        
        print(f"\n📍 [SOCKET] Location update #{connected_clients[client_id].get('location_count', 0)} from {driver_name}")
        print(f"   Location: {location.get('lat')}, {location.get('lng')}")
        
        # Store in database
        if supabase:
            try:
                location_data = {
                    'driver_id': driver_id,
                    'guardian_id': guardian_id,
                    'latitude': location.get('lat'),
                    'longitude': location.get('lng'),
                    'timestamp': timestamp,
                    'drowsiness': drowsiness,
                    'drowsiness_level': drowsiness_level,
                    'accuracy': location.get('accuracy', 0),
                    'method': location.get('method', 'gps')
                }
                
                supabase.table('driver_locations').insert(location_data).execute()
            except Exception as db_error:
                print(f"⚠️ [SOCKET] Database error: {db_error}")
        
        # Forward to guardian
        guardian_update = {
            'driver_id': driver_id,
            'driver_name': driver_name,
            'guardian_id': guardian_id,
            'location': location,
            'timestamp': timestamp,
            'drowsiness': drowsiness,
            'drowsiness_level': drowsiness_level
        }
        
        socketio.emit('driver_location_update', guardian_update, room=f'guardian_{guardian_id}')
        
    except Exception as e:
        print(f"❌ [SOCKET] Error handling driver location: {e}")
        import traceback
        traceback.print_exc()

@socketio.on('driver_alert')
def handle_driver_alert(data):
    """Receive a drowsiness alert from a driver, save it, and forward to guardian."""
    client_id = request.sid
    client_info = connected_clients.get(client_id, {})
    driver_id = client_info.get('driver_id') or data.get('driver_id')

    if not driver_id:
        emit('error', {'error': 'Unknown driver'})
        return

    guardian_id = client_info.get('guardian_id') or get_guardian_for_driver(driver_id)
    if not guardian_id:
        emit('error', {'error': 'Cannot determine guardian'})
        return

    alert_data = {
        'driver_id': driver_id,
        'driver_name': data.get('driver_name', 'Unknown'),
        'guardian_id': guardian_id,
        'severity': data.get('severity', 'high'),
        'message': data.get('message', 'Drowsiness detected'),
        'confidence': data.get('confidence', 0.0),
        'timestamp': datetime.now().isoformat(),
        'detection_details': data.get('detection_details', {})
    }

    # Save to database
    try:
        if supabase:
            alert_insert = {
                'driver_id': driver_id,
                'guardian_id': guardian_id,
                'severity': alert_data['severity'],
                'message': alert_data['message'],
                'detection_details': json.dumps(alert_data['detection_details']),
                'source': 'drowsiness_detection',
                'timestamp': datetime.now().isoformat()
            }
            supabase.table('alerts').insert(alert_insert).execute()
    except Exception as e:
        print(f"❌ Error saving alert: {e}")

    # Forward to guardian
    socketio.emit('guardian_alert', alert_data, room=f'guardian_{guardian_id}')
    print(f"🚨 Alert forwarded to guardian {guardian_id}")

@socketio.on('ping')
def handle_ping():
    """Handle ping from client to keep connection alive"""
    client_id = request.sid
    if client_id in connected_clients:
        connected_clients[client_id]['last_ping'] = datetime.now()
        emit('pong', {'timestamp': datetime.now().isoformat()})

@socketio.on('error')
def handle_error(error):
    """Handle socket errors"""
    client_id = request.sid
    print(f"❌ Socket error for client {client_id}: {error}")

#end Region

#region Application
@app.route('/')
def serve_home():
    """Redirect to Firebase Hosting - pure backend server"""
    return jsonify({
        'success': True,
        'message': 'Driver Alert System API Server',
        'backend': 'Render.com with Supabase',
        'database': 'Supabase PostgreSQL',
        'frontend': 'Firebase Hosting',
        'frontend_url': 'https://guardian-drive-app.web.app',
        'api_docs': 'https://driver-drowsiness-with-alert.onrender.com/api/health',
        'version': '2.0.0',
        'cloudinary_enabled': CLOUDINARY_ENABLED
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with face recognition status"""
    try:
        # Test Supabase connection
        supabase_connected = False
        drivers_with_embeddings = 0
        
        if supabase:
            try:
                result = supabase.table('guardians').select('*').limit(1).execute()
                supabase_connected = True
                
                # Count drivers with face embeddings
                embed_result = supabase.table('drivers') \
                    .select('count', count='exact') \
                    .not_.is_('face_embedding', 'null') \
                    .execute()
                drivers_with_embeddings = embed_result.count if hasattr(embed_result, 'count') else 0
            except Exception as e:
                print(f"⚠️ Supabase connection test failed: {e}")
        
        health_info = {
            'success': True,
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'server': 'Driver Alert System API',
            'version': '2.1.0',  # Updated version
            'connected_clients': len(connected_clients),
            'active_sessions': len(active_sessions),
            'database': 'supabase',
            'supabase_connected': supabase_connected,
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
        
#end Region

# =================== WEBSOCKET CHECKER ======================
@app.route('/api/websocket-status', methods=['GET'])
def websocket_status():
    """Check WebSocket server status with face recognition info"""
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
                    'auth_provider': info.get('auth_provider', 'unknown'),
                    'client_type': info.get('type', 'unknown') 
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
        'facebook_app_id': FACEBOOK_APP_ID,
        'facebook_app_secret': FACEBOOK_APP_SECRET,
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

#region Gmail Auth
@app.route('/api/google-login', methods=['POST'])
def google_login():
    """Handle Google OAuth login - using hybrid verification approach"""
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
            # Use hybrid verification (Google library + custom HTTP + fallback)
            idinfo = verify_google_token_hybrid(google_token, GOOGLE_CLIENT_ID)
            
            email = idinfo.get('email')
            name = idinfo.get('name', '')
            google_id = idinfo.get('sub')
            
            print(f"✅ Google token verified successfully: {email}")
            print(f"   Name: {name}")
            
        except Exception as e:
            print(f"❌ Google token verification failed: {e}")
            return jsonify({
                'success': False,
                'error': f'Token verification failed: {str(e)}'
            }), 401
        
        if not email:
            return jsonify({
                'success': False,
                'error': 'Email not found in token'
            }), 400
        
        # Database operations with Supabase
        try:
            if not supabase:
                return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
            
            # Check if user exists by Google ID or email
            result = supabase.table('guardians') \
                .select('*') \
                .or_(f'google_id.eq.{google_id},email.eq.{email}') \
                .limit(1) \
                .execute()
            
            if result.data and len(result.data) > 0:
                # User exists - login
                user_data = result.data[0]
                guardian_id = user_data['guardian_id']
                full_name = user_data['full_name']
                stored_email = user_data['email']
                stored_google_id = user_data.get('google_id')
                
                print(f"✅ Existing user found: {full_name} ({stored_email})")
                
                # Update Google ID if not set
                if not stored_google_id and google_id:
                    supabase.table('guardians') \
                        .update({'google_id': google_id}) \
                        .eq('guardian_id', guardian_id) \
                        .execute()
                    print(f"   Updated Google ID for user: {google_id}")
                
                # Update last login
                supabase.table('guardians') \
                    .update({'last_login': datetime.now().isoformat()}) \
                    .eq('guardian_id', guardian_id) \
                    .execute()
                
            else:
                # Create new user with Google data
                import random
                import secrets
                
                # Generate a unique phone number
                phone = '09' + ''.join([str(random.randint(0, 9)) for _ in range(9)])
                
                # Generate a random password hash
                temp_password = secrets.token_urlsafe(16)
                password_hash = hash_password(temp_password)
                
                new_user = {
                    'full_name': name,
                    'email': email,
                    'phone': phone,
                    'password_hash': password_hash,
                    'is_active': True,
                    'registration_date': datetime.now().isoformat(),
                    'last_login': datetime.now().isoformat(),
                    'google_id': google_id,
                    'auth_provider': 'google'
                }
                
                insert_result = supabase.table('guardians').insert(new_user).execute()
                
                if insert_result.data and len(insert_result.data) > 0:
                    guardian_id = insert_result.data[0]['guardian_id']
                else:
                    raise Exception("Failed to create user")
                
                full_name = name
                
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

# ==================== OTHER GOOGLE LOGIN ENDPOINTS (keep as fallback) ====================
@app.route('/api/google-login-direct', methods=['POST'])
def google_login_direct():
    """Direct Google login that bypasses token verification"""
    try:
        data = request.json
        email = data.get('email')
        name = data.get('name', email.split('@')[0] if email else 'Google User')
        
        if not email:
            return jsonify({'success': False, 'error': 'Email required'}), 400
        
        print(f"🔐 Google login direct - Email: {email}")
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Check if user exists
        result = supabase.table('guardians') \
            .select('guardian_id, full_name') \
            .eq('email', email) \
            .execute()
        
        if result.data and len(result.data) > 0:
            user_data = result.data[0]
            guardian_id = user_data['guardian_id']
            full_name = user_data['full_name']
            print(f"✅ Existing user: {full_name}")
            
            # Update last login
            supabase.table('guardians') \
                .update({'last_login': datetime.now().isoformat()}) \
                .eq('guardian_id', guardian_id) \
                .execute()
        else:
            # Create new user
            import random, secrets
            phone = '09' + ''.join([str(random.randint(0, 9)) for _ in range(9)])
            temp_password = secrets.token_urlsafe(16)
            
            new_user = {
                'full_name': name,
                'email': email,
                'phone': phone,
                'password_hash': hash_password(temp_password),
                'auth_provider': 'google',
                'registration_date': datetime.now().isoformat(),
                'last_login': datetime.now().isoformat()
            }
            
            insert_result = supabase.table('guardians').insert(new_user).execute()
            
            if insert_result.data and len(insert_result.data) > 0:
                guardian_id = insert_result.data[0]['guardian_id']
            else:
                raise Exception("Failed to create user")
            
            full_name = name
            print(f"✅ New user created: {full_name}")
        
        # Create session
        token = create_session(guardian_id, request.remote_addr, request.headers.get('User-Agent'))
        
        log_activity(guardian_id, 'GOOGLE_LOGIN', f'Google direct login from {request.remote_addr}')
        
        return jsonify({
            'success': True,
            'guardian_id': guardian_id,
            'full_name': full_name,
            'email': email,
            'session_token': token,
            'auth_provider': 'google',
            'message': 'Google login successful'
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

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
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Check if user exists by email
        result = supabase.table('guardians') \
            .select('*') \
            .eq('email', email) \
            .limit(1) \
            .execute()
        
        if result.data and len(result.data) > 0:
            # User exists - login
            user_data = result.data[0]
            guardian_id = user_data['guardian_id']
            full_name = user_data['full_name']
            stored_email = user_data['email']
            auth_provider = user_data.get('auth_provider')
            
            # Update auth provider if not google
            if auth_provider != 'google':
                supabase.table('guardians') \
                    .update({'auth_provider': 'google'}) \
                    .eq('guardian_id', guardian_id) \
                    .execute()
            
            print(f"✅ User found: {full_name} ({stored_email})")
            
        else:
            # Create new user
            import random
            phone = '09' + ''.join([str(random.randint(0, 9)) for _ in range(9)])
            
            new_user = {
                'full_name': name,
                'email': email,
                'phone': phone,
                'password_hash': hash_password(secrets.token_urlsafe(16)),
                'is_active': True,
                'registration_date': datetime.now().isoformat(),
                'last_login': datetime.now().isoformat(),
                'google_id': google_id,
                'auth_provider': 'google'
            }
            
            insert_result = supabase.table('guardians').insert(new_user).execute()
            
            if insert_result.data and len(insert_result.data) > 0:
                guardian_id = insert_result.data[0]['guardian_id']
            else:
                raise Exception("Failed to create user")
            
            full_name = name
            
            print(f"✅ New user created: {full_name} (ID: {guardian_id})")
        
        token = create_session(guardian_id, request.remote_addr, request.headers.get('User-Agent'))
       
        log_activity(guardian_id, 'GOOGLE_LOGIN', f'Google email login from {request.remote_addr}')
        
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
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Check if user exists by email
        result = supabase.table('guardians') \
            .select('guardian_id, full_name, email') \
            .eq('email', email) \
            .limit(1) \
            .execute()
        
        if result.data and len(result.data) > 0:
            user_data = result.data[0]
            guardian_id = user_data['guardian_id']
            full_name = user_data['full_name']
        else:
            # Create new user
            import random
            phone = '09' + ''.join([str(random.randint(0, 9)) for _ in range(9)])
            
            new_user = {
                'full_name': name,
                'email': email,
                'phone': phone,
                'password_hash': hash_password(secrets.token_urlsafe(16)),
                'is_active': True,
                'google_id': google_id,
                'auth_provider': 'google',
                'registration_date': datetime.now().isoformat(),
                'last_login': datetime.now().isoformat()
            }
            
            insert_result = supabase.table('guardians').insert(new_user).execute()
            
            if insert_result.data and len(insert_result.data) > 0:
                guardian_id = insert_result.data[0]['guardian_id']
            else:
                raise Exception("Failed to create user")
            
            full_name = name
        
        token = create_session(guardian_id, request.remote_addr, request.headers.get('User-Agent'))
       
        log_activity(guardian_id, 'GOOGLE_LOGIN', f'Google simple login from {request.remote_addr}')
        
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

#end Region Gmail Auth

@app.route('/api/debug/driver-status/<driver_id>', methods=['GET'])
def debug_driver_status(driver_id):
    """Check if a specific driver is connected and sending data"""
    try:
        driver_info = None
        location_count = 0
        
        # Check connected clients
        for sid, info in connected_clients.items():
            if info.get('driver_id') == driver_id:
                driver_info = {
                    'sid': sid,
                    'driver_name': info.get('driver_name'),
                    'guardian_id': info.get('guardian_id'),
                    'authenticated': info.get('authenticated'),
                    'connected_at': info.get('connected_at').isoformat() if info.get('connected_at') else None,
                    'last_ping': info.get('last_ping').isoformat() if info.get('last_ping') else None,
                    'last_location': info.get('last_location'),
                    'last_location_time': info.get('last_location_time')
                }
                break
        
        # Check database for recent locations
        if supabase:
            result = supabase.table('driver_locations') \
                .select('*') \
                .eq('driver_id', driver_id) \
                .order('timestamp', desc=True) \
                .limit(5) \
                .execute()
            recent_locations = result.data if result.data else []
            location_count = len(recent_locations)
        else:
            recent_locations = []
        
        return jsonify({
            'success': True,
            'driver_id': driver_id,
            'connected': driver_info is not None,
            'driver_info': driver_info,
            'recent_locations_count': location_count,
            'recent_locations': recent_locations,
            'server_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debug/guardian-room/<guardian_id>', methods=['GET'])
def debug_guardian_room(guardian_id):
    """Check if guardian is in the correct room"""
    clients_in_room = []
    for sid, info in connected_clients.items():
        if info.get('guardian_id') == str(guardian_id) and info.get('type') == 'guardian':
            clients_in_room.append({
                'sid': sid,
                'authenticated': info.get('authenticated'),
                'connected_at': info.get('connected_at').isoformat() if info.get('connected_at') else None
            })
    
    return jsonify({
        'success': True,
        'guardian_id': guardian_id,
        'clients_in_room': clients_in_room,
        'count': len(clients_in_room)
    })
    
@app.route('/api/debug-google-certs', methods=['GET'])
def debug_google_certs():
    """Debug endpoint to check Google certificates"""
    try:
        result = {
            'google_client_id': GOOGLE_CLIENT_ID,
            'google_client_id_valid': bool(GOOGLE_CLIENT_ID and 'googleusercontent.com' in GOOGLE_CLIENT_ID),
            'certs_cache_exists': _certs_cache is not None,
            'certs_cache_age': time.time() - _certs_cache_time if _certs_cache_time else None,
            'certs_count': len(_certs_cache) if _certs_cache else 0,
        }
        
        # Test certificate fetch
        try:
            certs = get_certs_with_retry(force_refresh=True)
            result['fetch_success'] = True
            result['certs_count_after_fetch'] = len(certs)
        except Exception as e:
            result['fetch_success'] = False
            result['fetch_error'] = str(e)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#region Facebook Auth
# ==================== FACEBOOK AUTH ====================
@app.route('/api/facebook-login', methods=['POST'])
def facebook_login():
    """Handle Facebook OAuth login - Only stores name, email, and phone"""
    try:
        data = request.json
        access_token = data.get('access_token')
        user_id = data.get('user_id')
        
        if not access_token or not user_id:
            return jsonify({
                'success': False,
                'error': 'Facebook access token and user ID required'
            }), 400
        
        print(f"🔵 [FACEBOOK LOGIN] Received Facebook token for user: {user_id}")
        
        # Verify token with Facebook
        if not FACEBOOK_APP_ID or not FACEBOOK_APP_SECRET:
            print(f"❌ [FACEBOOK LOGIN] Facebook credentials not set")
            return jsonify({
                'success': False,
                'error': 'Facebook authentication not configured on server'
            }), 500
        
        # Verify the access token with Facebook
        try:
            # Use our custom session with retries
            verification_url = f"https://graph.facebook.com/debug_token"
            params = {
                'input_token': access_token,
                'access_token': f"{FACEBOOK_APP_ID}|{FACEBOOK_APP_SECRET}"
            }
            
            verify_response = google_session.get(verification_url, params=params, timeout=10)
            verify_response.raise_for_status()
            token_data = verify_response.json()
            
            if not token_data.get('data', {}).get('is_valid'):
                error_msg = token_data.get('data', {}).get('error', {}).get('message', 'Invalid token')
                return jsonify({
                    'success': False,
                    'error': f'Invalid Facebook token: {error_msg}'
                }), 401
            
            # Get user info from Facebook - ONLY requesting name and email
            user_info_url = "https://graph.facebook.com/me"
            user_params = {
                'fields': 'id,name,email',  # REMOVED 'picture' - only name and email
                'access_token': access_token
            }
            
            user_response = google_session.get(user_info_url, params=user_params, timeout=10)
            user_response.raise_for_status()
            fb_user = user_response.json()
            
            email = fb_user.get('email')
            name = fb_user.get('name', 'Facebook User')
            facebook_id = fb_user.get('id')
            
            if not email:
                # Facebook may not provide email - this is common
                print(f"⚠️ [FACEBOOK LOGIN] No email provided for user {facebook_id}")
                # We'll continue without email - user can add it later
                email = None
            
            print(f"✅ Facebook token verified successfully: {name} (ID: {facebook_id})")
            
        except Exception as e:
            print(f"❌ Facebook token verification failed: {e}")
            return jsonify({
                'success': False,
                'error': f'Token verification failed: {str(e)}'
            }), 401
        
        # Database operations with Supabase
        try:
            if not supabase:
                return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
            
            # Check if user exists by Facebook ID
            result = supabase.table('guardians') \
                .select('*') \
                .eq('facebook_id', facebook_id) \
                .limit(1) \
                .execute()
            
            if result.data and len(result.data) > 0:
                # User exists - login
                user_data = result.data[0]
                guardian_id = user_data['guardian_id']
                full_name = user_data['full_name']
                stored_email = user_data.get('email')
                
                print(f"✅ Existing user found: {full_name}")
                
                # Update email if it was missing and now provided
                if email and not stored_email:
                    supabase.table('guardians') \
                        .update({'email': email}) \
                        .eq('guardian_id', guardian_id) \
                        .execute()
                    print(f"   Updated email for user: {email}")
                
                # Update last login
                supabase.table('guardians') \
                    .update({'last_login': datetime.now().isoformat()}) \
                    .eq('guardian_id', guardian_id) \
                    .execute()
                
            else:
                # Check if user exists by email (if email provided)
                existing_by_email = None
                if email:
                    email_result = supabase.table('guardians') \
                        .select('*') \
                        .eq('email', email) \
                        .limit(1) \
                        .execute()
                    
                    if email_result.data and len(email_result.data) > 0:
                        existing_by_email = email_result.data[0]
                
                if existing_by_email:
                    # Link Facebook ID to existing account
                    guardian_id = existing_by_email['guardian_id']
                    full_name = existing_by_email['full_name']
                    
                    # Update with Facebook ID
                    supabase.table('guardians') \
                        .update({
                            'facebook_id': facebook_id,
                            'auth_provider': 'facebook',
                            'last_login': datetime.now().isoformat()
                        }) \
                        .eq('guardian_id', guardian_id) \
                        .execute()
                    
                    print(f"✅ Linked Facebook to existing user: {full_name}")
                    
                else:
                    # Create new user with Facebook data - ONLY name, email, and phone
                    import random
                    
                    # Generate a unique phone number (required field)
                    phone = None
                    max_attempts = 10
                    for attempt in range(max_attempts):
                        test_phone = '09' + ''.join([str(random.randint(0, 9)) for _ in range(9)])
                        # Check if phone exists
                        phone_check = supabase.table('guardians') \
                            .select('guardian_id') \
                            .eq('phone', test_phone) \
                            .execute()
                        if not phone_check.data or len(phone_check.data) == 0:
                            phone = test_phone
                            break
                    
                    if not phone:
                        # Fallback - use timestamp to ensure uniqueness
                        import time
                        phone = f"09{int(time.time())%1000000000:09d}"
                    
                    # Generate a random password hash (required field)
                    import secrets
                    temp_password = secrets.token_urlsafe(16)
                    password_hash = hash_password(temp_password)
                    
                    new_user = {
                        'full_name': name,
                        'phone': phone,  # Auto-generated phone
                        'password_hash': password_hash,  # Required field
                        'is_active': True,
                        'registration_date': datetime.now().isoformat(),
                        'last_login': datetime.now().isoformat(),
                        'facebook_id': facebook_id,
                        'auth_provider': 'facebook'
                    }
                    
                    # Only add email if provided by Facebook
                    if email:
                        new_user['email'] = email
                    
                    insert_result = supabase.table('guardians').insert(new_user).execute()
                    
                    if insert_result.data and len(insert_result.data) > 0:
                        guardian_id = insert_result.data[0]['guardian_id']
                    else:
                        raise Exception("Failed to create user")
                    
                    full_name = name
                    
                    print(f"✅ New Facebook user created: {full_name} (ID: {guardian_id})")
                    if email:
                        print(f"   Email: {email}")
                    print(f"   Phone: {phone} (auto-generated)")
            
            # Create session
            token = create_session(guardian_id, request.remote_addr, request.headers.get('User-Agent'))
            
            # Log activity
            log_activity(guardian_id, 'FACEBOOK_LOGIN', f'Facebook login from {request.remote_addr}')
            
            # Get user data for response
            user_result = supabase.table('guardians') \
                .select('guardian_id, full_name, email, phone') \
                .eq('guardian_id', guardian_id) \
                .execute()
            
            user_details = user_result.data[0] if user_result.data else {}
            
            return jsonify({
                'success': True,
                'guardian_id': guardian_id,
                'full_name': full_name,
                'email': user_details.get('email', ''),
                'phone': user_details.get('phone', ''),
                'session_token': token,
                'auth_provider': 'facebook',
                'message': 'Facebook login successful',
                'is_facebook_user': True,
                'redirect_url': f'https://guardian-drive-app.web.app/guardian-dashboard.html?guardian_id={guardian_id}&token={token}'
            })
            
        except Exception as db_error:
            print(f"❌ Database error in facebook_login: {db_error}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Database error: {str(db_error)}'
            }), 500
        
    except Exception as e:
        print(f"❌ Facebook login error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Facebook login failed: {str(e)}'
        }), 500


@app.route('/api/facebook-login-simple', methods=['POST'])
def facebook_login_simple():
    """Simplified Facebook login for when email isn't available"""
    try:
        data = request.json
        facebook_id = data.get('facebook_id')
        name = data.get('name', 'Facebook User')
        email = data.get('email')  # May be None
        
        if not facebook_id:
            return jsonify({'success': False, 'error': 'Facebook ID required'}), 400
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Check if user exists by Facebook ID
        result = supabase.table('guardians') \
            .select('guardian_id, full_name, email, phone') \
            .eq('facebook_id', facebook_id) \
            .limit(1) \
            .execute()
        
        if result.data and len(result.data) > 0:
            user_data = result.data[0]
            guardian_id = user_data['guardian_id']
            full_name = user_data['full_name']
            stored_email = user_data.get('email')
            stored_phone = user_data.get('phone')
            
            # Update email if provided and not stored
            if email and not stored_email:
                supabase.table('guardians') \
                    .update({'email': email}) \
                    .eq('guardian_id', guardian_id) \
                    .execute()
                print(f"   Updated email for user: {email}")
            
        else:
            # Check if user exists by email (if email provided)
            existing_by_email = None
            if email:
                email_result = supabase.table('guardians') \
                    .select('*') \
                    .eq('email', email) \
                    .limit(1) \
                    .execute()
                
                if email_result.data and len(email_result.data) > 0:
                    existing_by_email = email_result.data[0]
            
            if existing_by_email:
                # Link Facebook ID to existing account
                guardian_id = existing_by_email['guardian_id']
                full_name = existing_by_email['full_name']
                
                supabase.table('guardians') \
                    .update({
                        'facebook_id': facebook_id,
                        'auth_provider': 'facebook',
                        'last_login': datetime.now().isoformat()
                    }) \
                    .eq('guardian_id', guardian_id) \
                    .execute()
                
                stored_phone = existing_by_email.get('phone')
                
            else:
                # Create new user
                import random
                import secrets
                
                # Generate unique phone
                phone = None
                max_attempts = 10
                for attempt in range(max_attempts):
                    test_phone = '09' + ''.join([str(random.randint(0, 9)) for _ in range(9)])
                    phone_check = supabase.table('guardians') \
                        .select('guardian_id') \
                        .eq('phone', test_phone) \
                        .execute()
                    if not phone_check.data or len(phone_check.data) == 0:
                        phone = test_phone
                        break
                
                if not phone:
                    import time
                    phone = f"09{int(time.time())%1000000000:09d}"
                
                new_user = {
                    'full_name': name,
                    'phone': phone,
                    'password_hash': hash_password(secrets.token_urlsafe(16)),
                    'is_active': True,
                    'facebook_id': facebook_id,
                    'auth_provider': 'facebook',
                    'registration_date': datetime.now().isoformat(),
                    'last_login': datetime.now().isoformat()
                }
                
                # Only add email if provided
                if email:
                    new_user['email'] = email
                
                insert_result = supabase.table('guardians').insert(new_user).execute()
                
                if insert_result.data and len(insert_result.data) > 0:
                    guardian_id = insert_result.data[0]['guardian_id']
                else:
                    raise Exception("Failed to create user")
                
                full_name = name
                stored_phone = phone
        
        # Create session
        token = create_session(guardian_id, request.remote_addr, request.headers.get('User-Agent'))
        
        # Log activity
        log_activity(guardian_id, 'FACEBOOK_LOGIN', f'Facebook simple login from {request.remote_addr}')
        
        # Get updated user data
        user_result = supabase.table('guardians') \
            .select('guardian_id, full_name, email, phone') \
            .eq('guardian_id', guardian_id) \
            .execute()
        
        user_details = user_result.data[0] if user_result.data else {}
        
        return jsonify({
            'success': True,
            'guardian_id': guardian_id,
            'full_name': full_name,
            'email': user_details.get('email', ''),
            'phone': user_details.get('phone', ''),
            'session_token': token,
            'auth_provider': 'facebook',
            'message': 'Facebook login successful',
            'redirect_url': f'https://guardian-drive-app.web.app/guardian-dashboard.html?guardian_id={guardian_id}&token={token}'
        })
        
    except Exception as e:
        print(f"❌ Facebook simple login error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
#end Region

#region Guardian Auth
@app.route('/api/login', methods=['POST'])
def login():
    """Guardian login with email/phone + password"""
    try:
        data = request.json
        identifier = data.get('identifier', '').strip()  
        password = data.get('password', '')


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
            if supabase:
                user_result = supabase.table('guardians') \
                    .select('guardian_id, full_name, email, phone') \
                    .eq('guardian_id', guardian['guardian_id']) \
                    .execute()
                
                if user_result.data and len(user_result.data) > 0:
                    user_details = user_result.data[0]
                else:
                    user_details = {}
            else:
                user_details = {}

            return jsonify({
                'success': True,
                'guardian_id': guardian['guardian_id'],
                'full_name': guardian['full_name'],
                'email': user_details.get('email', ''),
                'phone': user_details.get('phone', ''),
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
                'exact_value': os.environ.get('GOOGLE_CLIENT_ID', 'NOT SET')  
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
            'supabase': {
                'configured': bool(SUPABASE_URL and SUPABASE_KEY),
                'url_configured': bool(SUPABASE_URL),
                'key_configured': bool(SUPABASE_KEY)
            },
            'environment': {
                'render': bool(os.environ.get('RENDER')),
                'render_service': os.environ.get('RENDER_SERVICE_NAME', 'unknown'),
                'database_configured': bool(os.environ.get('DATABASE_URL') or (SUPABASE_URL and SUPABASE_KEY))
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
                results['dns_resolution'].append("Google APIs are reachable")
            else:
                results['google_apis_reachable'] = False
                results['dns_resolution'].append(f"Google APIs returned status {response.status_code}")
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
        
        # Test Supabase connection (basic)
        supabase_connected = False
        guardian_count = 0
        if supabase:
            try:
                result = supabase.table('guardians').select('*', count='exact').limit(1).execute()
                supabase_connected = True
                guardian_count = result.count if hasattr(result, 'count') else 0
            except Exception as db_error:
                supabase_connected = False
                results['common_issues'].append(f"Supabase connection issue: {str(db_error)}")
        
        results['supabase'] = {
            'connected': supabase_connected,
            'guardian_count': guardian_count
        }
        
        # Provide overall assessment
        if len(results['common_issues']) == 0:
            results['assessment'] = "✅ All checks passed! Google Auth should work properly."
            results['status_emoji'] = "✅"
        else:
            results['assessment'] = f"⚠️ Found {len(results['common_issues'])} potential issue(s):"
            for issue in results['common_issues']:
                results['assessment'] += f"\n   • {issue}"
            results['status_emoji'] = "⚠️"
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500
        
@app.route('/api/guardian/<guardian_id>/active-streams', methods=['GET'])
def get_active_streams(guardian_id):
    """Get list of active streams/drivers for a guardian - MORE LENIENT"""
    try:
        token = request.args.get('token')
        
        if not token:
            return jsonify({'success': False, 'error': 'Authentication required'}), 401
        
        if not validate_session(guardian_id, token):
            return jsonify({'success': False, 'error': 'Invalid session'}), 401
        
        print(f"\n🔍 Active streams check for guardian {guardian_id}")
        print(f"   Total connected clients: {len(connected_clients)}")
        
        active_streams = []
        current_time = datetime.now()
        
        for client_id, info in connected_clients.items():
            client_type = info.get('type')
            client_guardian = info.get('guardian_id')
            client_auth = info.get('authenticated')
            
            if (client_type == 'driver' and 
                str(client_guardian) == str(guardian_id) and 
                client_auth == True):
                
                connected_at = info.get('connected_at')
                last_message = info.get('last_message', connected_at)
                last_ping = info.get('last_ping')
                last_location = info.get('last_location')
                last_location_time = info.get('last_location_time')
                location_count = info.get('location_count', 0)
                
                # Calculate times
                seconds_since_message = 0
                if last_message:
                    if isinstance(last_message, datetime):
                        seconds_since_message = (current_time - last_message).total_seconds()
                
                # MORE LENIENT: Consider stable if message received in last 30 seconds
                is_stable = seconds_since_message < 30
                
                print(f"   Driver {info.get('driver_name')}: {seconds_since_message:.1f}s since last message, locations: {location_count}")
                
                active_streams.append({
                    'driver_id': info.get('driver_id'),
                    'driver_name': info.get('driver_name', 'Unknown'),
                    'connected_at': connected_at.isoformat() if connected_at else None,
                    'connected_seconds': int((current_time - connected_at).total_seconds()) if connected_at else 0,
                    'seconds_since_message': int(seconds_since_message),
                    'seconds_since_ping': int((current_time - last_ping).total_seconds()) if last_ping else 999,
                    'is_stable': is_stable,
                    'has_location': last_location is not None,
                    'location_count': location_count,
                    'last_location_time': last_location_time,
                    'status': 'connected',
                    'client_id': client_id,
                    'transport': info.get('transport', 'unknown')
                })
        
        print(f"   Found {len(active_streams)} active streams for guardian {guardian_id}")
        
        return jsonify({
            'success': True,
            'guardian_id': guardian_id,
            'count': len(active_streams),
            'streams': active_streams
        })
        
    except Exception as e:
        print(f"❌ Error in get_active_streams: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
        
#end Region

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
        
        if not phone_clean.isdigit():
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
        
        # Supabase operations
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Check if phone already exists
        phone_check = supabase.table('guardians') \
            .select('guardian_id') \
            .eq('phone', final_phone) \
            .execute()
        
        if phone_check.data and len(phone_check.data) > 0:
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
        
        # Insert into Supabase
        new_guardian = {
            'full_name': full_name,
            'phone': final_phone,
            'email': data.get('email', ''),
            'password_hash': password_hash,
            'address': data.get('address', ''),
            'registration_date': datetime.now().isoformat(),
            'last_login': datetime.now().isoformat(),
            'is_active': True,
            'auth_provider': 'phone'
        }
        
        insert_result = supabase.table('guardians').insert(new_guardian).execute()
        
        if insert_result.data and len(insert_result.data) > 0:
            guardian_id = insert_result.data[0]['guardian_id']
            print(f"✅ [REGISTRATION] Database record created with guardian_id: {guardian_id}")
        else:
            raise Exception("Failed to insert guardian")
        
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

#region Guardian Data
@app.route('/api/guardian/dashboard', methods=['GET'])
def guardian_dashboard():
    """Get guardian dashboard data with face recognition stats"""
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
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Driver count
        driver_count_result = supabase.table('drivers') \
            .select('*', count='exact') \
            .eq('guardian_id', guardian_id) \
            .execute()
        driver_count = driver_count_result.count if hasattr(driver_count_result, 'count') else 0
        
        # Alert counts
        total_alerts_result = supabase.table('alerts') \
            .select('*', count='exact') \
            .eq('guardian_id', guardian_id) \
            .execute()
        total_alerts = total_alerts_result.count if hasattr(total_alerts_result, 'count') else 0
        
        unread_alerts_result = supabase.table('alerts') \
            .select('*', count='exact') \
            .eq('guardian_id', guardian_id) \
            .eq('acknowledged', False) \
            .execute()
        unread_alerts = unread_alerts_result.count if hasattr(unread_alerts_result, 'count') else 0
        
        # Face recognition stats
        drivers_with_embeddings_result = supabase.table('drivers') \
            .select('*', count='exact') \
            .eq('guardian_id', guardian_id) \
            .not_.is_('face_embedding', 'null') \
            .execute()
        drivers_with_embeddings = drivers_with_embeddings_result.count if hasattr(drivers_with_embeddings_result, 'count') else 0
        
        return jsonify({
            'success': True,
            'guardian': guardian,
            'session_valid': True,
            'dashboard': {
                'driver_count': driver_count,
                'total_alerts': total_alerts,
                'unread_alerts': unread_alerts,
                'recent_alerts': recent_alerts,
                # NEW: Face recognition stats
                'face_recognition': {
                    'drivers_with_embeddings': drivers_with_embeddings,
                    'total_drivers': driver_count,
                    'completion_rate': round((drivers_with_embeddings / driver_count * 100) if driver_count > 0 else 0, 1)
                }
            },
            'drivers': drivers
        })
        
    except Exception as e:
        print(f"❌ Error in guardian_dashboard: {e}")
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
        
        # Add face image count and embedding status for each driver
        if supabase:
            for driver in drivers:
                # Face image count
                face_result = supabase.table('face_images') \
                    .select('*', count='exact') \
                    .eq('driver_id', driver['driver_id']) \
                    .execute()
                driver['face_image_count'] = face_result.count if hasattr(face_result, 'count') else 0
                
                # Check if driver has face embedding
                driver['has_face_embedding'] = driver.get('face_embedding') is not None
        
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
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        result = supabase.table('guardians') \
            .select('guardian_id, full_name, phone, email, address, registration_date, last_login, auth_provider, is_active') \
            .eq('guardian_id', guardian_id) \
            .execute()
        
        if not result.data or len(result.data) == 0:
            return jsonify({
                'success': False,
                'error': 'Guardian not found'
            }), 404
        
        guardian = result.data[0]
        
        # Convert datetime strings to ISO format if needed
        if guardian.get('registration_date'):
            guardian['registration_date'] = guardian['registration_date']
        if guardian.get('last_login'):
            guardian['last_login'] = guardian['last_login']
        
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

@app.route('/api/guardian/activity', methods=['GET'])
def get_guardian_activity():
    """Get activity log for a guardian"""
    try:
        guardian_id = request.args.get('guardian_id')
        token = request.args.get('token')
        limit = request.args.get('limit', 20, type=int)

        if not guardian_id or not token:
            return jsonify({'success': False, 'error': 'Authentication required'}), 401

        # Validate session
        if not validate_session(guardian_id, token):
            return jsonify({'success': False, 'error': 'Invalid or expired session'}), 401

        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500

        result = supabase.table('activity_log') \
            .select('log_id, action, details, ip_address, user_agent, timestamp') \
            .eq('guardian_id', guardian_id) \
            .order('timestamp', desc=True) \
            .limit(limit) \
            .execute()

        logs = result.data if result.data else []

        return jsonify({
            'success': True,
            'count': len(logs),
            'activities': logs
        })

    except Exception as e:
        print(f"❌ Error in get_guardian_activity: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

#end Region

#region Location
@app.route('/api/guardian/location', methods=['POST', 'OPTIONS'])
def update_guardian_location():
    """Update guardian's current location"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response, 200
    
    try:
        data = request.json
        guardian_id = data.get('guardian_id')
        token = data.get('token')
        location = data.get('location')
        
        print(f"\n📍 [LOCATION] Updating guardian location for {guardian_id}")
        print(f"   Location: {location}")
        
        if not guardian_id or not token:
            return jsonify({
                'success': False,
                'error': 'Missing guardian_id or token'
            }), 400
        
        # Validate session
        if not validate_session(guardian_id, token):
            return jsonify({
                'success': False,
                'error': 'Invalid or expired session'
            }), 401
        
        if not location:
            return jsonify({
                'success': False,
                'error': 'Location data required'
            }), 400
        
        # Store in database if Supabase is available
        if supabase:
            try:
                # Check if table exists, create if not
                location_data = {
                    'guardian_id': guardian_id,
                    'latitude': location.get('lat'),
                    'longitude': location.get('lng'),
                    'timestamp': location.get('timestamp', datetime.now().isoformat()),
                    'accuracy': location.get('accuracy', 0)
                }
                
                # Insert into guardian_locations table
                supabase.table('guardian_locations').insert(location_data).execute()
                print(f"✅ [LOCATION] Stored in database")
            except Exception as db_error:
                print(f"⚠️ [LOCATION] Database error: {db_error}")
                # Continue even if DB fails - location is still processed
        
        # Broadcast to connected clients if needed
        socketio.emit('guardian_location_response', {
            'success': True,
            'guardian_id': guardian_id,
            'timestamp': datetime.now().isoformat()
        }, room=f'guardian_{guardian_id}')
        
        return jsonify({
            'success': True,
            'message': 'Location updated successfully'
        })
        
    except Exception as e:
        print(f"❌ [LOCATION] Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
 
@app.route('/api/guardian/<guardian_id>/driver-locations', methods=['GET'])
def get_driver_locations(guardian_id):
    """Get latest locations for all drivers - WITH REAL-TIME DATA"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        return response, 200
    
    try:
        token = request.args.get('token')
        
        if not token:
            return jsonify({'success': False, 'error': 'Authentication required'}), 401
        
        if not validate_session(guardian_id, token):
            return jsonify({'success': False, 'error': 'Invalid or expired session'}), 401
        
        print(f"\n📍 [LOCATION] Getting driver locations for guardian {guardian_id}")
        
        locations = {}
        
        # FIRST: Get real-time locations from connected clients
        current_time = datetime.now()
        for client_id, info in connected_clients.items():
            if (info.get('type') == 'driver' and 
                str(info.get('guardian_id')) == str(guardian_id) and
                info.get('authenticated') == True):
                
                driver_id = info.get('driver_id')
                driver_name = info.get('driver_name', 'Unknown')
                last_location = info.get('last_location')
                last_location_time = info.get('last_location_time')
                drowsiness = info.get('drowsiness', 0)
                drowsiness_level = info.get('drowsiness_level', 'ALERT')
                last_message = info.get('last_message')
                
                if last_location and last_location_time:
                    # Calculate how recent this location is
                    time_since = 0
                    if last_message:
                        if isinstance(last_message, datetime):
                            time_since = (current_time - last_message).total_seconds()
                    
                    locations[driver_id] = {
                        'driver_name': driver_name,
                        'location': last_location,
                        'timestamp': last_location_time,
                        'drowsiness': drowsiness,
                        'drowsiness_level': drowsiness_level,
                        'source': 'realtime',
                        'seconds_ago': int(time_since),
                        'connected': True
                    }
                    print(f"   ✓ {driver_name}: location from real-time ({time_since:.0f}s ago)")
        
        # SECOND: Fill in missing drivers from database
        if supabase:
            try:
                # Get all drivers for this guardian
                drivers_result = supabase.table('drivers') \
                    .select('driver_id, name') \
                    .eq('guardian_id', guardian_id) \
                    .eq('is_active', True) \
                    .execute()
                
                for driver in drivers_result.data if drivers_result.data else []:
                    driver_id = driver['driver_id']
                    if driver_id not in locations:
                        # Get latest location from database
                        loc_result = supabase.table('driver_locations') \
                            .select('*') \
                            .eq('driver_id', driver_id) \
                            .order('timestamp', desc=True) \
                            .limit(1) \
                            .execute()
                        
                        if loc_result.data and len(loc_result.data) > 0:
                            loc_data = loc_result.data[0]
                            locations[driver_id] = {
                                'driver_name': driver['name'],
                                'location': {
                                    'lat': loc_data.get('latitude'),
                                    'lng': loc_data.get('longitude')
                                },
                                'timestamp': loc_data.get('timestamp'),
                                'drowsiness': loc_data.get('drowsiness', 0),
                                'drowsiness_level': loc_data.get('drowsiness_level', 'ALERT'),
                                'source': 'database',
                                'connected': False
                            }
                            print(f"   ○ {driver['name']}: location from database")
            except Exception as db_error:
                print(f"⚠️ [LOCATION] Database query error: {db_error}")
        
        print(f"   Total locations found: {len(locations)}")
        
        return jsonify({
            'success': True,
            'guardian_id': guardian_id,
            'count': len(locations),
            'locations': locations
        })
        
    except Exception as e:
        print(f"❌ [LOCATION] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    
#end Region

#region Jitsi
@app.route('/api/jitsi/room', methods=['POST'])
def create_jitsi_room():
    """Create a Jitsi room for driver-guardian streaming"""
    try:
        data = request.json
        guardian_id = data.get('guardian_id')
        driver_id = data.get('driver_id')
        token = data.get('token')
        api_key = request.args.get('api_key')
        
        # Validate either session token (for guardian) or API key (for driver)
        valid = False
        if guardian_id and token:
            valid = validate_session(guardian_id, token)
        elif api_key == os.environ.get('CLIENT_API_KEY'):
            valid = True
        
        if not valid:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
        # IMPORTANT: Get driver name from database
        driver_name = "Unknown Driver"
        if supabase:
            try:
                result = supabase.table('drivers') \
                    .select('name') \
                    .eq('driver_id', driver_id) \
                    .execute()
                if result.data and len(result.data) > 0:
                    driver_name = result.data[0]['name']
                    print(f"✅ Found driver name: {driver_name}")
                else:
                    print(f"⚠️ Driver not found in database: {driver_id}")
            except Exception as e:
                print(f"❌ Error fetching driver name: {e}")
        
        # Create unique room name
        import time
        room_name = f"drivesafe_{driver_id}_{guardian_id}_{int(time.time())}"
        
        # Store room info in database
        if supabase:
            # Check if there's an existing active room
            existing = supabase.table('jitsi_rooms') \
                .select('*') \
                .eq('driver_id', driver_id) \
                .eq('guardian_id', int(guardian_id)) \
                .eq('status', 'active') \
                .execute()
            
            if existing.data and len(existing.data) > 0:
                # Return existing room
                room_data = existing.data[0]
                return jsonify({
                    'success': True,
                    'room_name': room_data['room_name'],
                    'domain': JITSI_DOMAIN,
                    'is_new': False
                })
            
            # Create new room record with ALL fields
            room_data = {
                'room_name': room_name,
                'driver_id': driver_id,
                'guardian_id': int(guardian_id) if str(guardian_id).isdigit() else guardian_id,
                'driver_name': driver_name,  # This is CRITICAL
                'created_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            print(f"📝 Inserting room data: {room_data}")
            result = supabase.table('jitsi_rooms').insert(room_data).execute()
            print(f"✅ Room created: {result.data}")
        
        return jsonify({
            'success': True,
            'room_name': room_name,
            'domain': JITSI_DOMAIN,
            'is_new': True
        })
        
    except Exception as e:
        print(f"❌ Error creating Jitsi room: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/jitsi/rooms', methods=['GET'])
def get_active_jitsi_rooms():
    """Get all active Jitsi rooms for a guardian"""
    try:
        guardian_id = request.args.get('guardian_id')
        token = request.args.get('token')
        
        if not validate_session(guardian_id, token):
            return jsonify({'success': False, 'error': 'Invalid session'}), 401
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Database not connected'}), 500
        
        # Get active rooms
        result = supabase.table('jitsi_rooms') \
            .select('*') \
            .eq('guardian_id', int(guardian_id)) \
            .eq('status', 'active') \
            .order('created_at', desc=True) \
            .execute()
        
        rooms = []
        for room in result.data if result.data else []:
            rooms.append({
                'room_name': room['room_name'],
                'driver_id': room['driver_id'],
                'driver_name': room.get('driver_name', 'Unknown'),
                'created_at': room['created_at'],
                'domain': JITSI_DOMAIN
            })
        
        return jsonify({
            'success': True,
            'rooms': rooms,
            'count': len(rooms)
        })
        
    except Exception as e:
        print(f"❌ Error getting Jitsi rooms: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/jitsi/room/end', methods=['POST'])
def end_jitsi_room():
    """Mark a Jitsi room as ended"""
    try:
        data = request.json
        room_name = data.get('room_name')
        guardian_id = data.get('guardian_id')
        token = data.get('token')
        
        if not validate_session(guardian_id, token):
            return jsonify({'success': False, 'error': 'Invalid session'}), 401
        
        if supabase:
            supabase.table('jitsi_rooms') \
                .update({'status': 'ended', 'ended_at': datetime.now().isoformat()}) \
                .eq('room_name', room_name) \
                .execute()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

#end Region     

#region Driver Management
@app.route('/api/register-driver', methods=['POST'])
def register_driver():
    """Register a new driver with face images uploaded to Cloudinary - FIXED VERSION"""
    global CLOUDINARY_ENABLED 

    try:
        data = request.json
        print(f"   Received data keys: {list(data.keys())}")
        driver_name = data.get('driver_name')
        driver_phone = data.get('driver_phone')
        guardian_id = data.get('guardian_id')
        token = data.get('token')
        face_images = data.get('face_images', [])  
        driver_email = data.get('driver_email')
        driver_address = data.get('driver_address')
        
        # Validate required fields
        if not all([driver_name, driver_phone, guardian_id, token]):
            print("❌ [DRIVER REGISTRATION] Missing required fields:")
            print(f"   driver_name: {bool(driver_name)}")
            print(f"   driver_phone: {bool(driver_phone)}")
            print(f"   guardian_id: {bool(guardian_id)}")

            return jsonify({
                'success': False,
                'error': 'Missing required fields: driver_name, driver_phone, guardian_id, or token'
            }), 400
        
        
        # Validate guardian session
        if not validate_session(guardian_id, token):

            return jsonify({
                'success': False,
                'error': 'Invalid or expired guardian session. Please login again.'
            }), 401
                
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
        license_number = data.get('license_number', '')
        capture_angles = data.get('capture_angles', ['front', 'left', 'right'])
        print(f"   Driver ID: {driver_id}")
        print(f"   Reference Number: {reference_number}")
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Check if phone already exists
        phone_check = supabase.table('drivers') \
            .select('driver_id') \
            .eq('phone', driver_phone) \
            .execute()
        
        if phone_check.data and len(phone_check.data) > 0:
            print(f"❌ [DRIVER REGISTRATION] Phone already registered: {driver_phone}")
            return jsonify({
                'success': False,
                'error': f'Phone number {driver_phone} is already registered for another driver.'
            }), 409
        
        # Check if reference number already exists
        ref_check = supabase.table('drivers') \
            .select('driver_id') \
            .eq('reference_number', reference_number) \
            .execute()
        
        if ref_check.data and len(ref_check.data) > 0:
            # Generate new reference number
            new_ref_number = f"REF{str(uuid.uuid4().int)[:8].upper()}"
            print(f"🔧 [DRIVER REGISTRATION] Reference number already exists. Generated new: {new_ref_number}")
            reference_number = new_ref_number
        
        print(f"\n🗄️ [DRIVER REGISTRATION] Inserting driver into Supabase...")
        
        try:
            # Insert driver into database
            new_driver = {
                'driver_id': driver_id,
                'name': driver_name,
                'phone': driver_phone,
                'email': driver_email,
                'address': driver_address,
                'reference_number': reference_number,
                'license_number': license_number,
                'guardian_id': guardian_id,
                'registration_date': datetime.now().isoformat(),
                'is_active': True
            }
            
            driver_result = supabase.table('drivers').insert(new_driver).execute()
            
            if not driver_result.data:
                raise Exception("Failed to insert driver")
            
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
                    upload_errors.append(f"Image {i} is empty or invalid")
                    continue
                
                try:
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    public_id = f"driver_faces/{driver_id}/{capture_angle}_{timestamp_str}"
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
                    
                    # Insert face image record
                    face_image_data = {
                        'driver_id': driver_id,
                        'image_path': cloudinary_url,
                        'capture_date': datetime.now().isoformat()
                    }
                    supabase.table('face_images').insert(face_image_data).execute()
                    
                    saved_images.append({
                        'angle': capture_angle,
                        'public_id': public_id,
                        'url': cloudinary_url,
                        'upload_time': upload_time,
                        'upload_method': 'cloudinary'
                    })
                    
                    if i < len(face_images):
                        time.sleep(1)
                    
                except cloudinary.exceptions.Error as cloudinary_error:  
                    error_msg = f"Cloudinary API error for image {i} ({capture_angle}): {str(cloudinary_error)}"
                    print(f"Cloudinary is not available: {error_msg}")
                    upload_errors.append(error_msg)
                    
                except Exception as upload_error:
                    error_msg = f"Error uploading image {i} ({capture_angle}): {str(upload_error)}"
                    upload_errors.append(error_msg)
                    continue
            
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
                    face_image_data = {
                        'driver_id': driver_id,
                        'image_path': image_url,
                        'capture_date': datetime.now().isoformat()
                    }
                    supabase.table('face_images').insert(face_image_data).execute()
                    
                    saved_images.append({
                        'angle': capture_angle,
                        'url': image_url,
                        'filepath': filepath,
                        'upload_method': 'local'
                    })
                    
                except Exception as save_error:
                    error_msg = f"Error saving image {i} locally: {str(save_error)}"
                    print(f"❌ [DRIVER REGISTRATION] {error_msg}")
                    upload_errors.append(error_msg)
            
            print(f"\n📊 [DRIVER REGISTRATION] Local storage summary:")
            print(f"   Successfully saved: {len(saved_images)}")
            print(f"   Save errors: {len(upload_errors)}")
        
        # ==================== REGISTRATION COMPLETION ====================
        if len(saved_images) == 0:
            print(f"❌ [DRIVER REGISTRATION] NO face images were saved for driver {driver_id}")
            print(f"   Errors: {upload_errors}")
            
            # Rollback driver registration if no images were saved
            supabase.table('drivers').delete().eq('driver_id', driver_id).execute()
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
        log_data = {
            'guardian_id': guardian_id,
            'action': 'DRIVER_REGISTERED',
            'details': f'Registered driver: {driver_name} (ID: {driver_id}) with {len(saved_images)} face images',
            'timestamp': datetime.now().isoformat()
        }
        supabase.table('activity_log').insert(log_data).execute()
        
        # Get guardian info
        guardian_result = supabase.table('guardians') \
            .select('full_name') \
            .eq('guardian_id', guardian_id) \
            .execute()
        
        guardian_name = guardian_result.data[0]['full_name'] if guardian_result.data else 'Unknown Guardian'
        
        print(f"\n✅ [DRIVER REGISTRATION] Registration COMPLETE!")
        print(f"   Driver: {driver_name} (ID: {driver_id})")
        print(f"   Images saved: {len(saved_images)}")
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
            'message': f'Driver registered successfully with {len(saved_images)} face images'
        }
        
        # Send real-time notification to guardian
        try:
            notification_data = {
                'type': 'driver_registered',
                'driver_id': driver_id,
                'driver_name': driver_name,
                'guardian_id': guardian_id,
                'face_images_count': len(saved_images),
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
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        result = supabase.table('face_images') \
            .select('image_id, image_path, capture_date') \
            .eq('driver_id', driver_id) \
            .order('capture_date', desc=True) \
            .execute()
        
        images = result.data if result.data else []
        
        # Return Cloudinary URLs directly
        images_data = []
        for img in images:
            images_data.append({
                'image_id': img['image_id'],
                'url': img['image_path'],
                'capture_date': img['capture_date'],
                'is_cloudinary': 'cloudinary' in img['image_path'].lower()
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

@app.route('/api/driver/<driver_id>', methods=['GET'])
def get_driver_details(driver_id):
    """Get detailed information about a specific driver"""
    try:
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Get driver details with guardian info
        result = supabase.table('drivers') \
            .select('*, guardians!inner(full_name, phone, email)') \
            .eq('driver_id', driver_id) \
            .execute()
        
        if not result.data or len(result.data) == 0:
            return jsonify({
                'success': False,
                'error': 'Driver not found'
            }), 404
        
        driver = result.data[0]
        
        # Extract guardian info
        if 'guardians' in driver:
            driver['guardian_name'] = driver['guardians']['full_name']
            driver['guardian_phone'] = driver['guardians']['phone']
            driver['guardian_email'] = driver['guardians']['email']
            del driver['guardians']
        
        # Get alert count
        alert_result = supabase.table('alerts') \
            .select('*', count='exact') \
            .eq('driver_id', driver_id) \
            .execute()
        driver['alert_count'] = alert_result.count if hasattr(alert_result, 'count') else 0
        
        # Get unacknowledged alert count
        unread_result = supabase.table('alerts') \
            .select('*', count='exact') \
            .eq('driver_id', driver_id) \
            .eq('acknowledged', False) \
            .execute()
        driver['unread_alerts'] = unread_result.count if hasattr(unread_result, 'count') else 0
        
        # Get face image count
        face_result = supabase.table('face_images') \
            .select('*', count='exact') \
            .eq('driver_id', driver_id) \
            .execute()
        driver['face_image_count'] = face_result.count if hasattr(face_result, 'count') else 0
        
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

@app.route('/api/guardian/drivers-with-embeddings', methods=['GET'])
def get_drivers_with_embeddings():
    guardian_id = request.args.get('guardian_id')
    token = request.args.get('token')
    if not guardian_id or not token:
        return jsonify({'success': False, 'error': 'Missing credentials'}), 401
    if not validate_session(guardian_id, token):
        return jsonify({'success': False, 'error': 'Invalid session'}), 401

    if not supabase:
        return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500

    result = supabase.table('drivers') \
        .select('driver_id, name, face_embedding') \
        .eq('guardian_id', guardian_id) \
        .not_.is_('face_embedding', 'null') \
        .execute()

    drivers = result.data if result.data else []
    
    # Parse embedding from JSON string to Python list
    for d in drivers:
        if d.get('face_embedding') and isinstance(d['face_embedding'], str):
            try:
                d['face_embedding'] = json.loads(d['face_embedding'])
            except:
                pass

    return jsonify({'success': True, 'drivers': drivers})

@app.route('/api/driver/<driver_id>/store-embedding', methods=['POST'])
def store_embedding(driver_id):
    data = request.json
    guardian_id = data.get('guardian_id')
    token = data.get('token')
    embedding = data.get('embedding')   # list of floats
    if not all([guardian_id, token, embedding]):
        return jsonify({'success': False, 'error': 'Missing data'}), 400
    if not validate_session(guardian_id, token):
        return jsonify({'success': False, 'error': 'Invalid session'}), 401

    if not supabase:
        return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500

    # Verify driver belongs to guardian
    driver_check = supabase.table('drivers') \
        .select('driver_id') \
        .eq('driver_id', driver_id) \
        .eq('guardian_id', guardian_id) \
        .execute()

    if not driver_check.data or len(driver_check.data) == 0:
        return jsonify({'success': False, 'error': 'Driver not found'}), 404

    embedding_json = json.dumps(embedding)
    supabase.table('drivers') \
        .update({'face_embedding': embedding_json}) \
        .eq('driver_id', driver_id) \
        .execute()

    return jsonify({'success': True, 'message': 'Embedding stored'})

@app.route('/api/driver/<identifier>/name', methods=['GET'])
def get_driver_name(identifier):
    """Get driver name by either driver_id or reference_number"""
    try:
        api_key = request.args.get('api_key')
        
        expected_api_key = os.environ.get('CLIENT_API_KEY', 'your_secret_key')
        if api_key != expected_api_key:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Try by driver_id first
        result = supabase.table('drivers') \
            .select('name, guardian_id, driver_id') \
            .eq('driver_id', identifier) \
            .eq('is_active', True) \
            .execute()
        
        driver = result.data[0] if result.data else None
        
        # If not found, try by reference_number
        if not driver:
            result = supabase.table('drivers') \
                .select('name, guardian_id, driver_id') \
                .eq('reference_number', identifier) \
                .eq('is_active', True) \
                .execute()
            driver = result.data[0] if result.data else None
        
        if driver:
            return jsonify({
                'success': True,
                'name': driver['name'],
                'guardian_id': driver['guardian_id'],
                'driver_id': driver['driver_id']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Driver not found'
            }), 404
                
    except Exception as e:
        logger.error(f"Error in get_driver_name: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/driver/by-reference/<identifier>', methods=['GET'])
def get_driver_by_reference(identifier):
    """Get driver details by either driver_id or reference_number"""
    try:
        api_key = request.args.get('api_key')
        
        # Simple API key check
        expected_api_key = os.environ.get('CLIENT_API_KEY', 'your_secret_key')
        if api_key != expected_api_key:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Try to find by driver_id first
        result = supabase.table('drivers') \
            .select('driver_id, name, guardian_id, phone, email, license_number, reference_number') \
            .eq('driver_id', identifier) \
            .eq('is_active', True) \
            .execute()
        
        driver = result.data[0] if result.data else None
        
        # If not found, try by reference_number
        if not driver:
            result = supabase.table('drivers') \
                .select('driver_id, name, guardian_id, phone, email, license_number, reference_number') \
                .eq('reference_number', identifier) \
                .eq('is_active', True) \
                .execute()
            driver = result.data[0] if result.data else None
        
        if driver:
            logger.info(f"Found driver: {driver.get('name')} (ID: {driver.get('driver_id')})")
            return jsonify({
                'success': True,
                'driver': driver
            })
        else:
            # Debug: Show what's in the database
            sample = supabase.table('drivers') \
                .select('driver_id, name, reference_number') \
                .limit(5) \
                .execute()
            sample_list = sample.data if sample.data else []
            logger.info(f"Sample drivers in DB: {sample_list}")
            
            return jsonify({
                'success': False,
                'error': 'Driver not found',
                'identifier_searched': identifier,
                'sample_drivers': sample_list
            }), 404
                
    except Exception as e:
        logger.error(f"Error in get_driver_by_reference: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/driver/<driver_id>/guardian', methods=['GET'])
def get_driver_guardian(driver_id):
    """Get guardian ID for a driver (for WebRTC connection)"""
    try:
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        result = supabase.table('drivers') \
            .select('guardian_id, name') \
            .eq('driver_id', driver_id) \
            .eq('is_active', True) \
            .execute()
        
        if not result.data or len(result.data) == 0:
            return jsonify({
                'success': False,
                'error': 'Driver not found'
            }), 404
        
        driver_data = result.data[0]
        
        return jsonify({
            'success': True,
            'guardian_id': driver_data['guardian_id'],
            'driver_name': driver_data['name'],
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
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        result = supabase.table('drivers') \
            .select('guardian_id, name') \
            .eq('driver_id', driver_id) \
            .eq('guardian_id', provided_guardian_id) \
            .eq('is_active', True) \
            .execute()
        
        if result.data and len(result.data) > 0:
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

        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500

        # Check if driver belongs to this guardian
        driver_check = supabase.table('drivers') \
            .select('driver_id') \
            .eq('driver_id', driver_id) \
            .eq('guardian_id', guardian_id) \
            .execute()

        if not driver_check.data or len(driver_check.data) == 0:
            return jsonify({
                'success': False,
                'error': 'Driver not found or not authorized'
            }), 404

        # Build update dictionary
        update_data = {}
        
        # --- PHONE UPDATE ---
        new_phone = data.get('phone')
        if new_phone:
            phone_clean = re.sub(r'[\s\-\(\)\+]', '', new_phone)

            phone_check = supabase.table('drivers') \
                .select('driver_id') \
                .eq('phone', phone_clean) \
                .neq('driver_id', driver_id) \
                .execute()

            if phone_check.data and len(phone_check.data) > 0:
                return jsonify({
                    'success': False,
                    'error': f'Phone number {phone_clean} is already registered'
                }), 409

            update_data['phone'] = phone_clean

        # --- OTHER FIELDS ---
        fields_to_update = ['name', 'email', 'address', 'license_number']

        for field in fields_to_update:
            if field in data:
                update_data[field] = data[field]

        if not update_data:
            return jsonify({
                'success': False,
                'error': 'No fields to update'
            }), 400

        update_data['updated_at'] = datetime.now().isoformat()

        supabase.table('drivers') \
            .update(update_data) \
            .eq('driver_id', driver_id) \
            .execute()

        # Get updated driver
        driver_result = supabase.table('drivers') \
            .select('*') \
            .eq('driver_id', driver_id) \
            .execute()
        
        driver = driver_result.data[0] if driver_result.data else {}

        # Log activity
        log_activity(guardian_id, 'DRIVER_UPDATED', f'Updated driver: {driver.get("name")} (ID: {driver_id})')

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
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        update_data = {}
        
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
            phone_check = supabase.table('guardians') \
                .select('guardian_id') \
                .eq('phone', phone_clean) \
                .neq('guardian_id', guardian_id) \
                .execute()
            
            if phone_check.data and len(phone_check.data) > 0:
                return jsonify({
                    'success': False,
                    'error': f'Phone number {phone_clean} is already registered for another guardian'
                }), 409
            
            update_data['phone'] = phone_clean
        
        # Other fields that can be updated
        fields_to_update = ['full_name', 'email', 'address']
        for field in fields_to_update:
            if field in data:
                update_data[field] = data[field]
        
        if not update_data:
            return jsonify({
                'success': False,
                'error': 'No fields to update'
            }), 400
        
        # Execute update
        supabase.table('guardians') \
            .update(update_data) \
            .eq('guardian_id', guardian_id) \
            .execute()
        
        # Get updated guardian info
        guardian_result = supabase.table('guardians') \
            .select('*') \
            .eq('guardian_id', guardian_id) \
            .execute()
        
        guardian = guardian_result.data[0] if guardian_result.data else {}
        
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

#region Alert

@app.route('/api/driver/<driver_id>/details', methods=['GET'])
def get_driver_details_api(driver_id):
    """Get detailed driver information - Used by Project 2"""
    try:
        guardian_id = request.args.get('guardian_id')
        token = request.args.get('token')
        api_key = request.args.get('api_key')
        
        # Check either API key or session token
        expected_api_key = os.environ.get('CLIENT_API_KEY', 'your_secret_key')
        valid_api_key = api_key == expected_api_key
        
        # Validate session if guardian_id and token provided
        valid_session = False
        if guardian_id and token:
            valid_session = validate_session(guardian_id, token)
        
        if not (valid_api_key or valid_session):
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Get driver details with guardian info
        result = supabase.table('drivers') \
            .select('*, guardians!left(full_name, phone)') \
            .eq('driver_id', driver_id) \
            .eq('is_active', True) \
            .execute()
        
        if not result.data or len(result.data) == 0:
            return jsonify({
                'success': False,
                'error': 'Driver not found'
            }), 404
        
        driver = result.data[0]
        
        # Extract guardian info
        if 'guardians' in driver and driver['guardians']:
            driver['guardian_name'] = driver['guardians']['full_name']
            driver['guardian_phone'] = driver['guardians']['phone']
            del driver['guardians']
        
        # Get face image count
        face_result = supabase.table('face_images') \
            .select('*', count='exact') \
            .eq('driver_id', driver_id) \
            .execute()
        driver['face_image_count'] = face_result.count if hasattr(face_result, 'count') else 0
        
        # Remove embedding from response if exists
        if 'face_embedding' in driver:
            del driver['face_embedding']
        
        return jsonify({
            'success': True,
            'driver': driver
        })
            
    except Exception as e:
        logger.error(f"Error in get_driver_details_api: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/send-alert', methods=['POST'])
def receive_alert():
    """Receive alert from Project 2 and store in database"""
    try:
        data = request.json
        api_key = request.args.get('api_key')
        
        expected_api_key = os.environ.get('CLIENT_API_KEY', 'your_secret_key')
        if api_key != expected_api_key:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        driver_id = data.get('driver_id')
        driver_name = data.get('driver_name')
        guardian_id = data.get('guardian_id')
        severity = data.get('severity', 'medium')
        message = data.get('message', 'Drowsiness detected')
        confidence = data.get('confidence', 0.0)
        location = data.get('location', 'Unknown')
        detection_details = data.get('detection_details', {})
        
        if not driver_id:
            return jsonify({
                'success': False,
                'error': 'Driver ID required'
            }), 400
        
        # Insert alert
        alert_data = {
            'driver_id': driver_id,
            'guardian_id': guardian_id,
            'severity': severity,
            'message': message,
            'detection_details': json.dumps(detection_details),
            'source': 'drowsiness_detection',
            'timestamp': datetime.now().isoformat()
        }
        
        alert_result = supabase.table('alerts').insert(alert_data).execute()
        
        alert_id = None
        if alert_result.data and len(alert_result.data) > 0:
            alert_id = alert_result.data[0].get('alert_id')
        
        # Insert drowsiness event
        event_data = {
            'driver_id': driver_id,
            'guardian_id': guardian_id,
            'confidence': confidence,
            'ear': detection_details.get('ear', 0.0),
            'mar': detection_details.get('mar', 0.0),
            'perclos': detection_details.get('perclos', 0.0),
            'timestamp': datetime.now().isoformat()
        }
        supabase.table('drowsiness_events').insert(event_data).execute()
        
        # Notify via WebSocket if guardian connected
        if guardian_id:
            alert_notification = {
                'alert_id': alert_id,
                'driver_id': driver_id,
                'driver_name': driver_name,
                'severity': severity,
                'message': message,
                'location': location,
                'timestamp': datetime.now().isoformat()
            }
            # Emit to guardian's room
            socketio.emit('guardian_alert', alert_notification, room=f'guardian_{guardian_id}')
        
        logger.info(f"Alert received: {severity} from driver {driver_name} ({driver_id})")
        
        return jsonify({
            'success': True,
            'alert_id': alert_id,
            'message': 'Alert received and stored'
        })
            
    except Exception as e:
        logger.error(f"Error in receive_alert: {e}")
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
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Build query
        query = supabase.table('alerts') \
            .select('*, drivers!inner(name)') \
            .eq('guardian_id', guardian_id)
        
        if acknowledged is not None:
            if acknowledged.lower() == 'true':
                query = query.eq('acknowledged', True)
            elif acknowledged.lower() == 'false':
                query = query.eq('acknowledged', False)
        
        result = query.order('timestamp', desc=True).limit(limit).execute()
        
        alerts = result.data if result.data else []
        
        # Process alerts
        result_alerts = []
        for alert in alerts:
            if alert.get('detection_details'):
                try:
                    alert['detection_details'] = json.loads(alert['detection_details'])
                except:
                    pass
            
            # Extract driver name
            if 'drivers' in alert and alert['drivers']:
                alert['driver_name'] = alert['drivers']['name']
                del alert['drivers']
            
            result_alerts.append(alert)
        
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
#end Region
@app.route('/api/debug/active-connections', methods=['GET'])
def debug_active_connections():
    """Debug endpoint to see all active WebSocket connections"""
    try:
        connections = []
        for sid, info in connected_clients.items():
            connections.append({
                'sid': sid,
                'type': info.get('type'),
                'guardian_id': info.get('guardian_id'),
                'driver_id': info.get('driver_id'),
                'driver_name': info.get('driver_name'),
                'authenticated': info.get('authenticated'),
                'connected_at': info.get('connected_at').isoformat() if info.get('connected_at') else None,
                'last_ping': info.get('last_ping').isoformat() if info.get('last_ping') else None
            })
        
        return jsonify({
            'success': True,
            'total_connections': len(connections),
            'connections': connections
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/debug/connections', methods=['GET'])
def debug_connections():
    """Debug endpoint to see all connected WebSocket clients"""
    if not supabase:
        return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
    
    clients_info = []
    for sid, info in connected_clients.items():
        clients_info.append({
            'sid': sid,
            'type': info.get('type'),
            'guardian_id': info.get('guardian_id'),
            'driver_id': info.get('driver_id'),
            'driver_name': info.get('driver_name'),
            'authenticated': info.get('authenticated'),
            'connected_at': info.get('connected_at').isoformat() if info.get('connected_at') else None,
            'last_ping': info.get('last_ping').isoformat() if info.get('last_ping') else None,
            'transport': info.get('transport'),
            'query_params': info.get('query_params', {})
        })
    
    return jsonify({
        'success': True,
        'total_connections': len(connected_clients),
        'clients': clients_info,
        'guardians_online': len([c for c in clients_info if c.get('type') == 'guardian']),
        'drivers_online': len([c for c in clients_info if c.get('type') == 'driver'])
    })
    
@app.route('/api/debug/check-key', methods=['GET'])
def debug_check_key():
    """Check if API key is properly set in environment"""
    received_key = request.args.get('api_key', 'NOT PROVIDED')
    expected_key = os.environ.get('CLIENT_API_KEY', 'NOT SET IN ENV')
    
    return jsonify({
        'success': True,
        'received_key': received_key,
        'expected_key': expected_key,
        'keys_match': received_key == expected_key,
        'expected_key_length': len(expected_key) if expected_key != 'NOT SET IN ENV' else 0,
        'received_key_length': len(received_key),
        'environment': os.environ.get('RENDER', 'local'),
        'all_env_keys': list(os.environ.keys())  # See what env vars are available
    })

@app.route('/api/guardian/active-drivers', methods=['GET'])
def get_active_drivers():
    """Return list of drivers that should be monitored."""
    # Optionally require an API key for the AI service
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f'Bearer {os.getenv("AI_SERVICE_TOKEN")}':
        return jsonify({'error': 'Unauthorized'}), 401

    if not supabase:
        return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500

    result = supabase.table('drivers') \
        .select('driver_id, name as driver_name, guardian_id') \
        .eq('is_active', True) \
        .order('registration_date', desc=True) \
        .execute()

    drivers = result.data if result.data else []
    
    return jsonify({'success': True, 'drivers': drivers})

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
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        # Check if alert belongs to this guardian
        alert_check = supabase.table('alerts') \
            .select('*, drivers!inner(name)') \
            .eq('alert_id', alert_id) \
            .eq('guardian_id', guardian_id) \
            .execute()
        
        if not alert_check.data or len(alert_check.data) == 0:
            return jsonify({
                'success': False,
                'error': 'Alert not found or not authorized'
            }), 404
        
        alert = alert_check.data[0]
        
        # Acknowledge the alert
        supabase.table('alerts') \
            .update({'acknowledged': True}) \
            .eq('alert_id', alert_id) \
            .eq('guardian_id', guardian_id) \
            .execute()
        
        # Log activity
        driver_name = alert['drivers']['name'] if 'drivers' in alert else 'Unknown'
        log_activity(guardian_id, 'ALERT_ACKNOWLEDGED', f'Acknowledged alert #{alert_id} for driver {driver_name}')
        
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
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        result = supabase.table('drivers') \
            .select('*, guardians!left(full_name, phone)') \
            .order('registration_date', desc=True) \
            .execute()
        
        drivers = result.data if result.data else []
        
        # Extract guardian info
        for driver in drivers:
            if 'guardians' in driver and driver['guardians']:
                driver['guardian_name'] = driver['guardians']['full_name']
                driver['guardian_phone'] = driver['guardians']['phone']
                del driver['guardians']
            
        return jsonify({
            'success': True,
            'count': len(drivers),
            'drivers': drivers
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/db-alerts', methods=['GET'])
@require_admin_auth
def admin_get_alerts():
    """Get all alerts from database - ADMIN ONLY"""
    try:
        limit = request.args.get('limit', 100, type=int)
        
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        result = supabase.table('alerts') \
            .select('*, drivers!inner(name), guardians!left(full_name)') \
            .order('timestamp', desc=True) \
            .limit(limit) \
            .execute()
        
        alerts = result.data if result.data else []

        for alert in alerts:
            if alert.get('detection_details'):
                try:
                    alert['detection_details'] = json.loads(alert['detection_details'])
                except:
                    pass

            if 'drivers' in alert and alert['drivers']:
                alert['driver_name'] = alert['drivers']['name']
                del alert['drivers']
            
            # Extract guardian name
            if 'guardians' in alert and alert['guardians']:
                alert['guardian_name'] = alert['guardians']['full_name']
                del alert['guardians']
            
        return jsonify({
            'success': True,
            'count': len(alerts),
            'alerts': alerts
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/db-guardians', methods=['GET'])
@require_admin_auth
def admin_get_guardians():
    """Get all guardians from database - ADMIN ONLY"""
    try:
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        result = supabase.table('guardians') \
            .select('*') \
            .order('registration_date', desc=True) \
            .execute()
        
        guardians = result.data if result.data else []
        
        # Add driver and alert counts
        for guardian in guardians:
            # Driver count
            driver_result = supabase.table('drivers') \
                .select('*', count='exact') \
                .eq('guardian_id', guardian['guardian_id']) \
                .execute()
            guardian['driver_count'] = driver_result.count if hasattr(driver_result, 'count') else 0
            
            # Alert count
            alert_result = supabase.table('alerts') \
                .select('*', count='exact') \
                .eq('guardian_id', guardian['guardian_id']) \
                .execute()
            guardian['alert_count'] = alert_result.count if hasattr(alert_result, 'count') else 0
            
        return jsonify({
            'success': True,
            'count': len(guardians),
            'guardians': guardians
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
@require_admin_auth
def admin_stats():
    """Admin statistics endpoint - ADMIN ONLY"""
    try:
        if not supabase:
            return jsonify({'success': False, 'error': 'Supabase not initialized'}), 500
        
        stats = {}
        
        # Total alerts
        alert_result = supabase.table('alerts') \
            .select('*', count='exact') \
            .execute()
        stats['total_alerts'] = alert_result.count if hasattr(alert_result, 'count') else 0
        
        # Total drivers
        driver_result = supabase.table('drivers') \
            .select('*', count='exact') \
            .execute()
        stats['total_drivers'] = driver_result.count if hasattr(driver_result, 'count') else 0
        
        # Total guardians
        guardian_result = supabase.table('guardians') \
            .select('*', count='exact') \
            .execute()
        stats['total_guardians'] = guardian_result.count if hasattr(guardian_result, 'count') else 0
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'system_status': {
                'database': 'supabase',
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
        
        # Step 4: Check database with Supabase
        if not supabase:
            result['error'] = 'Supabase not initialized'
            result['success'] = False
            return jsonify(result)
        
        # First check what's in the database
        all_users_result = supabase.table('guardians') \
            .select('phone, guardian_id') \
            .order('guardian_id', desc=True) \
            .limit(5) \
            .execute()
        
        result['steps'].append({
            'step': 4,
            'action': 'Database check - all users',
            'all_users_in_db': all_users_result.data if all_users_result.data else []
        })
        
        # Now search for our phone
        user_result = supabase.table('guardians') \
            .select('guardian_id, full_name, password_hash, is_active') \
            .eq('phone', lookup_phone) \
            .execute()
        
        if not user_result.data or len(user_result.data) == 0:
            result['steps'].append({
                'step': 5,
                'action': 'User not found in database',
                'lookup_phone_used': lookup_phone,
                'user_found': False
            })
            result['error'] = f'No user found with phone: {lookup_phone}'
            result['success'] = False
            return jsonify(result)
        
        user_data = user_result.data[0]
        guardian_id = user_data['guardian_id']
        full_name = user_data['full_name']
        stored_hash = user_data['password_hash']
        is_active = user_data['is_active']
        
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
    """Check Supabase schema - tables should be created via SQL editor"""
    return jsonify({
        'success': True,
        'message': 'Schema fixes should be applied via Supabase SQL editor',
        'instructions': [
            'ALTER TABLE guardians ADD COLUMN IF NOT EXISTS google_id TEXT UNIQUE;',
            'ALTER TABLE guardians ADD COLUMN IF NOT EXISTS auth_provider TEXT DEFAULT \'phone\';',
            'ALTER TABLE drivers ADD COLUMN IF NOT EXISTS face_embedding JSONB;',
            'CREATE INDEX IF NOT EXISTS idx_guardians_google ON guardians(google_id);',
            'CREATE INDEX IF NOT EXISTS idx_drivers_face_embedding ON drivers((face_embedding IS NOT NULL)) WHERE face_embedding IS NOT NULL;'
        ]
    })

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

def add_face_embedding_column():
    """Add face_embedding column to drivers table if it doesn't exist"""
    try:
        if not supabase:
            print("⚠️ Supabase not initialized, can't add column")
            return
            
        # This should be done via SQL editor, but we'll check if it exists
        result = supabase.table('drivers').select('face_embedding').limit(1).execute()
        print("✅ face_embedding column exists in drivers table")
    except Exception as e:
        print(f"⚠️ Could not verify face_embedding column: {e}")
        print("   Please run in Supabase SQL editor: ALTER TABLE drivers ADD COLUMN face_embedding JSONB;")


#region Start Task
def startup_tasks():
    """Run startup tasks"""
    print(f"\n{'='*70}")
    print("🚗 DRIVER DROWSINESS ALERT SYSTEM - CAPSTONE PROJECT")
    print(f"{'='*70}")
    
    print("🌐 DEPLOYMENT: Firebase Hosting + Render Backend")
    print("📊 Database: Supabase PostgreSQL")
    print("🔒 Security: bcrypt password hashing enabled")
    print("☁️  Cloudinary: Image storage enabled" if CLOUDINARY_ENABLED else "⚠️  Cloudinary: Not configured")
    print("🔥 Firebase: Hosting integration enabled")
    
    # Check environment variables
    print("\n🔧 Environment Check:")
    
    # Check Supabase config
    if SUPABASE_URL and SUPABASE_KEY:
        print(f"   ✅ SUPABASE_URL: [configured]")
        print(f"   ✅ SUPABASE_KEY: [configured]")
    else:
        print("❌ Supabase configuration missing")
    
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
    
    # Face recognition setup
    print("\n🔧 Face Recognition Setup:")
    try:
        add_face_embedding_column()
        print("✅ Face recognition schema checked/updated")
        
        # Test DeepFace availability
        try:
            import deepface
            print(f"   ✅ DeepFace version: {deepface.__version__}")
            print(f"   ✅ Face recognition ready")
        except ImportError as e:
            print(f"   ⚠️ DeepFace not available: {e}")
    except Exception as e:
        print(f"   ⚠️ Face recognition setup error: {e}")
    
    # Database initialization
    print("\n🗄️  Database Initialization:")
    try:
        if init_db():
            print("✅ Supabase connection verified")
        else:
            print("⚠️ Supabase connection had issues")
        
        # Update schema check
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
        print(f"   ✅ Running on Render.com with Supabase")
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
        ("POST", "/api/identify-driver", "Identify driver by face"),
        ("POST", "/api/driver/<driver_id>/generate-embedding", "Generate face embedding"),
    ]
    
    for method, path, desc in endpoints:
        print(f"   • {method:6} {path:30} - {desc}")
    

#end Region

#region Cleanup
def cleanup_stale_connections():
    """Background task to clean up stale connections"""
    while True:
        time.sleep(30)  # Check every 30 seconds
        try:
            current_time = datetime.now()
            stale_clients = []
            
            for client_id, info in connected_clients.items():
                last_message = info.get('last_message')
                if last_message:
                    if isinstance(last_message, datetime):
                        seconds_since = (current_time - last_message).total_seconds()
                        # Only clean up if no messages for 2 minutes AND no location for 2 minutes
                        if seconds_since > 120 and info.get('last_location') is None:
                            stale_clients.append(client_id)
                            print(f"🧹 Found stale client {client_id} - {seconds_since:.0f}s no messages")
            
            for client_id in stale_clients:
                try:
                    socketio.server.disconnect(client_id, silent=True)
                    if client_id in connected_clients:
                        del connected_clients[client_id]
                except:
                    pass
            
            if stale_clients:
                print(f"   Cleaned up {len(stale_clients)} stale connections")
                
        except Exception as e:
            print(f"⚠️ Error in cleanup task: {e}")

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_stale_connections, daemon=True)
cleanup_thread.start()
print("✅ Connection cleanup task started")

#end Region

#region Main Entry
if __name__ == '__main__':
    startup_tasks()
    create_location_tables()
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    
    try:
        import dns.resolver
        dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
        dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4'] 
        print("✅ DNS resolver configured for Render")
    except Exception as e:
        print(f"⚠️ DNS resolver configuration warning: {e}")
    
    try:
        import eventlet
        import eventlet.wsgi
        from eventlet import listen

        # Monkey patch for Python 3.10 compatibility
        eventlet.monkey_patch(
            socket=True,
            select=True,
            time=True,
            os=True,
            thread=True,
            subprocess=True
        )
        
        listen_socket = eventlet.listen((host, port))
        listen_socket.setsockopt(eventlet.socket.SOL_SOCKET, eventlet.socket.SO_REUSEADDR, 1)
   
        eventlet.wsgi.MAX_HEADER_LINE = 16384
        eventlet.wsgi.MAX_REQUEST_LINE = 32768
        eventlet.wsgi.MAX_READ_BYTES = 65536
        eventlet.wsgi.DEFAULT_MAX_SIMULTANEOUS_REQUESTS = 1000
 
        
        # Add connection tracking middleware
        def connection_count_middleware(environ, start_response):
            client_id = environ.get('REMOTE_ADDR', 'unknown')
            print(f"📡 New connection from {client_id} at {datetime.now().strftime('%H:%M:%S')}")
            return app(environ, start_response)

        eventlet.wsgi.server(
            listen_socket,
            connection_count_middleware,  
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
            print("   - SUPABASE_URL=your-supabase-url")
            print("   - SUPABASE_KEY=your-supabase-key")
            print("   - WEBSOCKET_ENABLED=true")
            sys.exit(1)
            
#end Region