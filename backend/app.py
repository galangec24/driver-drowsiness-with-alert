"""
CAPSTONE PROJECT - DRIVER ALERT SYSTEM
PostgreSQL-only Version for Render.com Deployment
FIXED VERSION - Corrected bcrypt implementation and auto-login
"""

import os
from pathlib import Path
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / '.env')

from flask import Flask, request, jsonify, send_from_directory, Response, redirect
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

# ==================== POSTGRESQL CONFIGURATION ====================
import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from backend.utils.cloudinary_manager import get_cloudinary_storage
    cloudinary_storage = get_cloudinary_storage()
    print("✅ Cloudinary storage initialized")
except ImportError as e:
    print(f"⚠️ Cloudinary import error: {e}")
    cloudinary_storage = None
except Exception as e:
    print(f"⚠️ Cloudinary initialization error: {e}")
    cloudinary_storage = None

# ==================== APP SETUP ====================
app = Flask(__name__, 
            static_folder=str(project_root / 'frontend'),
            static_url_path='')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))

POSTGRES_AVAILABLE = True

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')

# Try alternative paths for Render
if not os.path.exists(FRONTEND_DIR):
    FRONTEND_DIR = os.path.join(BASE_DIR, '../../frontend')
if not os.path.exists(FRONTEND_DIR):
    FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

FACE_IMAGES_DIR = os.path.join(BASE_DIR, 'face_images')
SESSION_TOKENS_DIR = os.path.join(BASE_DIR, 'session_tokens')
DATA_DIR = BASE_DIR

# Ensure directories exist
os.makedirs(FRONTEND_DIR, exist_ok=True)
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)
os.makedirs(SESSION_TOKENS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Patch for async
eventlet.monkey_patch()

# Initialize Flask app
app = Flask(__name__, 
            static_folder=FRONTEND_DIR,
            static_url_path='')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Configure for production with proxy support
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
CORS(app, 
     supports_credentials=True,
     resources={
         r"/*": {
             "origins": ["*"],
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "X-Admin-Username", "X-Admin-Token"]
         }
     })

# Initialize SocketIO for real-time alerts
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='eventlet',
                   ping_timeout=60,
                   ping_interval=25,
                   async_handlers=True,
                   logger=False,
                   engineio_logger=False)

# ==================== SECURITY CONFIGURATION ====================
# Generate bcrypt hash for admin password (admin123)
ADMIN_PASSWORD_HASH = bcrypt.hashpw(b'admin123', bcrypt.gensalt(rounds=12)).decode('utf-8')
print(f"🔐 Generated admin bcrypt hash: {ADMIN_PASSWORD_HASH[:20]}...")

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
guardian_rate_limit = {}
admin_sessions = {}

# ==================== PASSWORD HASHING FUNCTIONS ====================
def hash_password(password):
    """Hash password using bcrypt with automatic salt generation"""
    try:
        # Convert password to bytes and hash with bcrypt
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt(rounds=12)  # Generate salt with 12 rounds
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')  # Convert to string for storage
    except Exception as e:
        print(f"❌ Error hashing password with bcrypt: {e}")
        traceback.print_exc()
        raise

def verify_password(password, hashed_password):
    """Verify password against bcrypt hash"""
    try:
        # Check if the hash is valid bcrypt format (should start with $2)
        if not hashed_password or len(hashed_password) < 60:
            print(f"⚠️ Invalid hash format or length: {hashed_password[:30] if hashed_password else 'None'}")
            return False
        
        if not hashed_password.startswith('$2'):
            print(f"⚠️ Invalid bcrypt hash prefix: {hashed_password[:10]}")
            return False
        
        # Convert both to bytes
        password_bytes = password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        
        # Use bcrypt to check password
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        print(f"❌ Error verifying password with bcrypt: {e}")
        traceback.print_exc()
        return False

def verify_admin_credentials(username, password):
    """Verify admin login credentials using bcrypt"""
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
        
        ip = request.remote_addr
        if rate_limit_exceeded(ip, 'admin'):
            return jsonify({
                'success': False,
                'error': 'Too many requests'
            }), 429
        
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

# ==================== DATABASE CONNECTION MANAGEMENT ====================
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            raise Exception("DATABASE_URL not set in environment variables")
        
        # Fix URL format
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        # Parse URL
        url = urllib.parse.urlparse(database_url)
        
        # Ensure port is set
        port = url.port or 5432
        
        # Connect with timeout
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
    except psycopg2.OperationalError as e:
        print(f"❌ PostgreSQL operational error: {e}")
        raise Exception(f"Database connection failed: {str(e)}")
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
            raise Exception("DATABASE_URL not set in environment variables")
        
        # Fix URL format
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        # Parse URL
        url = urllib.parse.urlparse(database_url)
        
        # Ensure port is set
        port = url.port or 5432
        
        # Connect with timeout
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
    except psycopg2.OperationalError as e:
        if conn:
            conn.rollback()
        print(f"❌ PostgreSQL operational error: {e}")
        raise Exception(f"Database connection failed: {str(e)}")
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
db_write_lock = threading.Lock()

# ==================== DATABASE INITIALIZATION ====================
def init_db():
    """Initialize database with all required tables"""
    print("🗄️  Initializing PostgreSQL database...")
    
    try:
        with get_db_cursor() as cursor:
            # PostgreSQL schema only
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS guardians (
                    guardian_id SERIAL PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    phone TEXT UNIQUE NOT NULL,
                    email TEXT,
                    password_hash TEXT NOT NULL,
                    address TEXT,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP
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
            
            # Create indexes for PostgreSQL
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_drivers_guardian ON drivers(guardian_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_guardian ON alerts(guardian_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tokens_expires ON session_tokens(expires_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tokens_valid ON session_tokens(is_valid)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_drowsiness_driver ON drowsiness_events(driver_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_drowsiness_timestamp ON drowsiness_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_guardians_active ON guardians(is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_activity_timestamp ON activity_log(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_admin_activity_timestamp ON admin_activity_log(timestamp)')
        
        print("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        traceback.print_exc()
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
            try:
                cursor.execute('UPDATE session_tokens SET is_valid = FALSE WHERE guardian_id = %s AND is_valid = TRUE', (guardian_id,))
            except Exception as e:
                print(f"⚠️ Error invalidating existing sessions: {e}")
            
            # Create new session
            try:
                cursor.execute('''
                    INSERT INTO session_tokens (guardian_id, token, expires_at, ip_address, user_agent)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (guardian_id, token, expires_at, ip_address, user_agent))
            except Exception as e:
                print(f"❌ Error creating session: {e}")
        
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
            try:
                cursor.execute('''
                    SELECT COUNT(*) FROM session_tokens 
                    WHERE guardian_id = %s AND token = %s AND is_valid = TRUE AND expires_at > %s
                ''', (guardian_id, token, datetime.now()))
            except Exception as e:
                print(f"❌ Error validating session query: {e}")
                return False
            
            result = cursor.fetchone()
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
                try:
                    cursor.execute('UPDATE session_tokens SET is_valid = FALSE WHERE guardian_id = %s AND token = %s', (guardian_id, token))
                except Exception as e:
                    print(f"⚠️ Error invalidating specific session: {e}")
            else:
                try:
                    cursor.execute('UPDATE session_tokens SET is_valid = FALSE WHERE guardian_id = %s', (guardian_id,))
                except Exception as e:
                    print(f"⚠️ Error invalidating all sessions: {e}")
        
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
def verify_guardian_credentials(phone, password):
    """Verify guardian login credentials using bcrypt"""
    try:
        # Clean the phone number (remove spaces, dashes, parentheses)
        phone = str(phone).strip()
        phone = re.sub(r'[\s\-\(\)]', '', phone)
        
        print(f"🔍 [LOGIN VERIFY] Phone: '{phone}'")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Try exact match first
            cursor.execute('''
                SELECT guardian_id, full_name, password_hash, failed_login_attempts, locked_until
                FROM guardians 
                WHERE phone = %s AND is_active = TRUE
            ''', (phone,))
            
            result = cursor.fetchone()
            
            if result:
                guardian_id, full_name, stored_hash, failed_attempts, locked_until = result
                
                print(f"✅ [LOGIN VERIFY] User found: {full_name}")
                print(f"   Stored hash length: {len(stored_hash) if stored_hash else 0}")
                print(f"   Hash preview: {stored_hash[:30] if stored_hash else 'None'}...")
                
                # Check if account is locked
                if locked_until:
                    try:
                        # Handle both string and datetime objects
                        if isinstance(locked_until, str):
                            # Try to parse the string
                            try:
                                lock_time = datetime.fromisoformat(locked_until.replace('Z', '+00:00'))
                            except:
                                # Try alternative format
                                lock_time = datetime.strptime(locked_until, '%Y-%m-%d %H:%M:%S.%f')
                        else:
                            lock_time = locked_until
                        
                        if lock_time > datetime.now():
                            remaining_minutes = (lock_time - datetime.now()).seconds // 60
                            print(f"🔒 [LOGIN VERIFY] Account locked for {remaining_minutes} more minutes")
                            return {'error': 'Account locked', 'locked_until': locked_until}
                    except Exception as e:
                        print(f"⚠️ [LOGIN VERIFY] Error parsing lock time '{locked_until}': {e}")
                        # Continue anyway, don't lock out due to parsing error
                
                # Check if hash is valid bcrypt format
                if not stored_hash or not stored_hash.startswith('$2'):
                    print(f"❌ [LOGIN VERIFY] Invalid bcrypt hash format in database")
                    return None
                
                # Verify password using bcrypt
                print(f"   Password verification...")
                
                try:
                    password_bytes = password.encode('utf-8')
                    hashed_bytes = stored_hash.encode('utf-8')
                    
                    if bcrypt.checkpw(password_bytes, hashed_bytes):
                        print(f"✅ [LOGIN VERIFY] Password verified successfully")
                        # Reset failed attempts on successful login
                        try:
                            cursor.execute('UPDATE guardians SET failed_login_attempts = 0, last_login = %s WHERE guardian_id = %s', 
                                         (datetime.now(), guardian_id))
                        except Exception as e:
                            print(f"⚠️ [LOGIN VERIFY] Error resetting failed attempts: {e}")
                        conn.commit()
                        
                        return {
                            'guardian_id': guardian_id, 
                            'full_name': full_name, 
                            'phone': phone
                        }
                    else:
                        print(f"❌ [LOGIN VERIFY] Password verification failed")
                        # Increment failed attempts
                        try:
                            cursor.execute('UPDATE guardians SET failed_login_attempts = failed_login_attempts + 1 WHERE guardian_id = %s', 
                                         (guardian_id,))
                            conn.commit()
                        except Exception as e:
                            print(f"⚠️ Error updating failed attempts: {e}")
                        
                        return None
                        
                except Exception as hash_error:
                    print(f"❌ [LOGIN VERIFY] Hash verification error: {hash_error}")
                    return None
            else:
                print(f"❌ [LOGIN VERIFY] No user found with phone: '{phone}'")
                    
    except Exception as e:
        print(f"❌ [LOGIN VERIFY] Error: {e}")
        traceback.print_exc()
    
    return None

def get_guardian_by_id(guardian_id):
    """Get guardian information by ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    SELECT guardian_id, full_name, phone, email, address, registration_date, last_login
                    FROM guardians WHERE guardian_id = %s
                ''', (guardian_id,))
            except Exception as e:
                print(f"❌ Error getting guardian query: {e}")
                return None
            
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
                try:
                    cursor.execute('''
                        INSERT INTO admin_activity_log (admin_username, action, details, ip_address, user_agent)
                        VALUES (%s, %s, %s, %s, %s)
                    ''', (admin_username, action, details, ip_address, user_agent))
                except Exception as e:
                    print(f"⚠️ Error logging admin activity: {e}")
            else:
                try:
                    cursor.execute('''
                        INSERT INTO activity_log (guardian_id, action, details, ip_address, user_agent)
                        VALUES (%s, %s, %s, %s, %s)
                    ''', (guardian_id, action, details, ip_address, user_agent))
                except Exception as e:
                    print(f"⚠️ Error logging activity: {e}")
    except Exception as e:
        print(f"⚠️ Error logging activity: {e}")

# ==================== DEVICE DETECTION ====================
def is_mobile_device():
    """Detect if request is from mobile device"""
    user_agent = request.headers.get('User-Agent', '').lower()
    
    mobile_keywords = ['mobile', 'android', 'iphone', 'ipad', 'tablet', 'windows phone']
    
    for keyword in mobile_keywords:
        if keyword in user_agent:
            return True
    
    return False

# ==================== UTILITY FUNCTIONS ====================
def save_base64_image(base64_string, driver_id, image_number):
    """Save base64 image to Cloudinary"""
    try:
        print(f"💾 Saving face {image_number} for driver {driver_id}...")
        
        # Upload to Cloudinary
        from utils.cloudinary_manager import get_cloudinary_storage
        storage = get_cloudinary_storage()
        
        cloudinary_url = storage.upload_driver_face(
            base64_string, driver_id, image_number
        )
        
        if not cloudinary_url:
            print(f"❌ Cloudinary upload failed for driver {driver_id}")
            return None
        
        print(f"✅ Uploaded to: {cloudinary_url}")
        
        # Save to PostgreSQL
        with get_db_cursor() as cursor:
            cursor.execute('''
                INSERT INTO face_images (driver_id, image_path, cloudinary_url, image_number)
                VALUES (%s, %s, %s, %s)
            ''', (driver_id, cloudinary_url, cloudinary_url, image_number))
        
        return cloudinary_url
        
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_guardian_drivers(guardian_id):
    """Get all drivers registered by a guardian"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    SELECT d.*, 
                           (SELECT COUNT(*) FROM alerts a WHERE a.driver_id = d.driver_id AND a.acknowledged = FALSE) as alert_count,
                           (SELECT COUNT(*) FROM face_images f WHERE f.driver_id = d.driver_id) as face_count
                    FROM drivers d
                    WHERE d.guardian_id = %s AND d.is_active = TRUE
                    ORDER BY d.registration_date DESC
                ''', (guardian_id,))
            except Exception as e:
                print(f"❌ Error in get_guardian_drivers query: {e}")
                return []
            
            drivers = cursor.fetchall()
            result = []
            for driver in drivers:
                result.append(dict(driver))
            return result
    except Exception as e:
        print(f"❌ Error in get_guardian_drivers: {e}")
        return []

def get_recent_alerts(guardian_id, limit=10):
    """Get recent alerts for a guardian"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    SELECT a.*, d.name as driver_name
                    FROM alerts a
                    JOIN drivers d ON a.driver_id = d.driver_id
                    WHERE a.guardian_id = %s
                    ORDER BY a.timestamp DESC
                    LIMIT %s
                ''', (guardian_id, limit))
            except Exception as e:
                print(f"❌ Error in get_recent_alerts query: {e}")
                return []
            
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
            try:
                cursor.execute('''
                    SELECT d.*, g.full_name as guardian_name, g.phone as guardian_phone
                    FROM drivers d
                    JOIN guardians g ON d.guardian_id = g.guardian_id
                    WHERE d.driver_id = %s AND d.is_active = TRUE
                ''', (identifier,))
            except Exception as e:
                print(f"❌ Error in get_driver_by_name_or_id query (ID): {e}")
                return None
            
            result = cursor.fetchone()
            
            # If not found by ID, try by name
            if not result:
                try:
                    cursor.execute('''
                        SELECT d.*, g.full_name as guardian_name, g.phone as guardian_phone
                        FROM drivers d
                        JOIN guardians g ON d.guardian_id = g.guardian_id
                        WHERE d.name LIKE %s AND d.is_active = TRUE
                    ''', (f'%{identifier}%',))
                except Exception as e:
                    print(f"❌ Error in get_driver_by_name_or_id query (name): {e}")
                    return None
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
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "connect-src 'self' ws: wss:"
    )
    
    return response

# ==================== SOCKET.IO HANDLERS ====================
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    connected_clients[client_id] = {
        'connected_at': datetime.now(),
        'ip': request.remote_addr,
        'type': None,
        'guardian_id': None,
        'authenticated': False
    }
    emit('connected', {'status': 'connected', 'client_id': client_id})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    if client_id in connected_clients:
        del connected_clients[client_id]

@socketio.on('guardian_authenticate')
def handle_guardian_auth(data):
    """Guardian authentication via WebSocket"""
    client_id = request.sid
    guardian_id = data.get('guardian_id')
    token = data.get('token')
    
    if guardian_id and token and validate_session(guardian_id, token):
        connected_clients[client_id]['type'] = 'guardian'
        connected_clients[client_id]['guardian_id'] = guardian_id
        connected_clients[client_id]['authenticated'] = True
        
        guardian = get_guardian_by_id(guardian_id)
        if guardian:
            emit('auth_confirmed', {
                'guardian_id': guardian_id,
                'full_name': guardian['full_name'],
                'phone': guardian['phone']
            })
            return
    
    emit('auth_failed', {'error': 'Authentication failed'})
    socketio.disconnect(client_id)

def send_pending_alerts(guardian_id, client_id):
    """Send pending alerts to a newly connected guardian"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    SELECT a.*, d.name as driver_name
                    FROM alerts a
                    JOIN drivers d ON a.driver_id = d.driver_id
                    WHERE a.guardian_id = %s AND a.acknowledged = FALSE 
                    AND a.timestamp > NOW() - INTERVAL '1 hour'
                    ORDER BY a.timestamp DESC
                ''', (guardian_id,))
            except Exception as e:
                print(f"❌ Error in send_pending_alerts query: {e}")
                return
            
            alerts = cursor.fetchall()
            
            for alert in alerts:
                alert_data = dict(alert)
                if alert_data.get('detection_details'):
                    try:
                        alert_data['detection_details'] = json.loads(alert_data['detection_details'])
                    except:
                        pass
                
                socketio.emit('guardian_alert', alert_data, room=client_id)
                
    except Exception as e:
        print(f"❌ Error sending pending alerts: {e}")

# ==================== MAIN ROUTES ====================
@app.route('/')
def serve_home():
    """Main route - Redirects to admin login for desktop, mobile login for mobile"""
    # Check for logout parameter
    logged_out = request.args.get('logged_out')
    
    if request.args.get('force') == 'mobile':
        return send_from_directory(FRONTEND_DIR, 'login.html')
    if request.args.get('force') == 'desktop':
        return send_from_directory(FRONTEND_DIR, 'admin_login.html')
    
    if is_mobile_device():
        # If logged out, pass the parameter
        if logged_out:
            return redirect('/login.html?logged_out=true')
        return send_from_directory(FRONTEND_DIR, 'login.html')
    else:
        return send_from_directory(FRONTEND_DIR, 'admin_login.html')

@app.route('/admin_login')
def serve_admin_login():
    """Admin login page"""
    return send_from_directory(FRONTEND_DIR, 'admin_login.html')

@app.route('/admin')
def serve_admin_dashboard():
    """Admin dashboard page"""
    return send_from_directory(FRONTEND_DIR, 'admin_login.html')

@app.route('/guardian-register')
def serve_guardian_register():
    """Guardian registration page"""
    return send_from_directory(FRONTEND_DIR, 'guardian-register.html')

@app.route('/guardian-dashboard')
def serve_guardian_dashboard():
    """Guardian dashboard page"""
    token = request.args.get('token')
    guardian_id = request.args.get('guardian_id')
    
    if token and guardian_id and validate_session(guardian_id, token):
        response = send_from_directory(FRONTEND_DIR, 'guardian-dashboard.html')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    else:
        return redirect('/?logged_out=true')

@app.route('/register-driver')
def serve_register_driver():
    """Driver registration page"""
    token = request.args.get('token')
    guardian_id = request.args.get('guardian_id')
    
    if token and guardian_id and validate_session(guardian_id, token):
        response = send_from_directory(FRONTEND_DIR, 'register-driver.html')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        return response
    
    return redirect('/?logged_out=true')

# ==================== PWA ROUTES ====================
@app.route('/manifest.json')
def serve_manifest():
    try:
        response = send_from_directory(FRONTEND_DIR, 'manifest.json')
        response.headers['Content-Type'] = 'application/manifest+json'
        return response
    except:
        # Return a basic manifest if file doesn't exist
        return jsonify({
            "name": "Driver Alert System",
            "short_name": "DriverAlert",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#ffffff",
            "theme_color": "#007bff",
            "icons": [
                {
                    "src": "/icon-192.png",
                    "sizes": "192x192",
                    "type": "image/png"
                },
                {
                    "src": "/icon-512.png",
                    "sizes": "512x512",
                    "type": "image/png"
                }
            ]
        })

@app.route('/service-worker.js')
def serve_service_worker():
    try:
        response = send_from_directory(FRONTEND_DIR, 'service-worker.js')
        response.headers['Content-Type'] = 'application/javascript'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
    except:
        # Return a basic service worker if file doesn't exist
        return Response('''
            self.addEventListener('install', function(event) {
                event.waitUntil(self.skipWaiting());
            });
            self.addEventListener('activate', function(event) {
                event.waitUntil(self.clients.claim());
            });
        ''', mimetype='application/javascript')

@app.route('/icon-<int:size>.png')
def serve_icon(size):
    icon_filename = f'icon-{size}.png'
    icon_path = os.path.join(FRONTEND_DIR, icon_filename)
    
    if os.path.exists(icon_path):
        return send_from_directory(FRONTEND_DIR, icon_filename)
    
    # If icon doesn't exist, serve a default icon or 404
    default_icon = os.path.join(FRONTEND_DIR, 'icon-192.png')
    if os.path.exists(default_icon):
        return send_from_directory(FRONTEND_DIR, 'icon-192.png')
    
    # If no icons exist, return a 404 with a helpful message
    return jsonify({'error': 'Icon not found'}), 404

# ==================== API ENDPOINTS ====================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint - FIXED VERSION"""
    try:
        # Basic health info
        health_info = {
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'server': 'Driver Alert System',
            'version': '2.0.0',
            'environment': 'production',
            'connected_clients': len(connected_clients),
            'active_sessions': len(active_sessions),
            'admin_sessions': len(admin_sessions),
            'database': 'postgresql',
        }
        
        # Try to get database info
        try:
            database_url = os.environ.get('DATABASE_URL')
            if not database_url:
                health_info.update({
                    'database_status': 'disconnected',
                    'database_error': 'DATABASE_URL not set'
                })
            else:
                # Parse URL to get info
                url = urllib.parse.urlparse(database_url)
                health_info.update({
                    'database_host': url.hostname,
                    'database_port': url.port or 5432,
                    'database_name': url.path[1:] if url.path.startswith('/') else url.path
                })
                
                # Use a simpler connection method for health check
                try:
                    # Fix URL format
                    if database_url.startswith('postgres://'):
                        database_url = database_url.replace('postgres://', 'postgresql://', 1)
                    
                    # Connect without RealDictCursor for health check
                    conn = psycopg2.connect(
                        database=url.path[1:] if url.path.startswith('/') else url.path,
                        user=url.username,
                        password=url.password,
                        host=url.hostname,
                        port=url.port or 5432,
                        connect_timeout=5
                    )
                    
                    cursor = conn.cursor()
                    
                    # Get database version
                    cursor.execute('SELECT version()')
                    db_version = cursor.fetchone()[0]
                    
                    # Get counts - handle empty tables
                    counts = {}
                    
                    # Guardians count
                    try:
                        cursor.execute('SELECT COUNT(*) as count FROM guardians')
                        counts['guardians'] = cursor.fetchone()[0]
                    except:
                        counts['guardians'] = 0
                    
                    # Drivers count
                    try:
                        cursor.execute('SELECT COUNT(*) as count FROM drivers')
                        counts['drivers'] = cursor.fetchone()[0]
                    except:
                        counts['drivers'] = 0
                    
                    # Alerts count
                    try:
                        cursor.execute('SELECT COUNT(*) as count FROM alerts')
                        counts['alerts'] = cursor.fetchone()[0]
                    except:
                        counts['alerts'] = 0
                    
                    cursor.close()
                    conn.close()
                    
                    health_info.update({
                        'database_status': 'connected',
                        'database_version': db_version,
                        'statistics': counts
                    })
                    
                except Exception as conn_error:
                    health_info.update({
                        'database_status': 'disconnected',
                        'database_error': f'Connection failed: {str(conn_error)}'
                    })
                    
        except Exception as db_error:
            health_info.update({
                'database_status': 'disconnected',
                'database_error': f'Setup error: {str(db_error)}'
            })
        
        return jsonify(health_info)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'running_with_errors',
            'error': str(e)
        }), 200

@app.route('/api/debug/db', methods=['GET'])
def debug_database():
    """Debug endpoint to check database connection details"""
    try:
        # Get DATABASE_URL
        database_url = os.environ.get('DATABASE_URL')
        
        if not database_url:
            return jsonify({
                'success': False,
                'error': 'DATABASE_URL not found in environment',
                'environment_keys': [k for k in os.environ.keys() if 'DATABASE' in k.upper() or 'POSTGRES' in k.upper()]
            })
        
        # Fix URL format
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        # Parse URL
        url = urllib.parse.urlparse(database_url)
        
        # Mask password for security
        masked_url = f"{url.scheme}://{url.username}:****@{url.hostname}:{url.port or 5432}{url.path}"
        
        # Try connection
        try:
            conn = psycopg2.connect(
                database=url.path[1:] if url.path.startswith('/') else url.path,
                user=url.username,
                password=url.password,
                host=url.hostname,
                port=url.port or 5432,
                connect_timeout=5
            )
            
            cursor = conn.cursor()
            cursor.execute('SELECT version()')
            version = cursor.fetchone()[0]
            
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
            tables = [table[0] for table in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            return jsonify({
                'success': True,
                'connection': 'successful',
                'database_url_masked': masked_url,
                'database_version': version,
                'tables': tables,
                'tables_count': len(tables),
                'parsed_url': {
                    'scheme': url.scheme,
                    'username': url.username,
                    'hostname': url.hostname,
                    'port': url.port or 5432,
                    'database': url.path[1:] if url.path.startswith('/') else url.path,
                    'has_password': bool(url.password)
                }
            })
            
        except Exception as conn_error:
            return jsonify({
                'success': False,
                'connection': 'failed',
                'database_url_masked': masked_url,
                'error': str(conn_error),
                'parsed_url': {
                    'scheme': url.scheme,
                    'username': url.username,
                    'hostname': url.hostname,
                    'port': url.port or 5432,
                    'database': url.path[1:] if url.path.startswith('/') else url.path,
                    'has_password': bool(url.password)
                }
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'database_url': database_url[:50] + '...' if database_url and len(database_url) > 50 else database_url
        })

@app.route('/api/test-connection', methods=['GET'])
def test_connection():
    """Simple connection test"""
    database_url = os.environ.get('DATABASE_URL')
    
    if not database_url:
        return jsonify({'success': False, 'error': 'No DATABASE_URL found'})
    
    try:
        # Fix URL if needed
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        # Parse
        url = urllib.parse.urlparse(database_url)
        
        # Connect
        conn = psycopg2.connect(
            dbname=url.path[1:] if url.path.startswith('/') else url.path,
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port or 5432,
            connect_timeout=5
        )
        
        cursor = conn.cursor()
        cursor.execute('SELECT 1 as test_value')
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'connected': True,
            'test_query': result[0] == 1,
            'database_info': {
                'host': url.hostname,
                'port': url.port or 5432,
                'database': url.path[1:] if url.path.startswith('/') else url.path
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'database_url_preview': database_url[:50] + '...' if len(database_url) > 50 else database_url
        })

@app.route('/api/test-bcrypt', methods=['GET'])
def test_bcrypt():
    """Test bcrypt functionality"""
    try:
        # Test bcrypt
        test_password = "test123"
        
        # Generate hash
        password_hash = hash_password(test_password)
        
        # Verify hash
        is_valid = verify_password(test_password, password_hash)
        
        # Test with wrong password
        is_wrong = verify_password("wrong", password_hash)
        
        # Test admin password
        admin_valid = verify_password("admin123", ADMIN_PASSWORD_HASH)
        
        return jsonify({
            'success': True,
            'test_password': test_password,
            'hash_generated': password_hash[:50] + "..." if len(password_hash) > 50 else password_hash,
            'hash_length': len(password_hash),
            'hash_starts_with': password_hash[:10],
            'hash_format_valid': password_hash.startswith('$2'),
            'correct_password_matches': is_valid,
            'wrong_password_matches': is_wrong,
            'admin_password_valid': admin_valid,
            'note': 'bcrypt hash should start with $2b$12$ and be ~60 chars long'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

# ==================== ADMIN AUTHENTICATION ENDPOINTS ====================
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
            
            print(f"👑 Admin logged in: {admin['full_name']} ({username})")
            
            return jsonify({
                'success': True,
                'username': username,
                'full_name': admin['full_name'],
                'role': admin['role'],
                'email': admin['email'],
                'token': token,
                'expires': expires_at.isoformat(),
                'message': 'Admin login successful'
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

@app.route('/api/admin/validate', methods=['POST'])
def admin_validate():
    """Validate admin session"""
    try:
        data = request.json
        username = data.get('username')
        token = data.get('token')
        
        if not username or not token:
            return jsonify({
                'success': False,
                'valid': False,
                'error': 'Missing authentication data'
            }), 400
        
        is_valid = validate_admin_token(username, token)
        
        if is_valid:
            admin_info = ADMIN_CREDENTIALS.get(username, {})
            return jsonify({
                'success': True,
                'valid': True,
                'username': username,
                'full_name': admin_info.get('full_name', 'Admin'),
                'role': admin_info.get('role', 'admin'),
                'expires': admin_sessions[username]['expires'].isoformat() if username in admin_sessions else None,
                'message': 'Admin session valid'
            })
        else:
            return jsonify({
                'success': False,
                'valid': False,
                'error': 'Invalid or expired admin session'
            }), 401
        
    except Exception as e:
        return jsonify({
            'success': False,
            'valid': False,
            'error': str(e)
        }), 500

@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    """Admin logout"""
    try:
        data = request.json
        username = data.get('username')
        token = data.get('token')
        
        if not username or not token:
            return jsonify({
                'success': False,
                'error': 'Missing authentication data'
            }), 400
        
        # Validate before logout
        if not validate_admin_token(username, token):
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 401
        
        # Remove session
        if username in admin_sessions:
            del admin_sessions[username]
        
        # Log activity
        log_activity(admin_username=username, action='ADMIN_LOGOUT', 
                    details=f'Admin logged out')
        
        print(f"👑 Admin logged out: {username}")
        
        return jsonify({
            'success': True,
            'message': 'Admin logged out successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/admin/extend-session', methods=['POST'])
def admin_extend_session():
    """Extend admin session"""
    try:
        data = request.json
        username = data.get('username')
        token = data.get('token')
        
        if not username or not token:
            return jsonify({
                'success': False,
                'error': 'Missing authentication data'
            }), 400
        
        if not validate_admin_token(username, token):
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 401
        
        # Extend session by 1 hour
        if username in admin_sessions:
            admin_sessions[username]['expires'] = datetime.now() + timedelta(hours=1)
            admin_sessions[username]['last_activity'] = datetime.now()
        
        return jsonify({
            'success': True,
            'expires': admin_sessions[username]['expires'].isoformat(),
            'message': 'Session extended successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== ADMIN MANAGEMENT ENDPOINTS ====================
@app.route('/api/admin/db-drivers', methods=['GET'])
@require_admin_auth
def admin_get_drivers():
    """Get all drivers from database - ADMIN ONLY"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    SELECT d.*, g.full_name as guardian_name, g.phone as guardian_phone
                    FROM drivers d
                    LEFT JOIN guardians g ON d.guardian_id = g.guardian_id
                    ORDER BY d.registration_date DESC
                ''')
            except Exception as e:
                print(f"Query error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
            
            drivers = cursor.fetchall()
            drivers_list = []
            for driver in drivers:
                drivers_list.append(dict(driver))
            
            try:
                cursor.execute('SELECT COUNT(*) as count FROM drivers')
                count = cursor.fetchone()[0]
            except Exception as e:
                print(f"❌ Error getting counts: {e}")
                count = len(drivers_list)
            
        return jsonify({
            'success': True,
            'count': count,
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
            
            try:
                cursor.execute('''
                    SELECT a.*, d.name as driver_name, g.full_name as guardian_name
                    FROM alerts a
                    JOIN drivers d ON a.driver_id = d.driver_id
                    JOIN guardians g ON a.guardian_id = g.guardian_id
                    ORDER BY a.timestamp DESC
                    LIMIT %s
                ''', (limit,))
            except Exception as e:
                print(f"❌ Error in admin_get_alerts query: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
            
            alerts = cursor.fetchall()
            
            # Parse detection details
            result = []
            for alert in alerts:
                alert_dict = dict(alert)
                if alert_dict.get('detection_details'):
                    try:
                        alert_dict['detection_details'] = json.loads(alert_dict['detection_details'])
                    except:
                        pass
                result.append(alert_dict)
            
            try:
                cursor.execute('SELECT COUNT(*) as count FROM alerts')
                count = cursor.fetchone()[0]
            except Exception as e:
                print(f"❌ Error getting alert counts: {e}")
                count = len(alerts)
            
        return jsonify({
            'success': True,
            'count': count,
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
            try:
                cursor.execute('''
                    SELECT g.*, 
                           (SELECT COUNT(*) FROM drivers d WHERE d.guardian_id = g.guardian_id) as driver_count,
                           (SELECT COUNT(*) FROM alerts a WHERE a.guardian_id = g.guardian_id) as alert_count
                    FROM guardians g
                    ORDER BY g.registration_date DESC
                ''')
            except Exception as e:
                print(f"❌ Error in admin_get_guardians query: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
            
            guardians = cursor.fetchall()
            guardians_list = []
            for guardian in guardians:
                guardians_list.append(dict(guardian))
            
        return jsonify({
            'success': True,
            'count': len(guardians_list),
            'guardians': guardians_list
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/db-tables', methods=['GET'])
@require_admin_auth
def admin_get_tables():
    """Get all table names and row counts - ADMIN ONLY"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """)
            
            tables = cursor.fetchall()
            
            result = []
            for table in tables:
                table_name = table[0]
                
                try:
                    cursor.execute(f'SELECT COUNT(*) as count FROM {table_name}')
                    count = cursor.fetchone()[0]
                except:
                    count = 0
                
                result.append({
                    'table_name': table_name,
                    'row_count': count
                })
            
        return jsonify({
            'success': True,
            'tables': result
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
            
            # Get comprehensive statistics
            stats = {}
            try:
                cursor.execute('SELECT COUNT(*) as total_alerts FROM alerts')
                stats['total_alerts'] = cursor.fetchone()[0]
            except:
                stats['total_alerts'] = 0
            
            try:
                cursor.execute('SELECT COUNT(*) as total_drivers FROM drivers')
                stats['total_drivers'] = cursor.fetchone()[0]
            except:
                stats['total_drivers'] = 0
            
            try:
                cursor.execute('SELECT COUNT(*) as total_guardians FROM guardians')
                stats['total_guardians'] = cursor.fetchone()[0]
            except:
                stats['total_guardians'] = 0
            
        return jsonify({
            'success': True,
            'statistics': stats,
            'system_status': {
                'database': 'postgresql',
                'connected_clients': len(connected_clients),
                'admin_sessions': len(admin_sessions),
                'server_time': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/admin/clear-alerts', methods=['POST'])
@require_admin_auth
def admin_clear_alerts():
    """Clear old alerts - ADMIN ONLY"""
    try:
        days = request.json.get('days', 30)
        
        with get_db_cursor() as cursor:
            try:
                cursor.execute('''
                    DELETE FROM alerts 
                    WHERE acknowledged = TRUE AND timestamp < CURRENT_DATE - INTERVAL %s days
                ''', (days,))
            except Exception as e:
                print(f"❌ Error clearing alerts: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
            
            deleted_count = cursor.rowcount
            
            # Log activity
            username = request.headers.get('X-Admin-Username')
            log_activity(admin_username=username, action='CLEAR_ALERTS', 
                        details=f'Cleared {deleted_count} alerts older than {days} days')
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Cleared {deleted_count} alerts older than {days} days'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== GUARDIAN AUTHENTICATION ENDPOINTS ====================
@app.route('/api/login', methods=['POST'])
def login():
    """Login guardian with bcrypt verification"""
    try:
        data = request.json
        
        if not data or 'phone' not in data or 'password' not in data:
            return jsonify({'success': False, 'error': 'Missing credentials'}), 400
        
        phone = data['phone'].strip()
        password = data['password']
        
        # Clean phone number for query
        phone_clean = re.sub(r'[\s\-\(\)]', '', phone)
        
        with get_db_cursor() as cursor:
            cursor.execute('''
                SELECT guardian_id, full_name, password_hash, is_active
                FROM guardians 
                WHERE phone = %s
            ''', (phone_clean,))
            
            guardian = cursor.fetchone()
            
            if not guardian:
                return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
            
            guardian_id, full_name, password_hash, is_active = guardian
            
            # Check if account is active
            if not is_active:
                return jsonify({'success': False, 'error': 'Account is deactivated'}), 403
            
            # IMPORTANT: Convert stored hash back to bytes for bcrypt
            # password_hash is stored as VARCHAR string, need bytes for verification
            try:
                # If it's already bytes
                if isinstance(password_hash, bytes):
                    hash_bytes = password_hash
                else:
                    # If it's string (should be after our VARCHAR change)
                    hash_bytes = password_hash.encode('utf-8')
                
                # Verify password
                if bcrypt.checkpw(password.encode('utf-8'), hash_bytes):
                    # Update last login
                    cursor.execute('''
                        UPDATE guardians 
                        SET last_login = %s, failed_login_attempts = 0
                        WHERE guardian_id = %s
                    ''', (datetime.now(), guardian_id))
                    
                    # Create session
                    token = create_session(guardian_id, request.remote_addr,
                                         request.headers.get('User-Agent'))
                    
                    if token:
                        return jsonify({
                            'success': True,
                            'guardian_id': guardian_id,
                            'full_name': full_name,
                            'phone': phone_clean,
                            'session_token': token,
                            'message': 'Login successful'
                        })
                    else:
                        return jsonify({
                            'success': True,
                            'guardian_id': guardian_id,
                            'full_name': full_name,
                            'phone': phone_clean,
                            'message': 'Login successful (no session)'
                        })
                else:
                    # Increment failed attempts
                    cursor.execute('''
                        UPDATE guardians 
                        SET failed_login_attempts = failed_login_attempts + 1
                        WHERE guardian_id = %s
                    ''', (guardian_id,))
                    return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
                    
            except Exception as hash_error:
                print(f"❌ Password verification error: {hash_error}")
                return jsonify({'success': False, 'error': 'Authentication error'}), 500
                
    except Exception as e:
        print(f"❌ Login error: {e}")
        return jsonify({'success': False, 'error': 'Login failed'}), 500

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
        
        response = jsonify({
            'success': True,
            'message': 'Logged out successfully',
            'redirect_required': True,
            'redirect_url': '/?logged_out=true'
        })
        
        # Set headers to prevent caching
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        
        return response
        
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
    """Register a new guardian - FIXED VERSION with better debugging"""
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
        
        print(f"🔍 [REGISTRATION] Processing: Name='{full_name}', Phone='{phone}'")
        
        # Validate password length
        if len(password) < 6:
            print(f"❌ [REGISTRATION] Password too short: {len(password)} chars")
            return jsonify({
                'success': False,
                'error': 'Password must be at least 6 characters long'
            }), 400
        
        # Clean phone number
        import re
        phone_clean = re.sub(r'[\s\-\(\)]', '', phone)
        print(f"🔍 [REGISTRATION] Cleaned phone: '{phone_clean}'")
        
        # Validate Philippine phone number
        # Accept: 09XXXXXXXXX (11 digits), +639XXXXXXXXX (13 digits), 639XXXXXXXXX (12 digits)
        validation_phone = phone_clean.replace('+', '') if phone_clean.startswith('+') else phone_clean
        
        if not (validation_phone.startswith('09') or validation_phone.startswith('639')):
            print(f"❌ [REGISTRATION] Invalid phone prefix: {validation_phone[:3]}")
            return jsonify({
                'success': False,
                'error': 'Invalid Philippine phone number. Must start with 09 or 639'
            }), 400
        
        # Convert 09 to 639 format for consistent storage
        if validation_phone.startswith('09'):
            validation_phone = '63' + validation_phone[1:]
        
        # Final validation
        if len(validation_phone) != 12:
            print(f"❌ [REGISTRATION] Invalid phone length: {len(validation_phone)}")
            return jsonify({
                'success': False,
                'error': f'Phone number must be 11 digits (09 format) or 13 digits with +639'
            }), 400
        
        if not validation_phone.isdigit():
            print(f"❌ [REGISTRATION] Phone contains non-digits: {validation_phone}")
            return jsonify({
                'success': False,
                'error': 'Phone number can only contain digits'
            }), 400
        
        # Store in 639XXXXXXXXX format (12 digits)
        final_phone = validation_phone
        
        print(f"🔍 [REGISTRATION] Final phone to store: {final_phone}")
        
        # Get database connection
        from your_database_module import get_db_cursor  # Import your actual DB module
        
        with get_db_cursor(commit=True) as cursor:
            # Check if phone already exists
            cursor.execute('SELECT guardian_id FROM guardians WHERE phone = %s', (final_phone,))
            existing = cursor.fetchone()
            
            if existing:
                print(f"❌ [REGISTRATION] Phone already exists: {final_phone}")
                return jsonify({
                    'success': False,
                    'error': 'Phone number already registered'
                }), 409
            
            # Generate bcrypt hash
            import bcrypt
            password_bytes = password.encode('utf-8')
            
            try:
                salt = bcrypt.gensalt(rounds=10)  # Use lower rounds for testing
                password_hash_bytes = bcrypt.hashpw(password_bytes, salt)
                password_hash = password_hash_bytes.decode('utf-8')
                print(f"✅ [REGISTRATION] Hash generated successfully, length: {len(password_hash)}")
            except Exception as hash_error:
                print(f"❌ [REGISTRATION] Hash generation failed: {hash_error}")
                return jsonify({
                    'success': False,
                    'error': 'Password processing failed'
                }), 500
            
            # Insert into database
            try:
                cursor.execute('''
                    INSERT INTO guardians (
                        full_name, 
                        phone, 
                        email, 
                        password_hash, 
                        address, 
                        registration_date,
                        last_login,
                        is_active
                    )
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE)
                ''', (
                    full_name,
                    final_phone,
                    data.get('email', ''),
                    password_hash,
                    data.get('address', '')
                ))
                
                print(f"✅ [REGISTRATION] Database insert successful")
                
            except Exception as db_error:
                print(f"❌ [REGISTRATION] Database error: {db_error}")
                import traceback
                traceback.print_exc()
                
                # Check for specific database errors
                if 'unique constraint' in str(db_error).lower():
                    return jsonify({
                        'success': False,
                        'error': 'Phone number already registered'
                    }), 409
                elif 'password_hash' in str(db_error):
                    return jsonify({
                        'success': False,
                        'error': 'System error with password storage'
                    }), 500
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Database error during registration'
                    }), 500
            
            # Get the new guardian ID
            try:
                cursor.execute('SELECT guardian_id FROM guardians WHERE phone = %s', (final_phone,))
                result = cursor.fetchone()
                guardian_id = result[0] if result else None
                
                print(f"✅ [REGISTRATION] Guardian created with ID: {guardian_id}")
                
            except Exception as id_error:
                print(f"⚠️ [REGISTRATION] Could not get guardian ID: {id_error}")
                guardian_id = None
        
        # Success response
        response_data = {
            'success': True,
            'guardian_id': guardian_id,
            'full_name': full_name,
            'phone': final_phone,
            'email': data.get('email', ''),
            'message': 'Registration successful! You can now login.'
        }
        
        print(f"✅ [REGISTRATION] Registration complete, returning success")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ [REGISTRATION] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Registration failed. Please try again.'
        }), 500

@app.route('/api/test-db', methods=['GET'])
def test_database():
    """Test database connection"""
    try:
        with get_db_cursor() as cursor:
            cursor.execute('SELECT 1 as test')
            result = cursor.fetchone()
            return jsonify({
                'success': True,
                'database': 'Connected',
                'test': result[0] if result else None
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/check-guardians', methods=['GET'])
def check_guardians():
    """Check if guardians table exists and has data"""
    try:
        with get_db_cursor() as cursor:
            # Check table structure
            cursor.execute('''
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = 'guardians'
                ORDER BY ordinal_position
            ''')
            columns = cursor.fetchall()
            
            # Check data
            cursor.execute('SELECT COUNT(*) FROM guardians')
            count = cursor.fetchone()[0]
            
            return jsonify({
                'success': True,
                'table_exists': len(columns) > 0,
                'columns': [{'name': c[0], 'type': c[1], 'max_length': c[2]} for c in columns],
                'guardian_count': count
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
# ==================== GUARDIAN DASHBOARD ENDPOINTS ====================
@app.route('/api/guardian/dashboard', methods=['GET'])
def guardian_dashboard():
    """Get guardian dashboard data"""
    try:
        guardian_id = request.args.get('guardian_id')
        token = request.args.get('token')
        
        if not guardian_id or not token:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'redirect': '/?logged_out=true'
            }), 401
        
        # Validate session
        if not validate_session(guardian_id, token):
            return jsonify({
                'success': False,
                'error': 'Session expired or invalid',
                'redirect': '/?logged_out=true'
            }), 401
        
        guardian = get_guardian_by_id(guardian_id)
        if not guardian:
            return jsonify({
                'success': False,
                'error': 'Guardian not found',
                'redirect': '/?logged_out=true'
            }), 404
        
        drivers = get_guardian_drivers(guardian_id)
        recent_alerts = get_recent_alerts(guardian_id, 5)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get counts
            try:
                cursor.execute('SELECT COUNT(*) as count FROM drivers WHERE guardian_id = %s', (guardian_id,))
                driver_count = cursor.fetchone()[0]
            except:
                driver_count = 0
            
            try:
                cursor.execute('SELECT COUNT(*) as count FROM alerts WHERE guardian_id = %s', (guardian_id,))
                total_alerts = cursor.fetchone()[0]
            except:
                total_alerts = 0
            
            try:
                cursor.execute('SELECT COUNT(*) as count FROM alerts WHERE guardian_id = %s AND acknowledged = FALSE', 
                             (guardian_id,))
                unread_alerts = cursor.fetchone()[0]
            except:
                unread_alerts = 0
            
            try:
                cursor.execute('''
                    SELECT COUNT(*) as count FROM alerts 
                    WHERE guardian_id = %s AND DATE(timestamp) = CURRENT_DATE
                ''', (guardian_id,))
                today_alerts = cursor.fetchone()[0]
            except:
                today_alerts = 0
        
        return jsonify({
            'success': True,
            'guardian': guardian,
            'session_valid': True,
            'dashboard': {
                'driver_count': driver_count,
                'total_alerts': total_alerts,
                'unread_alerts': unread_alerts,
                'today_alerts': today_alerts,
                'recent_alerts': recent_alerts
            },
            'drivers': drivers
        })
        
    except Exception as e:
        print(f"❌ Error in guardian_dashboard: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'redirect': '/?logged_out=true'
        }), 500

@app.route('/api/register-driver', methods=['POST'])
def register_driver():
    """Register a new driver (by guardian)"""
    try:
        data = request.json
        
        required = ['driver_name', 'driver_phone', 'guardian_id', 'token']
        for field in required:
            if not data.get(field):
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Validate session
        if not validate_session(data['guardian_id'], data['token']):
            return jsonify({
                'success': False,
                'error': 'Session expired or invalid',
                'redirect': '/?logged_out=true'
            }), 401
        
        face_images = data.get('face_images', [])
        if len(face_images) < 3:
            return jsonify({
                'success': False,
                'error': 'At least 3 face images are required'
            }), 400
        
        driver_id = f"DRV{uuid.uuid4().hex[:8].upper()}"
        reference_num = data.get('reference_number', 
                                f"REF{datetime.now().strftime('%Y%m%d%H%M%S')}")
        
        # Database transaction for driver registration
        with get_db_cursor() as cursor:
            try:
                cursor.execute('''
                    INSERT INTO drivers (driver_id, name, phone, email, address, 
                                        reference_number, license_number, guardian_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    driver_id,
                    data['driver_name'],
                    data['driver_phone'],
                    data.get('driver_email', ''),
                    data.get('driver_address', ''),
                    reference_num,
                    data.get('license_number', ''),
                    data['guardian_id']
                ))
            except Exception as e:
                print(f"❌ Error inserting driver: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Driver registration failed'
                }), 500
        
        # Save images
        saved_images = []
        for i, image_data in enumerate(face_images[:3]):
            try:
                saved_path = save_base64_image(image_data, driver_id, i+1)
                if saved_path:
                    saved_images.append(saved_path)
            except Exception as e:
                print(f"⚠️ Error saving image {i+1}: {e}")
        
        # Log activity
        log_activity(data['guardian_id'], 'REGISTER_DRIVER', 
                    f'Registered driver: {data["driver_name"]} (ID: {driver_id})')
        
        # Emit socket event
        socketio.emit('driver_registered', {
            'driver_id': driver_id,
            'driver_name': data['driver_name'],
            'guardian_id': data['guardian_id'],
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'session_valid': True,
            'driver_id': driver_id,
            'reference_number': reference_num,
            'driver_name': data['driver_name'],
            'images_saved': len(saved_images),
            'message': 'Driver registered successfully'
        })
        
    except Exception as e:
        print(f"❌ Driver registration error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'redirect': '/?logged_out=true'
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
        
        print(f"📥 Received alert: Driver={driver_id}, State={detection_details.get('state', 'unknown')}, Conf={confidence:.2%}")
        
        # Try to find the driver in the database
        driver_info = get_driver_by_name_or_id(driver_id)
        
        if driver_info:
            # Driver found in database
            driver_id = driver_info['driver_id']
            driver_name = driver_info['name']
            guardian_id = driver_info['guardian_id']
            guardian_name = driver_info['guardian_name']
            guardian_phone = driver_info['guardian_phone']
            
            print(f"✅ Found driver in DB: {driver_name} -> Guardian: {guardian_name}")
            
        else:
            # Driver not found - create a temporary record
            print(f"⚠️ Driver not found: {driver_id}. Creating temp record.")
            
            # Find any guardian
            with get_db_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute('SELECT guardian_id, full_name, phone FROM guardians LIMIT 1')
                    guardian_result = cursor.fetchone()
                    
                    if guardian_result:
                        guardian_id, guardian_name, guardian_phone = guardian_result['guardian_id'], guardian_result['full_name'], guardian_result['phone']
                        
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
                except Exception as e:
                    print(f"❌ Error finding guardian: {e}")
                    return jsonify({
                        'success': False,
                        'error': 'System error'
                    }), 500
        
        # Convert detection_details to JSON string for storage
        detection_details_json = json.dumps(detection_details) if detection_details else None
        
        with get_db_cursor() as cursor:
            # Create alert
            try:
                cursor.execute('''
                    INSERT INTO alerts (driver_id, guardian_id, severity, message, detection_details, source)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (driver_id, guardian_id, severity, message, detection_details_json, 'drowsiness_detection'))
            except Exception as e:
                print(f"❌ Error creating alert: {e}")
                return jsonify({'success': False, 'error': 'Failed to create alert'}), 500
            
            try:
                cursor.execute('SELECT LASTVAL()')
                alert_id = cursor.fetchone()[0]
            except:
                alert_id = None
            
            # Log drowsiness event with detailed metrics
            try:
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
            except Exception as e:
                print(f"⚠️ Error logging drowsiness event: {e}")
            
            # Log activity
            try:
                cursor.execute('''
                    INSERT INTO activity_log (guardian_id, action, details)
                    VALUES (%s, %s, %s)
                ''', (guardian_id, 'ALERT_GENERATED', 
                    f'Alert for driver {driver_name}: {message} (Confidence: {confidence:.1%})'))
            except Exception as e:
                print(f"⚠️ Error logging activity: {e}")
            
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
            guardian_clients = []
            for client_id, client_info in connected_clients.items():
                if client_info.get('guardian_id') == guardian_id and client_info.get('authenticated'):
                    socketio.emit('guardian_alert', alert_data, room=client_id)
                    guardian_clients.append(client_id)
            
            print(f"🚨 Alert #{alert_id} sent for {driver_name}")
            print(f"   → Guardian: {guardian_name} (Phone: {guardian_phone})")
            print(f"   → Connected clients: {len(guardian_clients)}")
            
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
                'error': 'Session expired or invalid',
                'redirect': '/?logged_out=true'
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
            
            try:
                cursor.execute(base_query, tuple(params))
            except Exception as e:
                print(f"❌ Error in get_guardian_alerts query: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
            
            alerts = cursor.fetchall()
            
            # Parse detection details
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
            'error': str(e),
            'redirect': '/?logged_out=true'
        }), 500

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
                'error': 'Session expired or invalid',
                'redirect': '/?logged_out=true'
            }), 401
        
        with get_db_cursor() as cursor:
            # Check if alert belongs to this guardian
            try:
                cursor.execute('''
                    SELECT a.*, d.name as driver_name 
                    FROM alerts a
                    JOIN drivers d ON a.driver_id = d.driver_id
                    WHERE a.alert_id = %s AND a.guardian_id = %s
                ''', (alert_id, guardian_id))
            except Exception as e:
                print(f"❌ Error checking alert ownership: {e}")
                return jsonify({'success': False, 'error': 'Database error'}), 500
            
            alert_result = cursor.fetchone()
            if not alert_result:
                return jsonify({
                    'success': False,
                    'error': 'Alert not found or not authorized'
                }), 404
            
            alert = dict(alert_result)
            
            # Acknowledge the alert
            try:
                cursor.execute('''
                    UPDATE alerts SET acknowledged = TRUE 
                    WHERE alert_id = %s AND guardian_id = %s
                ''', (alert_id, guardian_id))
            except Exception as e:
                print(f"❌ Error acknowledging alert: {e}")
                return jsonify({'success': False, 'error': 'Failed to acknowledge alert'}), 500
            
            # Log activity
            try:
                cursor.execute('''
                    INSERT INTO activity_log (guardian_id, action, details)
                    VALUES (%s, %s, %s)
                ''', (guardian_id, 'ALERT_ACKNOWLEDGED', 
                    f'Acknowledged alert #{alert_id} for driver {alert["driver_name"]}'))
            except Exception as e:
                print(f"⚠️ Error logging activity: {e}")
            
            # Emit socket event for real-time update
            socketio.emit('alert_acknowledged', {
                'alert_id': alert_id,
                'guardian_id': guardian_id,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"✅ Alert #{alert_id} acknowledged by guardian {guardian_id}")
            
            return jsonify({
                'success': True,
                'alert_id': alert_id,
                'acknowledged': True,
                'message': 'Alert acknowledged successfully'
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'redirect': '/?logged_out=true'
        }), 500

@app.route('/api/guardian/drowsiness-events', methods=['GET'])
def get_drowsiness_events():
    """Get drowsiness events for a guardian"""
    try:
        guardian_id = request.args.get('guardian_id')
        token = request.args.get('token')
        limit = request.args.get('limit', 50, type=int)
        driver_id = request.args.get('driver_id')
        
        if not guardian_id or not token:
            return jsonify({
                'success': False,
                'error': 'Authentication required'
            }), 401
        
        # Validate session
        if not validate_session(guardian_id, token):
            return jsonify({
                'success': False,
                'error': 'Session expired or invalid',
                'redirect': '/?logged_out=true'
            }), 401
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build query
            base_query = '''
                SELECT e.*, d.name as driver_name
                FROM drowsiness_events e
                JOIN drivers d ON e.driver_id = d.driver_id
                WHERE e.guardian_id = %s
            '''
            params = [guardian_id]
            
            if driver_id:
                base_query += ' AND e.driver_id = %s'
                params.append(driver_id)
            
            base_query += ' ORDER BY e.timestamp DESC LIMIT %s'
            params.append(limit)
            
            try:
                cursor.execute(base_query, tuple(params))
            except Exception as e:
                print(f"❌ Error in get_drowsiness_events query: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
            
            events = cursor.fetchall()
            events_list = []
            for event in events:
                events_list.append(dict(event))
            
            return jsonify({
                'success': True,
                'session_valid': True,
                'count': len(events_list),
                'events': events_list
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'redirect': '/?logged_out=true'
        }), 500

@app.route('/api/test-alert', methods=['POST'])
def test_alert():
    """Endpoint to test alert sending (for development)"""
    try:
        data = request.json or {}
        
        # Use first guardian for testing
        with get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('SELECT guardian_id, full_name FROM guardians LIMIT 1')
                guardian_result = cursor.fetchone()
                
                if not guardian_result:
                    return jsonify({
                        'success': False,
                        'error': 'No guardian found'
                    }), 404
                
                guardian_id, guardian_name = guardian_result['guardian_id'], guardian_result['full_name']
                
            except Exception as e:
                print(f"❌ Error finding guardian: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Guardian not found'
                }), 404
            
            # Create test driver if needed
            test_driver_id = data.get('driver_id', 'TEST123')
            test_driver_name = data.get('driver_name', 'Test Driver')
            
            try:
                cursor.execute('SELECT driver_id FROM drivers WHERE driver_id = %s', (test_driver_id,))
            except Exception as e:
                print(f"❌ Error checking test driver: {e}")
                # Continue anyway
            
            if not cursor.fetchone():
                try:
                    cursor.execute('''
                        INSERT INTO drivers (driver_id, name, phone, guardian_id)
                        VALUES (%s, %s, %s, %s)
                    ''', (test_driver_id, test_driver_name, '00000000000', guardian_id))
                    conn.commit()
                except Exception as e:
                    print(f"⚠️ Error creating test driver: {e}")
            
            # Create test alert
            detection_details = {
                'state': 'Drowsy',
                'ear': 0.15,
                'mar': 0.3,
                'perclos': 0.42,
                'test': True
            }
            
            test_data = {
                'driver_id': test_driver_id,
                'driver_name': test_driver_name,
                'severity': 'high',
                'message': 'Test alert from API endpoint',
                'confidence': 0.95,
                'detection_details': detection_details
            }
            
            # Call send_alert with test data
            with app.test_request_context():
                request.json = test_data
                response = send_alert()
                return response
        
    except Exception as e:
        print(f"❌ Test alert error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== DATABASE FIX FUNCTIONS ====================
def fix_password_hashes():
    """Fix password hashes for existing users (run once)"""
    try:
        print("🛠️  Checking password hashes...")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get all guardians
            cursor.execute('SELECT guardian_id, phone, password_hash FROM guardians')
            guardians = cursor.fetchall()
            
            for guardian in guardians:
                guardian_id, phone, stored_hash = guardian['guardian_id'], guardian['phone'], guardian['password_hash']
                
                # Check if hash is valid bcrypt format
                if stored_hash and not stored_hash.startswith('$2'):
                    print(f"⚠️ Found invalid hash for guardian {guardian_id} ({phone}): {stored_hash[:30]}...")
                    print(f"   Hash starts with: {stored_hash[:10] if stored_hash else 'None'}")
                    print(f"   Hash length: {len(stored_hash) if stored_hash else 0}")
                    
                    # You would need to reset password here
                    # For now, just log it
                    
        print("✅ Password hash check completed")
        
    except Exception as e:
        print(f"❌ Error checking password hashes: {e}")

# ==================== CLEANUP FUNCTIONS ====================
def cleanup_expired_sessions():
    """Clean up expired sessions periodically"""
    try:
        with get_db_cursor() as cursor:
            try:
                cursor.execute('''
                    UPDATE session_tokens 
                    SET is_valid = FALSE 
                    WHERE expires_at < %s AND is_valid = TRUE
                ''', (datetime.now(),))
            except Exception as e:
                print(f"⚠️ Error cleaning expired sessions: {e}")
                return 0
            
            expired_count = cursor.rowcount
        
        # Clean memory cache
        current_time = datetime.now()
        expired_guards = []
        for guardian_id, session_data in active_sessions.items():
            if session_data['expires'] < current_time:
                expired_guards.append(guardian_id)
        
        for guardian_id in expired_guards:
            del active_sessions[guardian_id]
        
        # Clean admin sessions
        cleanup_admin_sessions()
        
        return expired_count
    except Exception as e:
        print(f"❌ Error cleaning up sessions: {e}")
        return 0

# ==================== STATIC FILES ====================
@app.route('/<path:filename>')
def serve_static(filename):
    """Serve all frontend static files"""
    try:
        return send_from_directory(FRONTEND_DIR, filename)
    except:
        return jsonify({'error': 'File not found'}), 404

# ==================== APPLICATION STARTUP ====================
def startup_tasks():
    """Run startup tasks"""
    print(f"\n{'='*70}")
    print("🚗 DRIVER DROWSINESS ALERT SYSTEM - CAPSTONE PROJECT")
    print(f"{'='*70}")
    
    print("🌐 DEPLOYMENT: Render.com Cloud")
    print("📊 Database: PostgreSQL (Persistent)")
    print("🔒 Security: bcrypt password hashing enabled")
    
    # Check environment
    print("\n🔧 Environment Check:")
    database_url = os.environ.get('DATABASE_URL')
    
    if database_url:
        # Show masked URL (hide password)
        if '@' in database_url:
            parts = database_url.split('@')
            user_pass = parts[0]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = f"{user}:****@{parts[1]}"
                print(f"   ✅ DATABASE_URL: {masked_url}")
        else:
            print(f"   ✅ DATABASE_URL: {database_url[:50]}...")
        
        # Test connection
        print("\n🗄️  Database Initialization:")
        try:
            if init_db():
                print("✅ Database initialized successfully")
            else:
                print("⚠️ Database initialization had issues")
        except Exception as e:
            print(f"⚠️ Database initialization error: {e}")
    else:
        print("❌ DATABASE_URL not found")
        print("   Please add DATABASE_URL to environment variables")
    
    print(f"\n🔗 API Endpoints:")
    print("  • GET  /api/health - Health check")
    print("  • GET  /api/debug/db - Database debug")
    print("  • GET  /api/test-connection - Simple connection test")
    print("  • POST /api/send-alert - Receive alerts from drowsiness detection")
    print("  • POST /api/test-alert - Test alert endpoint")
    print("  • GET  /api/test-bcrypt - Test bcrypt functionality")
    
    print(f"\n🔐 Authentication Info:")
    print("  • Admin Username: admin")
    print("  • Admin Password: admin123")
    print("  • Password Hashing: bcrypt (12 rounds)")
    
    print(f"\n🛠️  Running database fixes...")
    fix_password_hashes()
    
    print(f"{'='*70}\n")
    
    # Store app start time
    global app_start_time
    app_start_time = time.time()

# ==================== MAIN ENTRY POINT ====================
if __name__ == '__main__':
    # Run startup tasks
    startup_tasks()
    
    # Start cleanup thread
    def cleanup_worker():
        while True:
            time.sleep(3600)  # Run every hour
            cleanup_expired_sessions()
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    
    print(f"🚀 Starting server on {host}:{port}")
    print(f"🌐 WebSocket endpoint: ws://{host}:{port}")
    print(f"📡 Alert endpoint: http://{host}:{port}/api/send-alert")
    print(f"🔐 Admin login: http://{host}:{port}/admin_login")
    print(f"🔐 Test bcrypt: http://{host}:{port}/api/test-bcrypt")
    
    # Run the application
    socketio.run(app, 
                host=host, 
                port=port, 
                debug=False,
                use_reloader=False,
                log_output=False,
                allow_unsafe_werkzeug=False)