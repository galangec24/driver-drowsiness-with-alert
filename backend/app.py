"""
CAPSTONE PROJECT - DRIVER ALERT SYSTEM
Firebase Hosting + Render Backend - Pure API Server
Updated for Firebase Hosting Deployment
"""

import os
from pathlib import Path
import sys

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

# ==================== FIREBASE HOSTING INTEGRATION ====================
# List of allowed origins (Firebase domains + localhost for development)
ALLOWED_ORIGINS = [
    # 'https://guardian-drive.web.app',
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
socketio = SocketIO(app, 
                   cors_allowed_origins=ALLOWED_ORIGINS,
                   async_mode='eventlet',
                   ping_timeout=60,
                   ping_interval=25,
                   async_handlers=True,
                   logger=False,
                   engineio_logger=False)

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
def verify_guardian_credentials(phone, password):
    """Verify guardian login credentials using bcrypt - FIXED VERSION"""
    try:
        print(f"\n🔍 [LOGIN VERIFY] Starting verification")
        print(f"   Phone received: '{phone}'")
        print(f"   Password received: '{password}' (length: {len(password)})")
        
        # Clean the phone number
        phone_clean = str(phone).strip()
        phone_clean = re.sub(r'[\s\-\(\)\+]', '', phone_clean)
        print(f"   Cleaned phone: '{phone_clean}'")
        
        # Check if it's all digits
        if not phone_clean.isdigit():
            print(f"❌ [LOGIN VERIFY] Phone contains non-digits")
            return None
        
        # Convert to 09 format - FIXED LOGIC
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
        
        # Check database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # SIMPLE query - just get the user
            cursor.execute('''
                SELECT guardian_id, full_name, password_hash, is_active
                FROM guardians 
                WHERE phone = %s AND auth_provider = 'phone'
            ''', (lookup_phone,))
            
            result = cursor.fetchone()
            
            if not result:
                print(f"❌ [LOGIN VERIFY] No user found with phone: '{lookup_phone}'")
                
                # Debug: Show what's in the database
                cursor.execute('SELECT phone, full_name FROM guardians LIMIT 5')
                all_phones = cursor.fetchall()
                print(f"   First 5 phones in DB: {[p['phone'] for p in all_phones]}")
                
                return None
            
            # FIX: Check what type of object result is
            print(f"   Result type: {type(result)}")
            
            # Handle both dictionary and tuple results
            if isinstance(result, dict):
                # Dictionary from RealDictCursor
                guardian_id = result['guardian_id']
                full_name = result['full_name']
                stored_hash = result['password_hash']
                is_active = result['is_active']
                print(f"   Using dictionary access")
            else:
                # Tuple from regular cursor
                guardian_id, full_name, stored_hash, is_active = result
                print(f"   Using tuple unpacking")
            
            print(f"✅ [LOGIN VERIFY] User found: {full_name} (ID: {guardian_id})")
            print(f"   Stored hash: {stored_hash[:30]}...")
            print(f"   Hash length: {len(stored_hash)}")
            print(f"   Is bcrypt format: {stored_hash.startswith('$2') if stored_hash else False}")
            
            # Check if we have a hash
            if not stored_hash:
                print(f"❌ [LOGIN VERIFY] No password hash stored for user")
                return None
            
            # Verify password - This will clean the password internally
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
                    'phone': lookup_phone
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
        "default-src 'self' https://guardian-drive.web.app; "
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
    """Handle client connection"""
    client_id = request.sid
    connected_clients[client_id] = {
        'connected_at': datetime.now(),
        'ip': request.remote_addr,
        'type': None,
        'guardian_id': None,
        'authenticated': False
    }
    print(f"✅ WebSocket client connected: {client_id} from {request.remote_addr}")
    emit('connected', {'status': 'connected', 'client_id': client_id})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    if client_id in connected_clients:
        print(f"⚠️  WebSocket client disconnected: {client_id}")
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
            print(f"✅ Guardian authenticated via WebSocket: {guardian['full_name']} ({guardian_id})")
            emit('auth_confirmed', {
                'guardian_id': guardian_id,
                'full_name': guardian['full_name'],
                'phone': guardian['phone']
            })
            return
    
    print(f"❌ WebSocket authentication failed for client: {client_id}")
    emit('auth_failed', {'error': 'Authentication failed'})
    socketio.disconnect(client_id)

# ==================== MAIN ROUTES ====================
@app.route('/')
def serve_home():
    """Redirect to Firebase Hosting - pure backend server"""
    return jsonify({
        'success': True,
        'message': 'Driver Alert System API Server',
        'backend': 'Render.com',
        'frontend': 'Firebase Hosting',
        'frontend_url': 'https://guardian-drive.web.app',
        'api_docs': 'https://driver-drowsiness-with-alert.onrender.com/api/health',
        'version': '2.0.0'
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
            'firebase_integration': True,
            'allowed_origins': ALLOWED_ORIGINS,
            'websocket_connections': len(connected_clients),
            'frontend_url': 'https://guardian-drive.web.app',
            'note': 'This is a pure API server. Frontend is hosted on Firebase.'
        }
        
        return jsonify(health_info)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'running_with_errors',
            'error': str(e)
        }), 200

# ==================== GOOGLE AUTH CONFIG ====================
@app.route('/api/config/google', methods=['GET'])
def get_google_config():
    """Get Google OAuth configuration for frontend"""
    return jsonify({
        'success': True,
        'google_client_id': GOOGLE_CLIENT_ID,
        'firebase_domain': 'guardian-drive.web.app',
        'backend_url': 'https://driver-drowsiness-with-alert.onrender.com',
        'websocket_url': 'wss://driver-drowsiness-with-alert.onrender.com',
        'message': 'Google OAuth configuration loaded'
    })

@app.route('/api/firebase-config', methods=['GET'])
def get_firebase_config():
    """Get Firebase configuration for clients"""
    return jsonify({
        'success': True,
        'firebase_domain': 'guardian-drive.web.app',
        'backend_url': 'https://driver-drowsiness-with-alert.onrender.com',
        'websocket_url': 'wss://driver-drowsiness-with-alert.onrender.com',
        'api_base': 'https://driver-drowsiness-with-alert.onrender.com/api',
        'message': 'Firebase Hosting configuration'
    })

# ==================== GOOGLE LOGIN ENDPOINT ====================
@app.route('/api/google-login', methods=['POST'])
def google_login():
    """Handle Google OAuth login"""
    try:
        data = request.json
        google_token = data.get('token')
        
        if not google_token:
            return jsonify({
                'success': False,
                'error': 'Google token required'
            }), 400
        
        print(f"🔐 [GOOGLE LOGIN] Received Google token")
        
        email = None
        name = None
        google_id = None
        
        try:
            # Try to verify the Google token
            idinfo = id_token.verify_oauth2_token(
                google_token, 
                google_requests.Request(),
                GOOGLE_CLIENT_ID
            )
            
            email = idinfo.get('email')
            name = idinfo.get('name', '')
            google_id = idinfo.get('sub')
            
            print(f"✅ Google token verified: {email}")
            
        except Exception as verify_error:
            print(f"❌ Google token verification failed: {verify_error}")
            
            # Try to decode without verification
            try:
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
        
        if not email:
            return jsonify({
                'success': False,
                'error': 'Email not found in token'
            }), 400
        
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
                
                print(f"✅ New Google user created: {full_name} (ID: {guardian_id})")
        
        # Create session
        token = create_session(guardian_id, request.remote_addr, request.headers.get('User-Agent'))
        log_activity(guardian_id, 'GOOGLE_LOGIN', f'Google login from {request.remote_addr}')
        
        return jsonify({
            'success': True,
            'guardian_id': guardian_id,
            'full_name': full_name,
            'email': email,
            'session_token': token,
            'message': 'Google login successful',
            'is_google_user': True,
            'redirect_url': f'https://guardian-drive.web.app/guardian-dashboard.html?guardian_id={guardian_id}&token={token}'
        })
        
    except Exception as e:
        print(f"❌ Google login error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Google login failed. Please try again.'
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
        
        # Create session
        token = create_session(guardian_id, request.remote_addr, request.headers.get('User-Agent'))
        
        return jsonify({
            'success': True,
            'guardian_id': guardian_id,
            'full_name': full_name,
            'email': email,
            'session_token': token,
            'message': 'Login successful',
            'method': 'email_based_google_login',
            'redirect_url': f'https://guardian-drive.web.app/guardian-dashboard.html?guardian_id={guardian_id}&token={token}'
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
        
        # Create session
        token = create_session(guardian_id, request.remote_addr, request.headers.get('User-Agent'))
        
        return jsonify({
            'success': True,
            'guardian_id': guardian_id,
            'full_name': full_name,
            'email': email,
            'session_token': token,
            'message': 'Google login successful',
            'redirect_url': f'https://guardian-drive.web.app/guardian-dashboard.html?guardian_id={guardian_id}&token={token}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== GUARDIAN AUTHENTICATION ====================
@app.route('/api/login', methods=['POST'])
def login():
    """Guardian login with bcrypt verification - FIXED VERSION"""
    try:
        data = request.json
        phone = data.get('phone', '').strip()
        password = data.get('password', '')

        print(f"\n🔑 [LOGIN API] Attempting login")
        print(f"   Phone: '{phone}'")
        print(f"   Password (raw): '{password}' (length: {len(password)})")

        if not phone or not password:
            return jsonify({'success': False, 'error': 'Phone and password required'}), 400

        ip = request.remote_addr
        if rate_limit_exceeded(ip, 'guardian_login', limit=10):
            return jsonify({'success': False, 'error': 'Too many login attempts'}), 429

        guardian = verify_guardian_credentials(phone, password)
        
        if guardian:
            token = create_session(guardian['guardian_id'], request.remote_addr, request.headers.get('User-Agent'))
            log_activity(guardian['guardian_id'], 'LOGIN', f'Guardian logged in from {request.remote_addr}')
            
            return jsonify({
                'success': True,
                'guardian_id': guardian['guardian_id'],
                'full_name': guardian['full_name'],
                'phone': guardian['phone'],
                'session_token': token,
                'message': 'Login successful',
                'redirect_url': f'https://guardian-drive.web.app/guardian-dashboard.html?guardian_id={guardian["guardian_id"]}&token={token}'
            })

        return jsonify({'success': False, 'error': 'Invalid phone or password'}), 401

    except Exception as e:
        print(f"❌ Login error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Login failed. Please try again.'}), 500

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
            'redirect_url': 'https://guardian-drive.web.app/?logged_out=true'
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
            'redirect_url': f'https://guardian-drive.web.app/login.html?registered=true&prefilled_phone={final_phone}'
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
                'redirect_url': f'https://guardian-drive.web.app/admin-dashboard.html?username={username}&token={token}'
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
                'allowed_origins': ALLOWED_ORIGINS,
                'frontend_url': 'https://guardian-drive.web.app'
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

# ==================== APPLICATION STARTUP ====================
def startup_tasks():
    """Run startup tasks"""
    print(f"\n{'='*70}")
    print("🚗 DRIVER DROWSINESS ALERT SYSTEM - CAPSTONE PROJECT")
    print(f"{'='*70}")
    
    print("🌐 DEPLOYMENT: Firebase Hosting + Render Backend")
    print("📊 Database: PostgreSQL (Persistent)")
    print("🔒 Security: bcrypt password hashing enabled")
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
    
    # Check secret key
    secret_key = os.environ.get('SECRET_KEY')
    if secret_key:
        print(f"   ✅ SECRET_KEY: [configured]")
    else:
        print(f"   ⚠️  SECRET_KEY: using generated key")
    
    # Firebase integration
    print(f"   ✅ Firebase Integration: Enabled")
    print(f"   ✅ Frontend URL: https://guardian-drive.web.app")
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
        print(f"   Firebase Frontend: https://guardian-drive.web.app")
    else:
        print(f"   ⚠️  Not running on Render (local development)")
    
    # API endpoints summary
    print(f"\n🔗 Available API Endpoints:")
    endpoints = [
        ("GET", "/api/health", "Health check"),
        ("POST", "/api/login", "Guardian login"),
        ("POST", "/api/google-login", "Google OAuth login"),
        ("POST", "/api/register-guardian", "Guardian registration"),
        ("POST", "/api/send-alert", "Send drowsiness alert"),
        ("GET", "/api/guardian/dashboard", "Guardian dashboard"),
        ("POST", "/api/admin/login", "Admin login"),
    ]
    
    for method, path, desc in endpoints:
        print(f"   • {method:6} {path:30} - {desc}")
    
    # Authentication info
    print(f"\n🔐 Authentication Info:")
    print("  • Admin Username: admin")
    print("  • Admin Password: admin123")
    print("  • Guardian Registration: Open to public")
    print("  • Google OAuth: " + ("Enabled" if google_client_id and google_client_id != 'your-google-client-id-here' else "Disabled"))
    print("  • Firebase Domain: guardian-drive.web.app")
    
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
    print(f"   Firebase Frontend: https://guardian-drive.web.app")
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
    
    print(f"🚀 Starting server on {host}:{port}")
    print(f"🌐 WebSocket endpoint: ws://{host}:{port}")
    print(f"📡 Alert endpoint: http://{host}:{port}/api/send-alert")
    print(f"🔥 Firebase Frontend: https://guardian-drive.web.app")
    print(f"🔐 Google OAuth: {'Enabled' if GOOGLE_CLIENT_ID else 'Disabled'}")
    
    # Run the application
    socketio.run(app, 
                host=host, 
                port=port, 
                debug=False,
                use_reloader=False,
                log_output=False,
                allow_unsafe_werkzeug=False)