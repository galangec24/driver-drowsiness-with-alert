"""
CAPSTONE PROJECT - DRIVER ALERT SYSTEM
Render.com Deployment Ready Version with PostgreSQL Support
COMPLETE VERSION - No parts skipped
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, send_from_directory, Response, redirect, g
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import sqlite3
from datetime import datetime, timedelta
import uuid
import eventlet
import base64
import json
import hashlib
import secrets
import threading
import time
import re
from contextlib import contextmanager

# ==================== RENDER DEPLOYMENT CONFIG ====================
IS_RENDER = os.environ.get('RENDER') is not None
IS_PRODUCTION = os.environ.get('ENVIRONMENT') == 'production' or IS_RENDER

# PostgreSQL Support
if IS_RENDER:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        import urllib.parse as urlparse
        POSTGRES_AVAILABLE = True
        print("✅ PostgreSQL libraries available")
    except ImportError as e:
        POSTGRES_AVAILABLE = False
        print(f"⚠️ PostgreSQL not available: {e}")
else:
    POSTGRES_AVAILABLE = False

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')
DB_PATH = os.path.join(BASE_DIR, 'drivers.db')
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
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['DATABASE'] = DB_PATH

# Configure for production with proxy support
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
CORS(app, 
     supports_credentials=True,
     resources={
         r"/*": {
             "origins": ["*"] if IS_PRODUCTION else ["http://localhost:*", "http://127.0.0.1:*"],
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
                   logger=not IS_PRODUCTION,
                   engineio_logger=not IS_PRODUCTION)

# ==================== SECURITY CONFIGURATION ====================
ADMIN_CREDENTIALS = {
    'admin': {
        'password_hash': '8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918',
        'full_name': 'System Administrator',
        'role': 'super_admin',
        'email': 'admin@driveralert.com',
        'created_at': datetime.now().isoformat()
    }
}

admin_rate_limit = {}
guardian_rate_limit = {}
admin_sessions = {}

# ==================== SECURITY FUNCTIONS ====================
def hash_password(password):
    """Hash password using SHA-256 with salt"""
    salt = "dr1v3r_@l3rt_s@lt_" + os.environ.get('PEPPER', 'default_pepper')
    return hashlib.sha256((password + salt).encode()).hexdigest()

def verify_admin_credentials(username, password):
    """Verify admin login credentials"""
    if username not in ADMIN_CREDENTIALS:
        return None
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    if password_hash == ADMIN_CREDENTIALS[username]['password_hash']:
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
    
    if expired_users and not IS_PRODUCTION:
        print(f"🧹 Cleaned up {len(expired_users)} expired admin sessions")

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
def get_db():
    """Get database connection - PostgreSQL on Render, SQLite locally"""
    if IS_RENDER and POSTGRES_AVAILABLE:
        DATABASE_URL = os.environ.get('DATABASE_URL')
        if DATABASE_URL:
            try:
                url = urlparse.urlparse(DATABASE_URL)
                conn = psycopg2.connect(
                    database=url.path[1:],
                    user=url.username,
                    password=url.password,
                    host=url.hostname,
                    port=url.port,
                    cursor_factory=RealDictCursor
                )
                return conn
            except Exception as e:
                print(f"❌ PostgreSQL connection error: {e}")
    
    # Fallback to SQLite
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = sqlite3.connect(app.config['DATABASE'], timeout=20)
        g.sqlite_db.row_factory = sqlite3.Row
        g.sqlite_db.execute("PRAGMA busy_timeout = 10000")
        g.sqlite_db.execute("PRAGMA journal_mode = WAL")
        g.sqlite_db.execute("PRAGMA synchronous = NORMAL")
        g.sqlite_db.execute("PRAGMA foreign_keys = ON")
    return g.sqlite_db

@app.teardown_appcontext
def close_db(error):
    """Close database connection at the end of request"""
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        if IS_RENDER and POSTGRES_AVAILABLE:
            DATABASE_URL = os.environ.get('DATABASE_URL')
            if DATABASE_URL:
                url = urlparse.urlparse(DATABASE_URL)
                conn = psycopg2.connect(
                    database=url.path[1:],
                    user=url.username,
                    password=url.password,
                    host=url.hostname,
                    port=url.port,
                    cursor_factory=RealDictCursor
                )
        if conn is None:
            # Fallback to SQLite
            conn = sqlite3.connect(app.config['DATABASE'], timeout=20)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 10000")
            conn.execute("PRAGMA journal_mode = WAL")
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
        if IS_RENDER and POSTGRES_AVAILABLE:
            DATABASE_URL = os.environ.get('DATABASE_URL')
            if DATABASE_URL:
                url = urlparse.urlparse(DATABASE_URL)
                conn = psycopg2.connect(
                    database=url.path[1:],
                    user=url.username,
                    password=url.password,
                    host=url.hostname,
                    port=url.port
                )
                cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if conn is None:
            # Fallback to SQLite
            conn = sqlite3.connect(app.config['DATABASE'], timeout=20)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 10000")
            conn.execute("PRAGMA journal_mode = WAL")
            cursor = conn.cursor()
        
        yield cursor
        if conn:
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
db_write_lock = threading.Lock()

# ==================== DATABASE INITIALIZATION ====================
def init_db():
    """Initialize database with all required tables"""
    try:
        with get_db_cursor() as cursor:
            if IS_RENDER and POSTGRES_AVAILABLE:
                # PostgreSQL schema
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
                
            else:
                # SQLite schema
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS guardians (
                        guardian_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        full_name TEXT NOT NULL,
                        phone TEXT UNIQUE NOT NULL,
                        email TEXT,
                        password_hash TEXT NOT NULL,
                        address TEXT,
                        registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
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
                        guardian_id INTEGER,
                        registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        driver_id TEXT NOT NULL,
                        guardian_id INTEGER,
                        severity TEXT NOT NULL CHECK(severity IN ('low', 'medium', 'high')),
                        message TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        acknowledged BOOLEAN DEFAULT 0,
                        detection_details TEXT,
                        source TEXT DEFAULT 'system',
                        FOREIGN KEY (driver_id) REFERENCES drivers(driver_id) ON DELETE CASCADE,
                        FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_images (
                        image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        driver_id TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        capture_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (driver_id) REFERENCES drivers(driver_id) ON DELETE CASCADE
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS drowsiness_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        driver_id TEXT NOT NULL,
                        guardian_id INTEGER,
                        confidence REAL,
                        state TEXT,
                        ear REAL,
                        mar REAL,
                        perclos REAL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed BOOLEAN DEFAULT 0,
                        FOREIGN KEY (driver_id) REFERENCES drivers(driver_id) ON DELETE CASCADE,
                        FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS activity_log (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                        token_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        guardian_id INTEGER NOT NULL,
                        token TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        is_valid BOOLEAN DEFAULT 1,
                        ip_address TEXT,
                        user_agent TEXT,
                        FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS admin_activity_log (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        admin_username TEXT NOT NULL,
                        action TEXT,
                        details TEXT,
                        ip_address TEXT,
                        user_agent TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Check and add missing columns for SQLite
                columns_to_check = [
                    ('guardians', 'is_active'),
                    ('guardians', 'failed_login_attempts'),
                    ('guardians', 'locked_until'),
                    ('drivers', 'is_active'),
                    ('alerts', 'source'),
                    ('activity_log', 'admin_username'),
                    ('activity_log', 'ip_address'),
                    ('activity_log', 'user_agent'),
                    ('session_tokens', 'ip_address'),
                    ('session_tokens', 'user_agent')
                ]
                
                for table, column in columns_to_check:
                    try:
                        cursor.execute(f"SELECT {column} FROM {table} LIMIT 1")
                    except Exception:
                        if column == 'is_active':
                            cursor.execute(f'ALTER TABLE {table} ADD COLUMN {column} BOOLEAN DEFAULT 1')
                        elif column == 'failed_login_attempts':
                            cursor.execute(f'ALTER TABLE {table} ADD COLUMN {column} INTEGER DEFAULT 0')
                        elif column == 'locked_until':
                            cursor.execute(f'ALTER TABLE {table} ADD COLUMN {column} TIMESTAMP')
                        elif column == 'source':
                            cursor.execute(f'ALTER TABLE {table} ADD COLUMN {column} TEXT DEFAULT "system"')
                        else:
                            cursor.execute(f'ALTER TABLE {table} ADD COLUMN {column} TEXT')
                        print(f"📝 Added {column} column to {table} table")
                
                # Create indexes for SQLite
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_drivers_guardian ON drivers(guardian_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_drivers_active ON drivers(is_active)')
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
            
            # Insert demo guardian if not exists
            try:
                # Try PostgreSQL syntax first
                cursor.execute('SELECT COUNT(*) FROM guardians WHERE phone = %s', ('09123456789',))
            except:
                # Fallback to SQLite syntax
                cursor.execute('SELECT COUNT(*) FROM guardians WHERE phone = ?', ('09123456789',))
            
            count = cursor.fetchone()[0]
            if count == 0:
                demo_password_hash = hash_password('demo123')
                try:
                    # Try PostgreSQL syntax
                    cursor.execute('''
                        INSERT INTO guardians (full_name, phone, email, password_hash, address, last_login)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    ''', ('Demo Guardian', '09123456789', 'demo@driveralert.com', 
                          demo_password_hash, 'Demo Address', datetime.now()))
                except:
                    # Fallback to SQLite syntax
                    cursor.execute('''
                        INSERT INTO guardians (full_name, phone, email, password_hash, address, last_login)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', ('Demo Guardian', '09123456789', 'demo@driveralert.com', 
                          demo_password_hash, 'Demo Address', datetime.now()))
                print("✅ Demo guardian created")
        
        print("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        import traceback
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
                cursor.execute('UPDATE session_tokens SET is_valid = 0 WHERE guardian_id = %s AND is_valid = 1', (guardian_id,))
            except:
                cursor.execute('UPDATE session_tokens SET is_valid = 0 WHERE guardian_id = ? AND is_valid = 1', (guardian_id,))
            
            # Create new session
            try:
                cursor.execute('''
                    INSERT INTO session_tokens (guardian_id, token, expires_at, ip_address, user_agent)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (guardian_id, token, expires_at, ip_address, user_agent))
            except:
                cursor.execute('''
                    INSERT INTO session_tokens (guardian_id, token, expires_at, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?)
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
            try:
                cursor.execute('''
                    SELECT COUNT(*) FROM session_tokens 
                    WHERE guardian_id = %s AND token = %s AND is_valid = 1 AND expires_at > %s
                ''', (guardian_id, token, datetime.now()))
            except:
                cursor.execute('''
                    SELECT COUNT(*) FROM session_tokens 
                    WHERE guardian_id = ? AND token = ? AND is_valid = 1 AND expires_at > ?
                ''', (guardian_id, token, datetime.now()))
            
            result = cursor.fetchone()[0] > 0
            
            if result and guardian_id not in active_sessions:
                active_sessions[guardian_id] = {
                    'token': token,
                    'expires': datetime.now() + timedelta(hours=23),
                    'created': datetime.now()
                }
            
            return result
    except Exception as e:
        print(f"❌ Error validating session: {e}")
        return False

def invalidate_session(guardian_id, token=None):
    """Invalidate a guardian session"""
    try:
        with get_db_cursor() as cursor:
            if token:
                try:
                    cursor.execute('UPDATE session_tokens SET is_valid = 0 WHERE guardian_id = %s AND token = %s', (guardian_id, token))
                except:
                    cursor.execute('UPDATE session_tokens SET is_valid = 0 WHERE guardian_id = ? AND token = ?', (guardian_id, token))
            else:
                try:
                    cursor.execute('UPDATE session_tokens SET is_valid = 0 WHERE guardian_id = %s', (guardian_id,))
                except:
                    cursor.execute('UPDATE session_tokens SET is_valid = 0 WHERE guardian_id = ?', (guardian_id,))
        
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
    """Verify guardian login credentials"""
    try:
        # Clean and normalize phone number for lookup
        phone = str(phone).strip()
        phone = re.sub(r'[\s\-\(\)]', '', phone)
        
        # Generate all possible phone formats to check
        phone_formats = []
        
        # Original format
        phone_formats.append(phone)
        
        # Try normalized formats
        if phone.startswith('0'):
            # 09XXXXXXXXX -> +639XXXXXXXXX
            phone_formats.append('+63' + phone[1:])
        elif phone.startswith('63') and not phone.startswith('+'):
            # 639XXXXXXXXX -> +639XXXXXXXXX
            phone_formats.append('+' + phone)
        elif phone.startswith('+63'):
            # +639XXXXXXXXX -> 09XXXXXXXXX
            phone_formats.append('0' + phone[3:])
        
        # Remove duplicates
        phone_formats = list(set(phone_formats))
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build query with multiple phone formats
            if IS_RENDER and POSTGRES_AVAILABLE:
                placeholders = ', '.join(['%s'] * len(phone_formats))
            else:
                placeholders = ', '.join(['?'] * len(phone_formats))
            
            query = f'''
                SELECT guardian_id, full_name, password_hash, failed_login_attempts, locked_until
                FROM guardians 
                WHERE phone IN ({placeholders}) AND is_active = 1
            '''
            
            try:
                cursor.execute(query, phone_formats)
            except Exception as e:
                print(f"❌ Query error: {e}")
                return None
            
            result = cursor.fetchone()
            
            if result:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    guardian_id, full_name, stored_hash, failed_attempts, locked_until = result
                else:
                    guardian_id, full_name, stored_hash, failed_attempts, locked_until = result
                
                # Check if account is locked
                if locked_until:
                    try:
                        # Try to parse as datetime object or string
                        if isinstance(locked_until, str):
                            lock_time = datetime.fromisoformat(locked_until.replace('Z', '+00:00'))
                        else:
                            lock_time = locked_until
                        
                        if lock_time > datetime.now():
                            remaining_minutes = (lock_time - datetime.now()).seconds // 60
                            print(f"🔒 Account locked for {remaining_minutes} more minutes")
                            return {'error': 'Account locked', 'locked_until': locked_until}
                    except Exception as e:
                        print(f"⚠️ Error parsing lock time: {e}")
                        # Continue if we can't parse the lock time
                
                # Verify password
                if hash_password(password) == stored_hash:
                    # Reset failed attempts on successful login
                    try:
                        cursor.execute('UPDATE guardians SET failed_login_attempts = 0, last_login = %s WHERE guardian_id = %s', 
                                     (datetime.now(), guardian_id))
                    except:
                        cursor.execute('UPDATE guardians SET failed_login_attempts = 0, last_login = ? WHERE guardian_id = ?', 
                                     (datetime.now(), guardian_id))
                    conn.commit()
                    
                    return {
                        'guardian_id': guardian_id, 
                        'full_name': full_name, 
                        'phone': phone_formats[0]  # Return the original phone format
                    }
                else:
                    # Increment failed attempts
                    failed_attempts = (failed_attempts or 0) + 1
                    
                    if failed_attempts >= 5:
                        locked_until = datetime.now() + timedelta(minutes=30)
                        lock_time_str = locked_until.isoformat()
                        try:
                            cursor.execute('UPDATE guardians SET failed_login_attempts = %s, locked_until = %s WHERE guardian_id = %s', 
                                         (failed_attempts, lock_time_str, guardian_id))
                        except:
                            cursor.execute('UPDATE guardians SET failed_login_attempts = ?, locked_until = ? WHERE guardian_id = ?', 
                                         (failed_attempts, lock_time_str, guardian_id))
                        print(f"🔒 Account locked for 30 minutes due to {failed_attempts} failed attempts")
                    else:
                        try:
                            cursor.execute('UPDATE guardians SET failed_login_attempts = %s WHERE guardian_id = %s', 
                                         (failed_attempts, guardian_id))
                        except:
                            cursor.execute('UPDATE guardians SET failed_login_attempts = ? WHERE guardian_id = ?', 
                                         (failed_attempts, guardian_id))
                    
                    conn.commit()
                    
                    # Still return None for failed login
                    return None
                    
    except Exception as e:
        print(f"❌ Error verifying credentials: {e}")
        import traceback
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
            except:
                cursor.execute('''
                    SELECT guardian_id, full_name, phone, email, address, registration_date, last_login
                    FROM guardians WHERE guardian_id = ?
                ''', (guardian_id,))
            
            result = cursor.fetchone()
            if result:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    return dict(result)
                else:
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
                except:
                    cursor.execute('''
                        INSERT INTO admin_activity_log (admin_username, action, details, ip_address, user_agent)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (admin_username, action, details, ip_address, user_agent))
            else:
                try:
                    cursor.execute('''
                        INSERT INTO activity_log (guardian_id, action, details, ip_address, user_agent)
                        VALUES (%s, %s, %s, %s, %s)
                    ''', (guardian_id, action, details, ip_address, user_agent))
                except:
                    cursor.execute('''
                        INSERT INTO activity_log (guardian_id, action, details, ip_address, user_agent)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (guardian_id, action, details, ip_address, user_agent))
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
    """Save base64 image to file and database - THREAD-SAFE"""
    try:
        driver_dir = os.path.join(FACE_IMAGES_DIR, driver_id)
        os.makedirs(driver_dir, exist_ok=True)
        
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image_filename = f'face_{image_number}_{int(time.time())}.jpg'
        image_path = os.path.join(driver_dir, image_filename)
        
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        with db_write_lock:
            with get_db_cursor() as cursor:
                try:
                    cursor.execute('INSERT INTO face_images (driver_id, image_path) VALUES (%s, %s)', (driver_id, image_path))
                except:
                    cursor.execute('INSERT INTO face_images (driver_id, image_path) VALUES (?, ?)', (driver_id, image_path))
        
        return image_path
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return None

def get_guardian_drivers(guardian_id):
    """Get all drivers registered by a guardian"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    SELECT d.*, 
                           (SELECT COUNT(*) FROM alerts a WHERE a.driver_id = d.driver_id AND a.acknowledged = 0) as alert_count,
                           (SELECT COUNT(*) FROM face_images f WHERE f.driver_id = d.driver_id) as face_count
                    FROM drivers d
                    WHERE d.guardian_id = %s AND d.is_active = 1
                    ORDER BY d.registration_date DESC
                ''', (guardian_id,))
            except:
                cursor.execute('''
                    SELECT d.*, 
                           (SELECT COUNT(*) FROM alerts a WHERE a.driver_id = d.driver_id AND a.acknowledged = 0) as alert_count,
                           (SELECT COUNT(*) FROM face_images f WHERE f.driver_id = d.driver_id) as face_count
                    FROM drivers d
                    WHERE d.guardian_id = ? AND d.is_active = 1
                    ORDER BY d.registration_date DESC
                ''', (guardian_id,))
            
            drivers = cursor.fetchall()
            result = []
            for driver in drivers:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    result.append(dict(driver))
                else:
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
                    SELECT a.*, d.name as driver_name,
                           CASE 
                               WHEN a.detection_details IS NOT NULL AND a.detection_details != '' 
                               THEN detection_details->>'state'
                               ELSE 'alert'
                           END as alert_type
                    FROM alerts a
                    JOIN drivers d ON a.driver_id = d.driver_id
                    WHERE a.guardian_id = %s
                    ORDER BY a.timestamp DESC
                    LIMIT %s
                ''', (guardian_id, limit))
            except:
                cursor.execute('''
                    SELECT a.*, d.name as driver_name,
                           CASE 
                               WHEN a.detection_details IS NOT NULL AND a.detection_details != '' 
                               THEN json_extract(a.detection_details, '$.state')
                               ELSE 'alert'
                           END as alert_type
                    FROM alerts a
                    JOIN drivers d ON a.driver_id = d.driver_id
                    WHERE a.guardian_id = ?
                    ORDER BY a.timestamp DESC
                    LIMIT ?
                ''', (guardian_id, limit))
            
            alerts = cursor.fetchall()
            result = []
            for alert in alerts:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    alert_dict = dict(alert)
                else:
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
                    WHERE d.driver_id = %s AND d.is_active = 1
                ''', (identifier,))
            except:
                cursor.execute('''
                    SELECT d.*, g.full_name as guardian_name, g.phone as guardian_phone
                    FROM drivers d
                    JOIN guardians g ON d.guardian_id = g.guardian_id
                    WHERE d.driver_id = ? AND d.is_active = 1
                ''', (identifier,))
            
            result = cursor.fetchone()
            
            # If not found by ID, try by name
            if not result:
                try:
                    cursor.execute('''
                        SELECT d.*, g.full_name as guardian_name, g.phone as guardian_phone
                        FROM drivers d
                        JOIN guardians g ON d.guardian_id = g.guardian_id
                        WHERE d.name LIKE %s AND d.is_active = 1
                    ''', (f'%{identifier}%',))
                except:
                    cursor.execute('''
                        SELECT d.*, g.full_name as guardian_name, g.phone as guardian_phone
                        FROM drivers d
                        JOIN guardians g ON d.guardian_id = g.guardian_id
                        WHERE d.name LIKE ? AND d.is_active = 1
                    ''', (f'%{identifier}%',))
                result = cursor.fetchone()
            
            if result:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    return dict(result)
                else:
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
    
    if IS_PRODUCTION:
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
    if not IS_PRODUCTION:
        print(f"✅ Client connected: {client_id}")
    emit('connected', {'status': 'connected', 'client_id': client_id})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    if client_id in connected_clients:
        del connected_clients[client_id]
    if not IS_PRODUCTION:
        print(f"❌ Client disconnected: {client_id}")

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
            if not IS_PRODUCTION:
                print(f"👤 Guardian authenticated via WebSocket: {guardian['full_name']}")
            emit('auth_confirmed', {
                'guardian_id': guardian_id,
                'full_name': guardian['full_name'],
                'phone': guardian['phone']
            })
            return
    
    if not IS_PRODUCTION:
        print(f"🔒 WebSocket authentication failed for client {client_id}")
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
                    WHERE a.guardian_id = %s AND a.acknowledged = 0 
                    AND a.timestamp > NOW() - INTERVAL '1 hour'
                    ORDER BY a.timestamp DESC
                ''', (guardian_id,))
            except:
                cursor.execute('''
                    SELECT a.*, d.name as driver_name
                    FROM alerts a
                    JOIN drivers d ON a.driver_id = d.driver_id
                    WHERE a.guardian_id = ? AND a.acknowledged = 0 
                    AND a.timestamp > datetime('now', '-1 hour')
                    ORDER BY a.timestamp DESC
                ''', (guardian_id,))
            
            alerts = cursor.fetchall()
            
            for alert in alerts:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    alert_data = dict(alert)
                else:
                    alert_data = dict(alert)
                if alert_data.get('detection_details'):
                    try:
                        alert_data['detection_details'] = json.loads(alert_data['detection_details'])
                    except:
                        pass
                
                socketio.emit('guardian_alert', alert_data, room=client_id)
                
            if alerts and not IS_PRODUCTION:
                print(f"📨 Sent {len(alerts)} pending alerts to guardian {guardian_id}")
                
    except Exception as e:
        print(f"❌ Error sending pending alerts: {e}")

# ==================== MAIN ROUTES ====================
@app.route('/')
def serve_home():
    """Main route - Redirects to admin login for desktop, mobile login for mobile"""
    if request.args.get('force') == 'mobile':
        return send_from_directory(FRONTEND_DIR, 'login.html')
    if request.args.get('force') == 'desktop':
        return send_from_directory(FRONTEND_DIR, 'admin-login.html')
    
    if is_mobile_device():
        return send_from_directory(FRONTEND_DIR, 'login.html')
    else:
        return send_from_directory(FRONTEND_DIR, 'admin-login.html')

@app.route('/admin-login')
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
    response = send_from_directory(FRONTEND_DIR, 'manifest.json')
    response.headers['Content-Type'] = 'application/manifest+json'
    return response

@app.route('/service-worker.js')
def serve_service_worker():
    response = send_from_directory(FRONTEND_DIR, 'service-worker.js')
    response.headers['Content-Type'] = 'application/javascript'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route('/icon-<int:size>.png')
def serve_icon(size):
    icon_path = os.path.join(FRONTEND_DIR, f'icon-{size}.png')
    if os.path.exists(icon_path):
        return send_from_directory(FRONTEND_DIR, f'icon-{size}.png')
    return send_from_directory(FRONTEND_DIR, 'icon-192.png')

# ==================== API ENDPOINTS ====================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            db_status = cursor.fetchone()[0] == 1
            
            cursor.execute('SELECT COUNT(*) FROM guardians')
            guardian_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM drivers')
            driver_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM alerts')
            alert_count = cursor.fetchone()[0]
            
            try:
                cursor.execute('SELECT COUNT(*) FROM drowsiness_events WHERE DATE(timestamp) = CURRENT_DATE')
            except:
                cursor.execute('SELECT COUNT(*) FROM drowsiness_events WHERE DATE(timestamp) = DATE("now")')
            drowsiness_events_today = cursor.fetchone()[0]
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'server': 'Driver Alert System',
            'version': '2.0.0',
            'environment': 'production' if IS_PRODUCTION else 'development',
            'deployment': 'render' if IS_RENDER else 'local',
            'database': 'postgresql' if (IS_RENDER and POSTGRES_AVAILABLE) else 'sqlite',
            'database_status': 'connected' if db_status else 'disconnected',
            'connected_clients': len(connected_clients),
            'active_sessions': len(active_sessions),
            'admin_sessions': len(admin_sessions),
            'statistics': {
                'guardians': guardian_count,
                'drivers': driver_count,
                'alerts': alert_count,
                'drowsiness_events_today': drowsiness_events_today
            }
        })
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
                if IS_RENDER and POSTGRES_AVAILABLE:
                    drivers_list.append(dict(driver))
                else:
                    drivers_list.append(dict(driver))
            
            cursor.execute('SELECT COUNT(*) as count FROM drivers')
            count_result = cursor.fetchone()
            count = count_result['count'] if IS_RENDER and POSTGRES_AVAILABLE else count_result[0]
            
            cursor.execute('SELECT COUNT(*) as active_count FROM drivers WHERE is_active = 1')
            active_result = cursor.fetchone()
            active_count = active_result['active_count'] if IS_RENDER and POSTGRES_AVAILABLE else active_result[0]
            
        return jsonify({
            'success': True,
            'count': count,
            'active_count': active_count,
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
            except:
                cursor.execute('''
                    SELECT a.*, d.name as driver_name, g.full_name as guardian_name
                    FROM alerts a
                    JOIN drivers d ON a.driver_id = d.driver_id
                    JOIN guardians g ON a.guardian_id = g.guardian_id
                    ORDER BY a.timestamp DESC
                    LIMIT ?
                ''', (limit,))
            
            alerts = cursor.fetchall()
            
            # Parse detection details
            result = []
            for alert in alerts:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    alert_dict = dict(alert)
                else:
                    alert_dict = dict(alert)
                if alert_dict.get('detection_details'):
                    try:
                        alert_dict['detection_details'] = json.loads(alert_dict['detection_details'])
                    except:
                        pass
                result.append(alert_dict)
            
            cursor.execute('SELECT COUNT(*) as count FROM alerts')
            count_result = cursor.fetchone()
            count = count_result['count'] if IS_RENDER and POSTGRES_AVAILABLE else count_result[0]
            
            cursor.execute('SELECT COUNT(*) as unread_count FROM alerts WHERE acknowledged = 0')
            unread_result = cursor.fetchone()
            unread_count = unread_result['unread_count'] if IS_RENDER and POSTGRES_AVAILABLE else unread_result[0]
            
            try:
                cursor.execute('SELECT COUNT(*) as today_count FROM alerts WHERE DATE(timestamp) = CURRENT_DATE')
            except:
                cursor.execute('SELECT COUNT(*) as today_count FROM alerts WHERE DATE(timestamp) = DATE("now")')
            today_result = cursor.fetchone()
            today_count = today_result['today_count'] if IS_RENDER and POSTGRES_AVAILABLE else today_result[0]
            
        return jsonify({
            'success': True,
            'count': count,
            'unread_count': unread_count,
            'today_count': today_count,
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
            except:
                cursor.execute('''
                    SELECT g.*, 
                           (SELECT COUNT(*) FROM drivers d WHERE d.guardian_id = g.guardian_id) as driver_count,
                           (SELECT COUNT(*) FROM alerts a WHERE a.guardian_id = g.guardian_id) as alert_count
                    FROM guardians g
                    ORDER BY g.registration_date DESC
                ''')
            
            guardians = cursor.fetchall()
            guardians_list = []
            for guardian in guardians:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    guardians_list.append(dict(guardian))
                else:
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
            if IS_RENDER and POSTGRES_AVAILABLE:
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    ORDER BY table_name
                """)
            else:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            
            tables = cursor.fetchall()
            
            result = []
            for table in tables:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    table_name = table['table_name']
                else:
                    table_name = table['name']
                
                cursor.execute(f'SELECT COUNT(*) as count FROM {table_name}')
                count_result = cursor.fetchone()
                count = count_result['count'] if IS_RENDER and POSTGRES_AVAILABLE else count_result[0]
                
                if IS_RENDER and POSTGRES_AVAILABLE:
                    cursor.execute("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = %s 
                        ORDER BY ordinal_position
                    """, (table_name,))
                else:
                    cursor.execute(f'PRAGMA table_info({table_name})')
                
                columns = cursor.fetchall()
                
                if IS_RENDER and POSTGRES_AVAILABLE:
                    columns_list = [{'name': col['column_name'], 'type': col['data_type']} for col in columns]
                else:
                    columns_list = [{'name': col[1], 'type': col[2]} for col in columns]
                
                result.append({
                    'table_name': table_name,
                    'row_count': count,
                    'columns': columns_list
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
            cursor.execute('SELECT COUNT(*) as total_alerts FROM alerts')
            total_alerts_result = cursor.fetchone()
            total_alerts = total_alerts_result['total_alerts'] if IS_RENDER and POSTGRES_AVAILABLE else total_alerts_result[0]
            
            try:
                cursor.execute('SELECT COUNT(*) as today_alerts FROM alerts WHERE DATE(timestamp) = CURRENT_DATE')
            except:
                cursor.execute('SELECT COUNT(*) as today_alerts FROM alerts WHERE DATE(timestamp) = DATE("now")')
            today_alerts_result = cursor.fetchone()
            today_alerts = today_alerts_result['today_alerts'] if IS_RENDER and POSTGRES_AVAILABLE else today_alerts_result[0]
            
            cursor.execute('SELECT COUNT(*) as total_drivers FROM drivers')
            total_drivers_result = cursor.fetchone()
            total_drivers = total_drivers_result['total_drivers'] if IS_RENDER and POSTGRES_AVAILABLE else total_drivers_result[0]
            
            cursor.execute('SELECT COUNT(*) as total_guardians FROM guardians')
            total_guardians_result = cursor.fetchone()
            total_guardians = total_guardians_result['total_guardians'] if IS_RENDER and POSTGRES_AVAILABLE else total_guardians_result[0]
            
            try:
                cursor.execute('SELECT COUNT(*) as drowsiness_events FROM drowsiness_events WHERE DATE(timestamp) = CURRENT_DATE')
            except:
                cursor.execute('SELECT COUNT(*) as drowsiness_events FROM drowsiness_events WHERE DATE(timestamp) = DATE("now")')
            drowsiness_events_result = cursor.fetchone()
            drowsiness_events = drowsiness_events_result['drowsiness_events'] if IS_RENDER and POSTGRES_AVAILABLE else drowsiness_events_result[0]
            
            try:
                cursor.execute('SELECT COUNT(*) as active_sessions FROM session_tokens WHERE is_valid = 1 AND expires_at > %s', 
                             (datetime.now(),))
            except:
                cursor.execute('SELECT COUNT(*) as active_sessions FROM session_tokens WHERE is_valid = 1 AND expires_at > ?', 
                             (datetime.now(),))
            active_sessions_result = cursor.fetchone()
            active_sessions_count = active_sessions_result['active_sessions'] if IS_RENDER and POSTGRES_AVAILABLE else active_sessions_result[0]
            
            # Get recent activity
            cursor.execute('''
                SELECT action, details, timestamp 
                FROM activity_log 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            recent_activity = cursor.fetchall()
            recent_activity_list = []
            for activity in recent_activity:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    recent_activity_list.append(dict(activity))
                else:
                    recent_activity_list.append(dict(activity))
            
            # Get admin activity
            cursor.execute('''
                SELECT admin_username, action, details, timestamp 
                FROM admin_activity_log 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            admin_activity = cursor.fetchall()
            admin_activity_list = []
            for activity in admin_activity:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    admin_activity_list.append(dict(activity))
                else:
                    admin_activity_list.append(dict(activity))
            
            # Get alert statistics by severity
            cursor.execute('''
                SELECT severity, COUNT(*) as count 
                FROM alerts 
                GROUP BY severity
            ''')
            severity_stats = cursor.fetchall()
            severity_stats_list = []
            for stat in severity_stats:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    severity_stats_list.append(dict(stat))
                else:
                    severity_stats_list.append(dict(stat))
            
            # Get daily alerts for last 7 days
            try:
                cursor.execute('''
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM alerts 
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                ''')
            except:
                cursor.execute('''
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM alerts 
                    WHERE timestamp >= datetime('now', '-7 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                ''')
            weekly_alerts = cursor.fetchall()
            weekly_alerts_list = []
            for alert in weekly_alerts:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    weekly_alerts_list.append(dict(alert))
                else:
                    weekly_alerts_list.append(dict(alert))
        
        # Get system status
        system_status = {
            'database': 'postgresql' if (IS_RENDER and POSTGRES_AVAILABLE) else 'sqlite',
            'connected_clients': len(connected_clients),
            'admin_sessions': len(admin_sessions),
            'server_time': datetime.now().isoformat(),
            'uptime': time.time() - app_start_time if 'app_start_time' in globals() else 0
        }
        
        return jsonify({
            'success': True,
            'statistics': {
                'total_alerts': total_alerts,
                'today_alerts': today_alerts,
                'total_drivers': total_drivers,
                'total_guardians': total_guardians,
                'drowsiness_events_today': drowsiness_events,
                'active_sessions': active_sessions_count
            },
            'system_status': system_status,
            'severity_stats': severity_stats_list,
            'weekly_alerts': weekly_alerts_list,
            'recent_activity': recent_activity_list,
            'admin_activity': admin_activity_list
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
                    WHERE acknowledged = TRUE AND timestamp < CURRENT_DATE - INTERVAL '%s days'
                ''', (days,))
            except:
                try:
                    cursor.execute('''
                        DELETE FROM alerts 
                        WHERE acknowledged = 1 AND timestamp < datetime('now', ?)
                    ''', (f'-{days} days',))
                except:
                    cursor.execute('''
                        DELETE FROM alerts 
                        WHERE acknowledged = 1 AND timestamp < datetime('now', ?)
                    ''', (f'-{days} days',))
            
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
    """Guardian login"""
    try:
        data = request.json
        phone = data.get('phone')
        password = data.get('password')
        
        if not phone or not password:
            return jsonify({
                'success': False,
                'error': 'Phone and password required'
            }), 400
        
        ip = request.remote_addr
        if rate_limit_exceeded(ip, 'guardian_login', limit=10):
            return jsonify({
                'success': False,
                'error': 'Too many login attempts'
            }), 429
        
        # This will now use the updated verify_guardian_credentials
        guardian = verify_guardian_credentials(phone, password)
        
        if guardian:
            if 'error' in guardian and guardian['error'] == 'Account locked':
                return jsonify({
                    'success': False,
                    'error': 'Account locked. Please try again later.',
                    'locked_until': guardian.get('locked_until')
                }), 423
            
            token = create_session(guardian['guardian_id'], request.remote_addr, 
                                 request.headers.get('User-Agent'))
            
            if token:
                log_activity(guardian['guardian_id'], 'LOGIN', 
                            f'Guardian logged in from {request.remote_addr}')
                
                return jsonify({
                    'success': True,
                    'guardian_id': guardian['guardian_id'],
                    'full_name': guardian['full_name'],
                    'phone': guardian['phone'],
                    'session_token': token,
                    'message': 'Login successful'
                })
        
        return jsonify({
            'success': False,
            'error': 'Invalid phone or password'
        }), 401
        
    except Exception as e:
        print(f"❌ Login error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Login failed. Please try again.'
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
    """Register a new guardian"""
    try:
        data = request.json
        
        required = ['full_name', 'phone', 'password']
        for field in required:
            if not data.get(field):
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Validate phone number - ACCEPT FILIPINO FORMATS
        phone = data['phone'].strip()
        
        # Remove any spaces, dashes, or parentheses
        phone = re.sub(r'[\s\-\(\)]', '', phone)
        
        # Accept Filipino phone numbers:
        # - 09XXXXXXXXX (11 digits starting with 09)
        # - +639XXXXXXXXX (13 digits starting with +639)
        # - 639XXXXXXXXX (12 digits starting with 639)
        filipino_pattern = r'^(\+?63|0)9\d{9}$'
        
        if not re.match(filipino_pattern, phone):
            return jsonify({
                'success': False,
                'error': 'Invalid Philippine phone number format. Use: 09XXXXXXXXX or +639XXXXXXXXX (11 digits total)'
            }), 400
        
        # Validate length
        if len(phone) < 11 or len(phone) > 13:
            return jsonify({
                'success': False,
                'error': 'Phone number must be 11-13 digits'
            }), 400
        
        # Normalize phone number for storage (convert to +639 format)
        if phone.startswith('0'):
            # Convert 09XXXXXXXXX to +639XXXXXXXXX
            normalized_phone = '+63' + phone[1:]
        elif phone.startswith('63') and not phone.startswith('+'):
            # Convert 639XXXXXXXXX to +639XXXXXXXXX
            normalized_phone = '+' + phone
        else:
            normalized_phone = phone
        
        with get_db_cursor() as cursor:
            # Check if phone already exists (check all possible formats)
            check_phones = [
                phone,  # original
                normalized_phone,  # normalized
                '0' + normalized_phone[3:] if normalized_phone.startswith('+63') else '',  # 0 format
                normalized_phone[1:] if normalized_phone.startswith('+') else ''  # without +
            ]
            
            # Filter out empty strings
            check_phones = [p for p in check_phones if p]
            
            # Build query with all phone formats to check
            placeholders = ', '.join(['%s'] * len(check_phones)) if IS_RENDER and POSTGRES_AVAILABLE else ', '.join(['?'] * len(check_phones))
            query = f'SELECT guardian_id FROM guardians WHERE phone IN ({placeholders})'
            
            cursor.execute(query, check_phones)
            
            if cursor.fetchone():
                return jsonify({
                    'success': False,
                    'error': 'Phone number already registered'
                }), 409
            
            password_hash = hash_password(data['password'])
            
            try:
                cursor.execute('''
                    INSERT INTO guardians (full_name, phone, email, password_hash, address, last_login)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (
                    data['full_name'],
                    normalized_phone,  # Store in normalized format
                    data.get('email', ''),
                    password_hash,
                    data.get('address', ''),
                    datetime.now()
                ))
            except:
                cursor.execute('''
                    INSERT INTO guardians (full_name, phone, email, password_hash, address, last_login)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    data['full_name'],
                    normalized_phone,  # Store in normalized format
                    data.get('email', ''),
                    password_hash,
                    data.get('address', ''),
                    datetime.now()
                ))
            
            try:
                cursor.execute('SELECT LASTVAL()')
                guardian_id = cursor.fetchone()[0]
            except:
                guardian_id = cursor.lastrowid
        
        print(f"✅ Guardian registered: {data['full_name']} (ID: {guardian_id}, Phone: {normalized_phone})")
        
        return jsonify({
            'success': True,
            'guardian_id': guardian_id,
            'full_name': data['full_name'],
            'phone': normalized_phone,
            'message': 'Registration successful'
        })
        
    except Exception as e:
        print(f"❌ Guardian registration error: {e}")
        import traceback
        traceback.print_exc()
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
                cursor.execute('SELECT COUNT(*) FROM drivers WHERE guardian_id = %s', (guardian_id,))
            except:
                cursor.execute('SELECT COUNT(*) FROM drivers WHERE guardian_id = ?', (guardian_id,))
            driver_count_result = cursor.fetchone()
            driver_count = driver_count_result[0]
            
            try:
                cursor.execute('SELECT COUNT(*) FROM alerts WHERE guardian_id = %s', (guardian_id,))
            except:
                cursor.execute('SELECT COUNT(*) FROM alerts WHERE guardian_id = ?', (guardian_id,))
            total_alerts_result = cursor.fetchone()
            total_alerts = total_alerts_result[0]
            
            try:
                cursor.execute('SELECT COUNT(*) FROM alerts WHERE guardian_id = %s AND acknowledged = 0', 
                             (guardian_id,))
            except:
                cursor.execute('SELECT COUNT(*) FROM alerts WHERE guardian_id = ? AND acknowledged = 0', 
                             (guardian_id,))
            unread_alerts_result = cursor.fetchone()
            unread_alerts = unread_alerts_result[0]
            
            try:
                cursor.execute('''
                    SELECT COUNT(*) FROM alerts 
                    WHERE guardian_id = %s AND DATE(timestamp) = CURRENT_DATE
                ''', (guardian_id,))
            except:
                cursor.execute('''
                    SELECT COUNT(*) FROM alerts 
                    WHERE guardian_id = ? AND DATE(timestamp) = DATE('now')
                ''', (guardian_id,))
            today_alerts_result = cursor.fetchone()
            today_alerts = today_alerts_result[0]
            
            # Get drowsiness statistics
            try:
                cursor.execute('''
                    SELECT COUNT(*) as drowsy_count,
                           AVG(confidence) as avg_confidence,
                           MAX(timestamp) as last_event
                    FROM drowsiness_events 
                    WHERE guardian_id = %s AND DATE(timestamp) = CURRENT_DATE
                ''', (guardian_id,))
            except:
                cursor.execute('''
                    SELECT COUNT(*) as drowsy_count,
                           AVG(confidence) as avg_confidence,
                           MAX(timestamp) as last_event
                    FROM drowsiness_events 
                    WHERE guardian_id = ? AND DATE(timestamp) = DATE('now')
                ''', (guardian_id,))
            drowsiness_stats_result = cursor.fetchone()
            if drowsiness_stats_result:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    drowsiness_stats = dict(drowsiness_stats_result)
                else:
                    drowsiness_stats = dict(drowsiness_stats_result)
            else:
                drowsiness_stats = {}
        
        return jsonify({
            'success': True,
            'guardian': guardian,
            'session_valid': True,
            'dashboard': {
                'driver_count': driver_count,
                'total_alerts': total_alerts,
                'unread_alerts': unread_alerts,
                'today_alerts': today_alerts,
                'drowsiness_stats': drowsiness_stats,
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
            except:
                cursor.execute('''
                    INSERT INTO drivers (driver_id, name, phone, email, address, 
                                        reference_number, license_number, guardian_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
    """Send drowsiness alert - Enhanced version with detailed detection data"""
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
            # Driver not found - create a temporary record or use demo guardian
            print(f"⚠️ Driver not found: {driver_id}. Using demo guardian.")
            
            # Find demo guardian
            with get_db_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute('SELECT guardian_id, full_name, phone FROM guardians WHERE phone = %s', ('09123456789',))
                except:
                    cursor.execute('SELECT guardian_id, full_name, phone FROM guardians WHERE phone = ?', ('09123456789',))
                
                demo_guardian = cursor.fetchone()
                
                if demo_guardian:
                    if IS_RENDER and POSTGRES_AVAILABLE:
                        guardian_id, guardian_name, guardian_phone = demo_guardian
                    else:
                        guardian_id, guardian_name, guardian_phone = demo_guardian
                    
                    # Create a temporary driver entry if needed
                    temp_driver_id = f"TEMP{int(time.time())}"
                    try:
                        cursor.execute('''
                            INSERT INTO drivers (driver_id, name, phone, guardian_id)
                            VALUES (%s, %s, %s, %s)
                        ''', (temp_driver_id, driver_name, '00000000000', guardian_id))
                    except:
                        cursor.execute('''
                            INSERT INTO drivers (driver_id, name, phone, guardian_id)
                            VALUES (?, ?, ?, ?)
                        ''', (temp_driver_id, driver_name, '00000000000', guardian_id))
                    conn.commit()
                    
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
            try:
                cursor.execute('''
                    INSERT INTO alerts (driver_id, guardian_id, severity, message, detection_details, source)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (driver_id, guardian_id, severity, message, detection_details_json, 'drowsiness_detection'))
            except:
                cursor.execute('''
                    INSERT INTO alerts (driver_id, guardian_id, severity, message, detection_details, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (driver_id, guardian_id, severity, message, detection_details_json, 'drowsiness_detection'))
            
            try:
                cursor.execute('SELECT LASTVAL()')
                alert_id = cursor.fetchone()[0]
            except:
                alert_id = cursor.lastrowid
            
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
            except:
                cursor.execute('''
                    INSERT INTO drowsiness_events (driver_id, guardian_id, confidence, state, ear, mar, perclos)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
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
            try:
                cursor.execute('''
                    INSERT INTO activity_log (guardian_id, action, details)
                    VALUES (%s, %s, %s)
                ''', (guardian_id, 'ALERT_GENERATED', 
                    f'Alert for driver {driver_name}: {message} (Confidence: {confidence:.1%})'))
            except:
                cursor.execute('''
                    INSERT INTO activity_log (guardian_id, action, details)
                    VALUES (?, ?, ?)
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
            guardian_clients = []
            for client_id, client_info in connected_clients.items():
                if client_info.get('guardian_id') == guardian_id and client_info.get('authenticated'):
                    socketio.emit('guardian_alert', alert_data, room=client_id)
                    guardian_clients.append(client_id)
            
            print(f"🚨 Alert #{alert_id} sent for {driver_name}")
            print(f"   → Guardian: {guardian_name} (Phone: {guardian_phone})")
            print(f"   → Connected clients: {len(guardian_clients)}")
            print(f"   → Details: {detection_details}")
            
            return jsonify({
                'success': True,
                'alert_id': alert_id,
                'data': alert_data,
                'message': f'Alert sent to guardian {guardian_name}'
            })
        
    except Exception as e:
        print(f"❌ Error in send_alert: {e}")
        import traceback
        traceback.print_exc()
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
            query = '''
                SELECT a.*, d.name as driver_name,
                       CASE 
                           WHEN a.detection_details IS NOT NULL AND a.detection_details != '' 
                           THEN detection_details->>'state'
                           ELSE 'alert'
                       END as alert_type
                FROM alerts a
                JOIN drivers d ON a.driver_id = d.driver_id
                WHERE a.guardian_id = %s
            '''
            
            params = [guardian_id]
            
            if acknowledged is not None:
                if acknowledged.lower() == 'true':
                    query += ' AND a.acknowledged = TRUE'
                elif acknowledged.lower() == 'false':
                    query += ' AND a.acknowledged = FALSE'
            
            query += ' ORDER BY a.timestamp DESC LIMIT %s'
            params.append(limit)
            
            try:
                cursor.execute(query, params)
            except:
                # Fallback to SQLite syntax
                query = query.replace('%s', '?')
                query = query.replace('TRUE', '1').replace('FALSE', '0')
                query = query.replace("detection_details->>'state'", "json_extract(a.detection_details, '$.state')")
                cursor.execute(query, params)
            
            alerts = cursor.fetchall()
            
            # Parse detection details
            result_alerts = []
            for alert in alerts:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    alert_dict = dict(alert)
                else:
                    alert_dict = dict(alert)
                if alert_dict.get('detection_details'):
                    try:
                        alert_dict['detection_details'] = json.loads(alert_dict['detection_details'])
                    except:
                        pass
                result_alerts.append(alert_dict)
            
            # Get counts
            try:
                cursor.execute('SELECT COUNT(*) FROM alerts WHERE guardian_id = %s', (guardian_id,))
            except:
                cursor.execute('SELECT COUNT(*) FROM alerts WHERE guardian_id = ?', (guardian_id,))
            total_count_result = cursor.fetchone()
            total_count = total_count_result[0]
            
            try:
                cursor.execute('SELECT COUNT(*) FROM alerts WHERE guardian_id = %s AND acknowledged = 0', 
                             (guardian_id,))
            except:
                cursor.execute('SELECT COUNT(*) FROM alerts WHERE guardian_id = ? AND acknowledged = 0', 
                             (guardian_id,))
            unread_count_result = cursor.fetchone()
            unread_count = unread_count_result[0]
            
            return jsonify({
                'success': True,
                'session_valid': True,
                'count': len(alerts),
                'total_count': total_count,
                'unread_count': unread_count,
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
            except:
                cursor.execute('''
                    SELECT a.*, d.name as driver_name 
                    FROM alerts a
                    JOIN drivers d ON a.driver_id = d.driver_id
                    WHERE a.alert_id = ? AND a.guardian_id = ?
                ''', (alert_id, guardian_id))
            
            alert_result = cursor.fetchone()
            if not alert_result:
                return jsonify({
                    'success': False,
                    'error': 'Alert not found or not authorized'
                }), 404
            
            if IS_RENDER and POSTGRES_AVAILABLE:
                alert = dict(alert_result)
            else:
                alert = dict(alert_result)
            
            # Acknowledge the alert
            try:
                cursor.execute('''
                    UPDATE alerts SET acknowledged = TRUE 
                    WHERE alert_id = %s AND guardian_id = %s
                ''', (alert_id, guardian_id))
            except:
                cursor.execute('''
                    UPDATE alerts SET acknowledged = 1 
                    WHERE alert_id = ? AND guardian_id = ?
                ''', (alert_id, guardian_id))
            
            # Log activity
            try:
                cursor.execute('''
                    INSERT INTO activity_log (guardian_id, action, details)
                    VALUES (%s, %s, %s)
                ''', (guardian_id, 'ALERT_ACKNOWLEDGED', 
                    f'Acknowledged alert #{alert_id} for driver {alert["driver_name"]}'))
            except:
                cursor.execute('''
                    INSERT INTO activity_log (guardian_id, action, details)
                    VALUES (?, ?, ?)
                ''', (guardian_id, 'ALERT_ACKNOWLEDGED', 
                    f'Acknowledged alert #{alert_id} for driver {alert["driver_name"]}'))
            
            # Prepare response data
            alert_dict = alert
            if alert_dict.get('detection_details'):
                try:
                    alert_dict['detection_details'] = json.loads(alert_dict['detection_details'])
                except:
                    pass
            
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
            query = '''
                SELECT e.*, d.name as driver_name
                FROM drowsiness_events e
                JOIN drivers d ON e.driver_id = d.driver_id
                WHERE e.guardian_id = %s
            '''
            
            params = [guardian_id]
            
            if driver_id:
                query += ' AND e.driver_id = %s'
                params.append(driver_id)
            
            query += ' ORDER BY e.timestamp DESC LIMIT %s'
            params.append(limit)
            
            try:
                cursor.execute(query, params)
            except:
                # Fallback to SQLite syntax
                query = query.replace('%s', '?')
                cursor.execute(query, params)
            
            events = cursor.fetchall()
            events_list = []
            for event in events:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    events_list.append(dict(event))
                else:
                    events_list.append(dict(event))
            
            # Get statistics
            try:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_events,
                        AVG(confidence) as avg_confidence,
                        COUNT(CASE WHEN state = 'Drowsy' THEN 1 END) as drowsy_count,
                        COUNT(CASE WHEN state = 'Yawning' THEN 1 END) as yawning_count,
                        DATE(timestamp) as date
                    FROM drowsiness_events 
                    WHERE guardian_id = %s AND DATE(timestamp) = CURRENT_DATE
                    GROUP BY DATE(timestamp)
                ''', (guardian_id,))
            except:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_events,
                        AVG(confidence) as avg_confidence,
                        COUNT(CASE WHEN state = 'Drowsy' THEN 1 END) as drowsy_count,
                        COUNT(CASE WHEN state = 'Yawning' THEN 1 END) as yawning_count,
                        DATE(timestamp) as date
                    FROM drowsiness_events 
                    WHERE guardian_id = ? AND DATE(timestamp) = DATE('now')
                    GROUP BY DATE(timestamp)
                ''', (guardian_id,))
            
            stats_result = cursor.fetchone()
            if stats_result:
                if IS_RENDER and POSTGRES_AVAILABLE:
                    stats = dict(stats_result)
                else:
                    stats = dict(stats_result)
            else:
                stats = {
                    'total_events': 0,
                    'avg_confidence': 0,
                    'drowsy_count': 0,
                    'yawning_count': 0
                }
            
            return jsonify({
                'success': True,
                'session_valid': True,
                'count': len(events_list),
                'events': events_list,
                'statistics': stats
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
        
        # Use demo guardian for testing
        with get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('SELECT guardian_id, full_name FROM guardians WHERE phone = %s', ('09123456789',))
            except:
                cursor.execute('SELECT guardian_id, full_name FROM guardians WHERE phone = ?', ('09123456789',))
            
            demo_guardian_result = cursor.fetchone()
            
            if not demo_guardian_result:
                return jsonify({
                    'success': False,
                    'error': 'Demo guardian not found'
                }), 404
            
            if IS_RENDER and POSTGRES_AVAILABLE:
                guardian_id, guardian_name = demo_guardian_result
            else:
                guardian_id, guardian_name = demo_guardian_result
            
            # Create test driver if needed
            test_driver_id = data.get('driver_id', 'TEST123')
            test_driver_name = data.get('driver_name', 'Test Driver')
            
            try:
                cursor.execute('SELECT driver_id FROM drivers WHERE driver_id = %s', (test_driver_id,))
            except:
                cursor.execute('SELECT driver_id FROM drivers WHERE driver_id = ?', (test_driver_id,))
            
            if not cursor.fetchone():
                try:
                    cursor.execute('''
                        INSERT INTO drivers (driver_id, name, phone, guardian_id)
                        VALUES (%s, %s, %s, %s)
                    ''', (test_driver_id, test_driver_name, '00000000000', guardian_id))
                except:
                    cursor.execute('''
                        INSERT INTO drivers (driver_id, name, phone, guardian_id)
                        VALUES (?, ?, ?, ?)
                    ''', (test_driver_id, test_driver_name, '00000000000', guardian_id))
                conn.commit()
            
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
            
            # Simulate the send_alert endpoint
            from flask import current_app
            with current_app.test_request_context():
                import json as json_module
                test_request_data = json_module.dumps(test_data)
                
                # Call send_alert directly
                response = send_alert()
                
                if response.status_code == 200:
                    return jsonify({
                        'success': True,
                        'message': f'Test alert sent to guardian {guardian_name}',
                        'alert_data': test_data
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Failed to send test alert: {response.status_code}'
                    }), 500
        
    except Exception as e:
        print(f"❌ Test alert error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== STATIC FILES ====================
@app.route('/<path:filename>')
def serve_static(filename):
    """Serve all frontend static files"""
    try:
        return send_from_directory(FRONTEND_DIR, filename)
    except:
        return jsonify({'error': 'File not found'}), 404

# ==================== CLEANUP TASKS ====================
def cleanup_expired_sessions():
    """Clean up expired sessions periodically"""
    try:
        with get_db_cursor() as cursor:
            try:
                cursor.execute('''
                    UPDATE session_tokens SET is_valid = FALSE 
                    WHERE expires_at < %s AND is_valid = TRUE
                ''', (datetime.now(),))
            except:
                cursor.execute('''
                    UPDATE session_tokens SET is_valid = 0 
                    WHERE expires_at < ? AND is_valid = 1
                ''', (datetime.now(),))
            
            expired_count = cursor.rowcount
            
            # Clean up old drowsiness events (keep last 30 days)
            try:
                cursor.execute('''
                    DELETE FROM drowsiness_events 
                    WHERE timestamp < CURRENT_DATE - INTERVAL '30 days'
                ''')
            except:
                cursor.execute('''
                    DELETE FROM drowsiness_events 
                    WHERE timestamp < datetime('now', '-30 days')
                ''')
            
            old_events_count = cursor.rowcount
            
            # Clean up old activity logs (keep last 90 days)
            try:
                cursor.execute('''
                    DELETE FROM activity_log 
                    WHERE timestamp < CURRENT_DATE - INTERVAL '90 days'
                ''')
            except:
                cursor.execute('''
                    DELETE FROM activity_log 
                    WHERE timestamp < datetime('now', '-90 days')
                ''')
            
            old_activity_count = cursor.rowcount
        
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
        
        if (expired_count > 0 or old_events_count > 0) and not IS_PRODUCTION:
            print(f"🧹 Cleaned up {expired_count} expired sessions, {old_events_count} old events, {old_activity_count} old activities")
        
        return expired_count
    except Exception as e:
        print(f"❌ Error cleaning up sessions: {e}")
        return 0

# ==================== APPLICATION STARTUP ====================
def startup_tasks():
    """Run startup tasks"""
    print(f"\n{'='*70}")
    print("🚗 DRIVER DROWSINESS ALERT SYSTEM - CAPSTONE PROJECT")
    print(f"{'='*70}")
    
    if IS_RENDER:
        print("🌐 DEPLOYMENT: Render.com Cloud")
        if POSTGRES_AVAILABLE:
            print("📊 Database: PostgreSQL (Persistent)")
        else:
            print("📊 Database: SQLite (Ephemeral - data may be lost on redeploy)")
    else:
        print("💻 DEPLOYMENT: Local Development")
        print("📊 Database: SQLite")
    
    print(f"🔧 Environment: {'Production' if IS_PRODUCTION else 'Development'}")
    print(f"📁 Frontend Directory: {FRONTEND_DIR}")
    
    if IS_RENDER and POSTGRES_AVAILABLE:
        print("🗄️  Database: PostgreSQL on Render")
    else:
        print(f"🗄️  Database Path: {DB_PATH}")
    
    print(f"🔐 Admin Username: admin")
    print(f"🔐 Admin Password: admin123")
    
    # Initialize database
    if init_db():
        print("✅ Database initialized successfully")
    else:
        print("❌ Database initialization failed")
    
    # Clean up expired sessions
    cleanup_expired_sessions()
    
    # Start cleanup thread
    def cleanup_worker():
        while True:
            time.sleep(3600)  # Run every hour
            cleanup_expired_sessions()
    
    if not IS_PRODUCTION:
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        print("✅ Cleanup worker started")
    
    print(f"{'='*70}")
    print("📱 DEMO CREDENTIALS:")
    print("  Phone: 09123456789")
    print("  Password: demo123")
    print(f"{'='*70}")
    print("🔗 API Endpoints:")
    print("  • POST /api/send-alert - Receive alerts from drowsiness detection")
    print("  • POST /api/admin/login - Admin login")
    print("  • GET  /api/health - Health check")
    print("  • POST /api/test-alert - Test alert endpoint")
    print("  • POST /api/login - Guardian login")
    print(f"{'='*70}\n")
    
    # Store app start time for uptime calculation
    global app_start_time
    app_start_time = time.time()

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
    print(f"🔐 Admin login: http://{host}:{port}/admin-login")
    print(f"👤 Guardian login: http://{host}:{port}/")
    
    # Run the application
    socketio.run(app, 
                host=host, 
                port=port, 
                debug=not IS_PRODUCTION,
                use_reloader=not IS_PRODUCTION,
                log_output=not IS_PRODUCTION,
                allow_unsafe_werkzeug=not IS_PRODUCTION)