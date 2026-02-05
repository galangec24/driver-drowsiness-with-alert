"""
CAPSTONE PROJECT - DRIVER ALERT SYSTEM
Render.com Deployment Ready Version
"""

from flask import Flask, request, jsonify, send_from_directory, Response, redirect, g
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sqlite3
from datetime import datetime, timedelta
import uuid
import eventlet
import socket
import base64
import json
import hashlib
import secrets
import threading
import time
from contextlib import contextmanager

# ==================== RENDER DEPLOYMENT CONFIG ====================
# Detect if running on Render
IS_RENDER = os.environ.get('RENDER') is not None
IS_PRODUCTION = os.environ.get('ENVIRONMENT') == 'production' or IS_RENDER

# Set base directory
if IS_RENDER:
    # Render deployment structure
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')  # Frontend in parent directory
    DB_PATH = os.path.join(BASE_DIR, 'drivers.db')
    FACE_IMAGES_DIR = os.path.join(BASE_DIR, 'face_images')
    SESSION_TOKENS_DIR = os.path.join(BASE_DIR, 'session_tokens')
    DATA_DIR = BASE_DIR  # Use app directory for data on Render
else:
    # Local development structure
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
             "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
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

# ==================== DATABASE CONNECTION MANAGEMENT ====================
def get_db():
    """Get database connection with thread-local storage"""
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = sqlite3.connect(app.config['DATABASE'], timeout=20)
        g.sqlite_db.row_factory = sqlite3.Row
        # Optimize for concurrency
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
    conn = sqlite3.connect(app.config['DATABASE'], timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_db_cursor():
    """Context manager for database cursor"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"❌ Database error: {e}")
            raise e
        finally:
            cursor.close()

# Store connected clients and active sessions
connected_clients = {}
active_sessions = {}
db_write_lock = threading.Lock()

# ==================== DATABASE INITIALIZATION ====================
def init_db():
    """Initialize database with all required tables"""
    try:
        with get_db_cursor() as cursor:
            
            # Guardians table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS guardians (
                guardian_id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                phone TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                address TEXT,
                registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
            ''')
            
            # Main drivers table
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
                FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
            )
            ''')
            
            # Alerts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id TEXT NOT NULL,
                guardian_id INTEGER,
                severity TEXT NOT NULL CHECK(severity IN ('low', 'medium', 'high')),
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                acknowledged BOOLEAN DEFAULT 0,
                FOREIGN KEY (driver_id) REFERENCES drivers(driver_id) ON DELETE CASCADE,
                FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
            )
            ''')
            
            # Face images table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                capture_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (driver_id) REFERENCES drivers(driver_id) ON DELETE CASCADE
            )
            ''')
            
            # Drowsiness events table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS drowsiness_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id TEXT NOT NULL,
                guardian_id INTEGER,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT 0,
                FOREIGN KEY (driver_id) REFERENCES drivers(driver_id) ON DELETE CASCADE,
                FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
            )
            ''')
            
            # Activity log table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guardian_id INTEGER,
                action TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
            )
            ''')
            
            # Session tokens table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_tokens (
                token_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guardian_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_valid BOOLEAN DEFAULT 1,
                FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id) ON DELETE CASCADE
            )
            ''')
            
            # Check and add last_login column if not exists
            try:
                cursor.execute("SELECT last_login FROM guardians LIMIT 1")
            except sqlite3.OperationalError:
                print("📝 Adding last_login column to guardians table...")
                cursor.execute('ALTER TABLE guardians ADD COLUMN last_login TIMESTAMP')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_drivers_guardian ON drivers(guardian_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_guardian ON alerts(guardian_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tokens_expires ON session_tokens(expires_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tokens_valid ON session_tokens(is_valid)')
            
            # Insert demo guardian if not exists
            cursor.execute('SELECT COUNT(*) FROM guardians WHERE phone = ?', ('09123456789',))
            if cursor.fetchone()[0] == 0:
                demo_password_hash = hash_password('demo123')
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
        return False

# ==================== SESSION MANAGEMENT ====================
def generate_session_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def create_session(guardian_id):
    """Create a new session for guardian"""
    token = generate_session_token()
    expires_at = datetime.now() + timedelta(hours=24)
    
    try:
        with get_db_cursor() as cursor:
            # Invalidate any existing sessions for this guardian
            cursor.execute('''
                UPDATE session_tokens SET is_valid = 0 
                WHERE guardian_id = ? AND is_valid = 1
            ''', (guardian_id,))
            
            # Create new session
            cursor.execute('''
                INSERT INTO session_tokens (guardian_id, token, expires_at)
                VALUES (?, ?, ?)
            ''', (guardian_id, token, expires_at))
        
        # Store in memory for quick access
        active_sessions[guardian_id] = {
            'token': token,
            'expires': expires_at,
            'created': datetime.now()
        }
        
        return token
    except Exception as e:
        print(f"❌ Error creating session: {e}")
        return None

def validate_session(guardian_id, token):
    """Validate a guardian session"""
    if not guardian_id or not token:
        return False
    
    # Check memory cache first (fast path)
    if guardian_id in active_sessions:
        session_data = active_sessions[guardian_id]
        if (session_data['token'] == token and 
            session_data['expires'] > datetime.now()):
            return True
    
    # Check database (slower path)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM session_tokens 
                WHERE guardian_id = ? AND token = ? AND is_valid = 1 AND expires_at > ?
            ''', (guardian_id, token, datetime.now()))
            
            result = cursor.fetchone()[0] > 0
            
            # Update memory cache if valid
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
                cursor.execute('''
                    UPDATE session_tokens SET is_valid = 0 
                    WHERE guardian_id = ? AND token = ?
                ''', (guardian_id, token))
            else:
                cursor.execute('''
                    UPDATE session_tokens SET is_valid = 0 
                    WHERE guardian_id = ?
                ''', (guardian_id,))
        
        # Remove from memory cache
        if guardian_id in active_sessions:
            del active_sessions[guardian_id]
        
        # Disconnect socket connections for this guardian
        disconnected = []
        for client_id, client_info in connected_clients.items():
            if client_info.get('guardian_id') == guardian_id:
                disconnected.append(client_id)
        
        # Remove from connected clients
        for client_id in disconnected:
            if client_id in connected_clients:
                del connected_clients[client_id]
        
        return True
    except Exception as e:
        print(f"❌ Error invalidating session: {e}")
        return False

# ==================== AUTHENTICATION FUNCTIONS ====================
def hash_password(password):
    """Hash password using SHA-256 with salt"""
    salt = "dr1v3r_@l3rt_s@lt_" + os.environ.get('PEPPER', 'default_pepper')
    return hashlib.sha256((password + salt).encode()).hexdigest()

def verify_guardian_credentials(phone, password):
    """Verify guardian login credentials"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get guardian with password hash
            cursor.execute('''
                SELECT guardian_id, full_name, password_hash 
                FROM guardians 
                WHERE phone = ?
            ''', (phone,))
            
            result = cursor.fetchone()
            
            if result:
                guardian_id, full_name, stored_hash = result
                if hash_password(password) == stored_hash:
                    # Update last login
                    cursor.execute('''
                        UPDATE guardians SET last_login = ? WHERE guardian_id = ?
                    ''', (datetime.now(), guardian_id))
                    conn.commit()
                    
                    return {'guardian_id': guardian_id, 'full_name': full_name, 'phone': phone}
    except Exception as e:
        print(f"❌ Error verifying credentials: {e}")
    
    return None

def get_guardian_by_id(guardian_id):
    """Get guardian information by ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT guardian_id, full_name, phone, email, address, registration_date, last_login
                FROM guardians WHERE guardian_id = ?
            ''', (guardian_id,))
            
            result = cursor.fetchone()
            return dict(result) if result else None
    except Exception as e:
        print(f"❌ Error getting guardian: {e}")
        return None

def log_activity(guardian_id, action, details):
    """Log guardian activity"""
    try:
        with get_db_cursor() as cursor:
            cursor.execute('''
                INSERT INTO activity_log (guardian_id, action, details)
                VALUES (?, ?, ?)
            ''', (guardian_id, action, details))
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
        
        # Save image file
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        # Use database lock to prevent concurrent writes
        with db_write_lock:
            with get_db_cursor() as cursor:
                cursor.execute('''
                    INSERT INTO face_images (driver_id, image_path)
                    VALUES (?, ?)
                ''', (driver_id, image_path))
        
        return image_path
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return None

def get_guardian_drivers(guardian_id):
    """Get all drivers registered by a guardian"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT d.*, 
                       (SELECT COUNT(*) FROM alerts a WHERE a.driver_id = d.driver_id AND a.acknowledged = 0) as alert_count,
                       (SELECT COUNT(*) FROM face_images f WHERE f.driver_id = d.driver_id) as face_count
                FROM drivers d
                WHERE d.guardian_id = ?
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
                WHERE a.guardian_id = ?
                ORDER BY a.timestamp DESC
                LIMIT ?
            ''', (guardian_id, limit))
            
            alerts = cursor.fetchall()
            return [dict(alert) for alert in alerts]
    except Exception as e:
        print(f"❌ Error in get_recent_alerts: {e}")
        return []

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
    
    # Authentication failed
    if not IS_PRODUCTION:
        print(f"🔒 WebSocket authentication failed for client {client_id}")
    emit('auth_failed', {'error': 'Authentication failed'})
    socketio.disconnect(client_id)

# ==================== MAIN ROUTES ====================
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint for debugging"""
    try:
        return jsonify({
            'success': True,
            'message': 'Server is running',
            'environment': 'production' if IS_PRODUCTION else 'development',
            'deployment': 'render' if IS_RENDER else 'local',
            'timestamp': datetime.now().isoformat(),
            'server_time': time.time(),
            'connected_clients': len(connected_clients),
            'active_sessions': len(active_sessions),
            'frontend_dir': FRONTEND_DIR,
            'database_path': DB_PATH
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/')
def serve_home():
    """Main route - Serves login.html for mobile, index.html for desktop"""
    
    # Check URL parameter first (for testing)
    if request.args.get('force') == 'mobile':
        return send_from_directory(FRONTEND_DIR, 'login.html')
    if request.args.get('force') == 'desktop':
        return send_from_directory(FRONTEND_DIR, 'index.html')
    
    # Device detection
    if is_mobile_device():
        return send_from_directory(FRONTEND_DIR, 'login.html')
    else:
        return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/guardian-register')
def serve_guardian_register():
    """Guardian registration page"""
    return send_from_directory(FRONTEND_DIR, 'guardian-register.html')

@app.route('/guardian-dashboard')
def serve_guardian_dashboard():
    """Guardian dashboard page - Check authentication"""
    token = request.args.get('token')
    guardian_id = request.args.get('guardian_id')
    
    if token and guardian_id and validate_session(guardian_id, token):
        response = send_from_directory(FRONTEND_DIR, 'guardian-dashboard.html')
        # Prevent caching
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    else:
        # Redirect to login if not authenticated
        return redirect('/?logged_out=true')

@app.route('/register-driver')
def serve_register_driver():
    """Driver registration page - check authentication"""
    token = request.args.get('token')
    guardian_id = request.args.get('guardian_id')
    
    if token and guardian_id and validate_session(guardian_id, token):
        response = send_from_directory(FRONTEND_DIR, 'register-driver.html')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        return response
    
    return redirect('/?logged_out=true')

@app.route('/admin')
def admin_dashboard():
    """Admin dashboard"""
    return send_from_directory(FRONTEND_DIR, 'index.html')

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
            
            # Check database connectivity
            cursor.execute('SELECT 1')
            db_status = cursor.fetchone()[0] == 1
            
            # Get counts
            cursor.execute('SELECT COUNT(*) FROM guardians')
            guardian_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM drivers')
            driver_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM alerts')
            alert_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM session_tokens WHERE is_valid = 1 AND expires_at > ?', 
                         (datetime.now(),))
            active_sessions_count = cursor.fetchone()[0]
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'server': 'Driver Alert System',
            'version': '2.0.0',
            'environment': 'production' if IS_PRODUCTION else 'development',
            'deployment': 'render' if IS_RENDER else 'local',
            'database': 'connected' if db_status else 'disconnected',
            'connected_clients': len(connected_clients),
            'active_sessions': len(active_sessions),
            'statistics': {
                'guardians': guardian_count,
                'drivers': driver_count,
                'alerts': alert_count,
                'valid_sessions': active_sessions_count
            }
        })
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
        
        guardian = verify_guardian_credentials(phone, password)
        
        if guardian:
            # Create session
            token = create_session(guardian['guardian_id'])
            
            if token:
                log_activity(guardian['guardian_id'], 'LOGIN', 'Guardian logged in')
                
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
        return jsonify({
            'success': False,
            'error': str(e)
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
        
        with get_db_cursor() as cursor:
            # Check if phone already exists
            cursor.execute('SELECT guardian_id FROM guardians WHERE phone = ?', (data['phone'],))
            if cursor.fetchone():
                return jsonify({
                    'success': False,
                    'error': 'Phone number already registered'
                }), 409
            
            password_hash = hash_password(data['password'])
            
            cursor.execute('''
                INSERT INTO guardians (full_name, phone, email, password_hash, address, last_login)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data['full_name'],
                data['phone'],
                data.get('email', ''),
                password_hash,
                data.get('address', ''),
                datetime.now()
            ))
            
            guardian_id = cursor.lastrowid
        
        print(f"✅ Guardian registered: {data['full_name']} (ID: {guardian_id})")
        
        return jsonify({
            'success': True,
            'guardian_id': guardian_id,
            'full_name': data['full_name'],
            'phone': data['phone'],
            'message': 'Registration successful'
        })
        
    except Exception as e:
        print(f"❌ Guardian registration error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
        
        with get_db_cursor() as cursor:
            # Get counts
            cursor.execute('SELECT COUNT(*) FROM drivers WHERE guardian_id = ?', (guardian_id,))
            driver_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM alerts WHERE guardian_id = ?', (guardian_id,))
            total_alerts = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM alerts WHERE guardian_id = ? AND acknowledged = 0', 
                         (guardian_id,))
            unread_alerts = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM alerts 
                WHERE guardian_id = ? AND DATE(timestamp) = DATE('now')
            ''', (guardian_id,))
            today_alerts = cursor.fetchone()[0]
        
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

@app.route('/api/send-alert', methods=['POST'])
def send_alert():
    """Send drowsiness alert"""
    try:
        data = request.json
        driver_id = data.get('driver_id')
        severity = data.get('severity', 'high')
        message = data.get('message', 'Drowsiness detected!')
        confidence = data.get('confidence', 0.0)
        
        if not driver_id:
            return jsonify({
                'success': False,
                'error': 'Driver ID required'
            }), 400
        
        with get_db_cursor() as cursor:
            cursor.execute('''
                SELECT d.name as driver_name, d.guardian_id, g.full_name as guardian_name, g.phone
                FROM drivers d
                JOIN guardians g ON d.guardian_id = g.guardian_id
                WHERE d.driver_id = ?
            ''', (driver_id,))
            
            result = cursor.fetchone()
            
            if result:
                driver_name, guardian_id, guardian_name, guardian_phone = result
                
                # Create alert
                cursor.execute('''
                    INSERT INTO alerts (driver_id, guardian_id, severity, message)
                    VALUES (?, ?, ?, ?)
                ''', (driver_id, guardian_id, severity, message))
                alert_id = cursor.lastrowid
                
                # Log drowsiness event
                cursor.execute('''
                    INSERT INTO drowsiness_events (driver_id, guardian_id, confidence)
                    VALUES (?, ?, ?)
                ''', (driver_id, guardian_id, confidence))
                
                # Log activity
                cursor.execute('''
                    INSERT INTO activity_log (guardian_id, action, details)
                    VALUES (?, ?, ?)
                ''', (guardian_id, 'ALERT_GENERATED', 
                    f'Alert for driver {driver_name}: {message}'))
                
                alert_data = {
                    'alert_id': alert_id,
                    'driver_id': driver_id,
                    'driver_name': driver_name,
                    'guardian_id': guardian_id,
                    'guardian_name': guardian_name,
                    'severity': severity,
                    'message': message,
                    'timestamp': datetime.now().isoformat(),
                    'confidence': confidence
                }
                
                # Emit socket events
                socketio.emit('new_alert', alert_data)
                
                # Send to specific guardian clients
                for client_id, client_info in connected_clients.items():
                    if client_info.get('guardian_id') == guardian_id and client_info.get('authenticated'):
                        socketio.emit('guardian_alert', alert_data, room=client_id)
                
                if not IS_PRODUCTION:
                    print(f"🚨 Alert sent for {driver_name} -> Guardian: {guardian_name}")
                
                return jsonify({
                    'success': True,
                    'alert_id': alert_id,
                    'data': alert_data
                })
        
        return jsonify({
            'success': False,
            'error': 'Driver not found'
        }), 404
        
    except Exception as e:
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
            
            cursor.execute('''
                SELECT a.*, d.name as driver_name
                FROM alerts a
                JOIN drivers d ON a.driver_id = d.driver_id
                WHERE a.guardian_id = ?
                ORDER BY a.timestamp DESC
                LIMIT ?
            ''', (guardian_id, limit))
            
            alerts = cursor.fetchall()
            
            return jsonify({
                'success': True,
                'session_valid': True,
                'count': len(alerts),
                'alerts': [dict(alert) for alert in alerts]
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'redirect': '/?logged_out=true'
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
            cursor.execute('''
                UPDATE session_tokens SET is_valid = 0 
                WHERE expires_at < ? AND is_valid = 1
            ''', (datetime.now(),))
            
            expired_count = cursor.rowcount
        
        # Clean memory cache
        current_time = datetime.now()
        expired_guards = []
        for guardian_id, session_data in active_sessions.items():
            if session_data['expires'] < current_time:
                expired_guards.append(guardian_id)
        
        for guardian_id in expired_guards:
            del active_sessions[guardian_id]
        
        if expired_count > 0 and not IS_PRODUCTION:
            print(f"🧹 Cleaned up {expired_count} expired sessions")
        
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
        print(f"📱 App URL: https://your-app-name.onrender.com")
    else:
        print("💻 DEPLOYMENT: Local Development")
    
    print(f"🔧 Environment: {'Production' if IS_PRODUCTION else 'Development'}")
    print(f"📁 Frontend Directory: {FRONTEND_DIR}")
    print(f"🗄️  Database Path: {DB_PATH}")
    
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
    
    # Run the application
    socketio.run(app, 
                host=host, 
                port=port, 
                debug=not IS_PRODUCTION,
                use_reloader=not IS_PRODUCTION,
                log_output=not IS_PRODUCTION,
                allow_unsafe_werkzeug=not IS_PRODUCTION)