#!/usr/bin/env python3
"""
Password Hash Generator for Driver Alert System
Use this to generate secure password hashes for admin accounts
"""

import hashlib
import secrets
import sys
import getpass
from datetime import datetime

def generate_password_hash(password, salt=None):
    """Generate SHA-256 hash of password with optional salt"""
    if not salt:
        salt = "dr1v3r_@l3rt_s@lt_"  # Default salt from app.py
    
    return hashlib.sha256((password + salt).encode()).hexdigest()

def generate_secure_password(length=16):
    """Generate a secure random password"""
    import string
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    while True:
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        if (any(c.islower() for c in password) and
            any(c.isupper() for c in password) and
            any(c.isdigit() for c in password) and
            any(c in "!@#$%^&*" for c in password)):
            break
    return password

def generate_admin_credentials():
    """Generate complete admin credentials"""
    print("\n" + "="*60)
    print("🔐 ADMIN CREDENTIALS GENERATOR")
    print("="*60)
    
    # Ask for username
    username = input("\nEnter admin username [default: admin]: ").strip()
    if not username:
        username = "admin"
    
    # Ask for password or generate one
    print("\nPassword options:")
    print("1. Enter custom password")
    print("2. Generate secure password")
    choice = input("\nChoose option (1 or 2): ").strip()
    
    if choice == "1":
        while True:
            password = getpass.getpass("Enter admin password: ")
            confirm = getpass.getpass("Confirm password: ")
            
            if password == confirm:
                if len(password) < 8:
                    print("⚠️  Warning: Password should be at least 8 characters")
                    continue_anyway = input("Continue anyway? (y/n): ").strip().lower()
                    if continue_anyway != 'y':
                        continue
                break
            else:
                print("❌ Passwords don't match. Try again.")
    else:
        password = generate_secure_password()
        print(f"\n✅ Generated secure password: {password}")
    
    # Generate hash
    password_hash = generate_password_hash(password)
    
    # Generate API token
    api_token = secrets.token_urlsafe(32)
    
    # Get admin details
    print("\n" + "-"*60)
    print("📝 Admin Details:")
    full_name = input("Full name [default: System Administrator]: ").strip()
    if not full_name:
        full_name = "System Administrator"
    
    email = input("Email [default: admin@driveralert.com]: ").strip()
    if not email:
        email = "admin@driveralert.com"
    
    role = input("Role [default: super_admin]: ").strip()
    if not role:
        role = "super_admin"
    
    # Display results
    print("\n" + "="*60)
    print("✅ CREDENTIALS GENERATED SUCCESSFULLY")
    print("="*60)
    
    print(f"\n📋 Admin Information:")
    print(f"   Username: {username}")
    print(f"   Full Name: {full_name}")
    print(f"   Email: {email}")
    print(f"   Role: {role}")
    
    print(f"\n🔑 Password (store securely): {password}")
    print(f"🔐 Password Hash (SHA-256): {password_hash}")
    print(f"🔧 API Token: {api_token}")
    
    print("\n" + "-"*60)
    print("📝 Update your app.py with these credentials:")
    print("-"*60)
    
    print(f"""
ADMIN_CREDENTIALS = {{
    '{username}': {{
        'password_hash': '{password_hash}',
        'full_name': '{full_name}',
        'role': '{role}',
        'email': '{email}',
        'created_at': '{datetime.now().isoformat()}'
    }}
}}

# API Token for admin requests
ADMIN_API_TOKEN = '{api_token}'
""")
    
    print("\n" + "-"*60)
    print("📋 Security Recommendations:")
    print("-"*60)
    print("1. ✅ Change the default salt in app.py")
    print("2. ✅ Use environment variables in production")
    print("3. ✅ Store passwords securely")
    print("4. ✅ Regularly rotate passwords")
    print("5. ✅ Enable two-factor authentication if possible")
    
    # Save to file (optional)
    save = input("\n💾 Save to credentials.txt? (y/n): ").strip().lower()
    if save == 'y':
        with open('admin_credentials.txt', 'w') as f:
            f.write(f"""ADMIN CREDENTIALS - DRIVER ALERT SYSTEM
Generated: {datetime.now().isoformat()}

Username: {username}
Password: {password}
Full Name: {full_name}
Email: {email}
Role: {role}

Password Hash: {password_hash}
API Token: {api_token}

SECURITY NOTES:
- Store this file securely
- Delete after updating app.py
- Never commit to version control
""")
        print("✅ Saved to admin_credentials.txt")
        print("⚠️  Remember to delete this file after use!")

def generate_guardian_password():
    """Generate password hash for guardians"""
    print("\n" + "="*60)
    print("👤 GUARDIAN PASSWORD GENERATOR")
    print("="*60)
    
    phone = input("\nEnter guardian phone number: ").strip()
    if not phone:
        print("❌ Phone number is required")
        return
    
    while True:
        password = getpass.getpass("Enter guardian password: ")
        confirm = getpass.getpass("Confirm password: ")
        
        if password == confirm:
            break
        else:
            print("❌ Passwords don't match. Try again.")
    
    password_hash = generate_password_hash(password)
    
    print("\n" + "="*60)
    print("✅ PASSWORD HASH GENERATED")
    print("="*60)
    
    print(f"\n📱 Phone: {phone}")
    print(f"🔐 Password Hash: {password_hash}")
    
    print("\n📝 SQL Insert Statement:")
    print(f"""
INSERT INTO guardians (phone, password_hash, full_name, email, address)
VALUES ('{phone}', '{password_hash}', 'Guardian Name', 'email@example.com', 'Address');
""")

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🔐 PASSWORD MANAGEMENT TOOL - DRIVER ALERT SYSTEM")
    print("="*60)
    
    print("\nSelect option:")
    print("1. Generate Admin Credentials")
    print("2. Generate Guardian Password Hash")
    print("3. Hash Existing Password")
    print("4. Generate Secure Password")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                generate_admin_credentials()
                break
            elif choice == "2":
                generate_guardian_password()
                break
            elif choice == "3":
                password = getpass.getpass("Enter password to hash: ")
                password_hash = generate_password_hash(password)
                print(f"\n🔐 Password Hash: {password_hash}")
                break
            elif choice == "4":
                length = input("Password length [default: 16]: ").strip()
                if length.isdigit():
                    length = int(length)
                else:
                    length = 16
                password = generate_secure_password(length)
                print(f"\n🔑 Generated Password: {password}")
                break
            elif choice == "5":
                print("\n👋 Goodbye!")
                sys.exit(0)
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Operation cancelled.")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()