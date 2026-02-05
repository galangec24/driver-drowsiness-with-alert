"""
Complete Database Rebuild Script
Rebuilds the drivers table with all required columns
"""

import sqlite3
import os
import json
from datetime import datetime

# Path to your database
DB_PATH = os.path.join(os.path.dirname(__file__), 'drivers.db')

def backup_database():
    """Create a backup of the database"""
    import shutil
    import datetime as dt
    
    backup_dir = os.path.join(os.path.dirname(__file__), 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = os.path.join(backup_dir, f'drivers_full_backup_{timestamp}.db')
    
    try:
        if os.path.exists(DB_PATH):
            shutil.copy2(DB_PATH, backup_file)
            print(f"📁 Backup created: {backup_file}")
            return True
        else:
            print("❌ Database file not found!")
            return False
    except Exception as e:
        print(f"❌ Could not create backup: {e}")
        return False

def export_existing_data():
    """Export existing data from drivers table"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Get all data from drivers table
        cursor.execute("SELECT * FROM drivers")
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        data = [dict(row) for row in rows]
        
        # Save to JSON file
        export_file = os.path.join(os.path.dirname(__file__), 'backups', f'drivers_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        with open(export_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"📊 Exported {len(data)} drivers to: {export_file}")
        return data
        
    except Exception as e:
        print(f"❌ Error exporting data: {e}")
        return []
    finally:
        conn.close()

def rebuild_drivers_table():
    """Completely rebuild the drivers table with proper schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("🔨 Rebuilding drivers table...")
    print("=" * 60)
    
    # 1. Export existing data
    print("\n1. Exporting existing data...")
    existing_data = export_existing_data()
    
    # 2. Create new table with correct schema
    print("\n2. Creating new drivers table...")
    
    # Drop the old table
    try:
        cursor.execute('DROP TABLE IF EXISTS drivers_old')
        print("   ✅ Dropped backup table")
    except:
        pass
    
    # Rename current table to backup
    try:
        cursor.execute('ALTER TABLE drivers RENAME TO drivers_old')
        print("   ✅ Renamed old table to drivers_old")
    except Exception as e:
        print(f"   ❌ Error renaming table: {e}")
        # Try to drop and recreate if rename fails
        cursor.execute('DROP TABLE IF EXISTS drivers')
        print("   ✅ Dropped old drivers table")
    
    # Create new table with correct schema
    cursor.execute('''
    CREATE TABLE drivers (
        driver_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        address TEXT,
        phone TEXT NOT NULL,
        email TEXT,
        reference_number TEXT UNIQUE,
        license_number TEXT,
        guardian_id INTEGER,
        registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id)
    )
    ''')
    print("   ✅ Created new drivers table with correct schema")
    
    # 3. Restore data with proper values
    print("\n3. Restoring data...")
    
    if existing_data:
        imported_count = 0
        for driver in existing_data:
            try:
                # Generate missing values
                driver_id = driver.get('driver_id')
                if not driver_id:
                    import uuid
                    driver_id = f"DRV{uuid.uuid4().hex[:8].upper()}"
                
                # Generate unique reference number if missing
                reference_number = driver.get('reference_number')
                if not reference_number:
                    reference_number = f"REF{datetime.now().strftime('%Y%m%d%H%M%S')}{imported_count:03d}"
                
                # Set default guardian_id if missing
                guardian_id = driver.get('guardian_id', 1)
                
                # Insert into new table
                cursor.execute('''
                    INSERT INTO drivers (driver_id, name, address, phone, email, 
                                        reference_number, license_number, guardian_id, registration_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    driver_id,
                    driver.get('name', ''),
                    driver.get('address', ''),
                    driver.get('phone', ''),
                    driver.get('email', ''),
                    reference_number,
                    driver.get('license_number', ''),
                    guardian_id,
                    driver.get('registration_date', datetime.now().isoformat())
                ))
                imported_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Skipping driver {driver.get('name', 'Unknown')}: {e}")
        
        print(f"   ✅ Imported {imported_count} drivers into new table")
    else:
        print("   ℹ️ No existing data to import")
    
    # 4. Create indexes
    print("\n4. Creating indexes...")
    try:
        cursor.execute('CREATE INDEX idx_drivers_guardian ON drivers(guardian_id)')
        print("   ✅ Created index on guardian_id")
    except Exception as e:
        print(f"   ⚠️ Error creating index: {e}")
    
    # 5. Commit changes
    conn.commit()
    
    # 6. Verify the new table
    print("\n5. Verifying new table structure...")
    cursor.execute("PRAGMA table_info(drivers)")
    columns = cursor.fetchall()
    
    print("\n📊 New drivers table structure:")
    print("-" * 50)
    for col in columns:
        col_id, name, col_type, notnull, default_value, pk = col
        constraints = []
        if pk: constraints.append("PRIMARY KEY")
        if notnull: constraints.append("NOT NULL")
        if default_value: constraints.append(f"DEFAULT {default_value}")
        
        constraint_str = " ".join(constraints) if constraints else ""
        print(f"  {name:20} {col_type:15} {constraint_str}")
    
    # Count rows
    cursor.execute("SELECT COUNT(*) FROM drivers")
    count = cursor.fetchone()[0]
    print(f"\n📈 Total drivers in new table: {count}")
    
    # Show sample
    if count > 0:
        cursor.execute("SELECT driver_id, name, phone, reference_number FROM drivers LIMIT 3")
        samples = cursor.fetchall()
        print("\n📋 Sample drivers:")
        for sample in samples:
            print(f"  ID: {sample[0]}, Name: {sample[1]}, Phone: {sample[2]}, Ref: {sample[3]}")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ Drivers table rebuilt successfully!")
    print("=" * 60)

def quick_fix_alternative():
    """Alternative: Try to add columns without UNIQUE constraint first, then make unique"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("🔧 Trying alternative fix...")
    
    # First check what columns exist
    cursor.execute("PRAGMA table_info(drivers)")
    columns = [col[1] for col in cursor.fetchall()]
    print(f"Existing columns: {columns}")
    
    # Try to add reference_number without UNIQUE first
    if 'reference_number' not in columns:
        try:
            print("Adding reference_number column without UNIQUE constraint...")
            cursor.execute('ALTER TABLE drivers ADD COLUMN reference_number TEXT')
            
            # Generate reference numbers for existing rows
            cursor.execute("SELECT rowid FROM drivers WHERE reference_number IS NULL")
            rows = cursor.fetchall()
            for i, row in enumerate(rows):
                rowid = row[0]
                ref_num = f"REF{datetime.now().strftime('%Y%m%d%H%M%S')}{i:03d}"
                cursor.execute("UPDATE drivers SET reference_number = ? WHERE rowid = ?", (ref_num, rowid))
            
            print("✅ Added reference_number column and populated it")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    # Now try to add UNIQUE constraint (SQLite doesn't support adding UNIQUE via ALTER)
    # We'll create a new table with the constraint
    try:
        print("\nCreating new table with UNIQUE constraint...")
        
        # Export data
        cursor.execute("SELECT * FROM drivers")
        data = cursor.fetchall()
        column_names = [col[1] for col in cursor.description]
        
        # Create new table
        cursor.execute('DROP TABLE IF EXISTS drivers_new')
        cursor.execute('''
            CREATE TABLE drivers_new (
                driver_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                address TEXT,
                phone TEXT NOT NULL,
                email TEXT,
                reference_number TEXT UNIQUE,
                license_number TEXT,
                guardian_id INTEGER,
                registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (guardian_id) REFERENCES guardians(guardian_id)
            )
        ''')
        
        # Re-insert data
        for row in data:
            # Convert row to dict
            row_dict = dict(zip(column_names, row))
            
            # Ensure reference_number is unique
            if not row_dict.get('reference_number'):
                import uuid
                row_dict['reference_number'] = f"REF{uuid.uuid4().hex[:8].upper()}"
            
            cursor.execute('''
                INSERT INTO drivers_new (driver_id, name, address, phone, email, 
                                       reference_number, license_number, guardian_id, registration_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row_dict.get('driver_id'),
                row_dict.get('name'),
                row_dict.get('address'),
                row_dict.get('phone'),
                row_dict.get('email'),
                row_dict.get('reference_number'),
                row_dict.get('license_number'),
                row_dict.get('guardian_id', 1),
                row_dict.get('registration_date')
            ))
        
        # Replace old table
        cursor.execute('DROP TABLE drivers')
        cursor.execute('ALTER TABLE drivers_new RENAME TO drivers')
        
        conn.commit()
        print("✅ Successfully created table with UNIQUE constraint")
        return True
        
    except Exception as e:
        print(f"❌ Error in alternative fix: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def main():
    print("🚗 Driver Alert System - Database Repair Tool")
    print("=" * 70)
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        return
    
    # Show options
    print("\nChoose repair method:")
    print("1. Complete rebuild (Recommended - fixes all issues)")
    print("2. Quick alternative fix")
    print("3. Check current table structure")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        print("\n⚠️  WARNING: This will rebuild the drivers table.")
        print("A backup will be created automatically.")
        confirm = input("Continue? (y/n): ").lower()
        
        if confirm == 'y':
            # Create backup first
            print("\n📁 Creating backup...")
            if backup_database():
                rebuild_drivers_table()
            else:
                confirm2 = input("Backup failed. Continue anyway? (y/n): ").lower()
                if confirm2 == 'y':
                    rebuild_drivers_table()
        else:
            print("❌ Operation cancelled.")
    
    elif choice == '2':
        print("\n🔄 Attempting quick fix...")
        if quick_fix_alternative():
            print("\n✅ Quick fix successful!")
        else:
            print("\n❌ Quick fix failed. Try option 1 instead.")
    
    elif choice == '3':
        print("\n🔍 Checking current table structure...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check drivers table
        try:
            cursor.execute("PRAGMA table_info(drivers)")
            columns = cursor.fetchall()
            print("\n📊 Current drivers table structure:")
            print("-" * 50)
            for col in columns:
                col_id, name, col_type, notnull, default_value, pk = col
                constraints = []
                if pk: constraints.append("PK")
                if notnull: constraints.append("NOT NULL")
                constraint_str = ", ".join(constraints)
                print(f"  {name:20} {col_type:15} {constraint_str}")
            
            # Count rows
            cursor.execute("SELECT COUNT(*) FROM drivers")
            count = cursor.fetchone()[0]
            print(f"\n📈 Total drivers: {count}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        conn.close()
    
    elif choice == '4':
        print("👋 Exiting...")
        return
    
    else:
        print("❌ Invalid choice!")
    
    print("\n🎉 Done! Restart your Flask server to apply changes.")

if __name__ == "__main__":
    main()