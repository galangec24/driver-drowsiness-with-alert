# backend/create_symlinks.py
import os

def create_symlinks():
    """Create symlinks from frontend to backend for PWA files"""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(backend_dir, '../frontend')
    
    files_to_link = ['manifest.json', 'service-worker.js']
    
    for filename in files_to_link:
        src = os.path.join(backend_dir, filename)
        dst = os.path.join(frontend_dir, filename)
        
        if os.path.exists(src):
            try:
                if os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
                print(f"✅ Created symlink: {dst} -> {src}")
            except Exception as e:
                print(f"⚠️ Could not create symlink for {filename}: {e}")
                # Copy file instead
                with open(src, 'r') as f_src:
                    with open(dst, 'w') as f_dst:
                        f_dst.write(f_src.read())
                print(f"✅ Copied {filename} to frontend directory")

if __name__ == '__main__':
    create_symlinks()