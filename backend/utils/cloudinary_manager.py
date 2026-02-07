import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
import base64
from io import BytesIO
from datetime import datetime
import requests
from PIL import Image
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).parent.parent.parent
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

class CloudinaryDriverStorage:
    """Cloudinary storage for driver face images"""
    
    def __init__(self):
        # Your credentials from Cloudinary dashboard
        self.cloud_name = "dxbyuxohr"  # Your cloud name
        self.api_key = "797159664754117"  # Your API key
        self.api_secret = os.getenv('CLOUDINARY_API_SECRET')  # From .env
        
        if not self.api_secret:
            raise ValueError("CLOUDINARY_API_SECRET not found in .env file")
        
        # Configure Cloudinary
        cloudinary.config(
            cloud_name=self.cloud_name,
            api_key=self.api_key,
            api_secret=self.api_secret,
            secure=True
        )
        
        print(f"✅ Cloudinary Driver Storage initialized")
        print(f"   Cloud: {self.cloud_name}")
        print(f"   API Key: {self.api_key[:10]}...")
        print(f"   Free credits: 25/month")
    
    def upload_driver_face(self, base64_image: str, driver_id: str, image_number: int) -> str:
        """
        Upload driver face image to Cloudinary
        
        Returns: Public URL of the uploaded image
        """
        try:
            print(f"📤 Uploading face {image_number} for driver {driver_id}...")
            
            # Decode base64
            if base64_image.startswith('data:image'):
                base64_image = base64_image.split(',')[1]
            
            image_bytes = base64.b64decode(base64_image)
            
            # Create public ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            public_id = f"driver_faces/{driver_id}/face_{image_number}"
            
            # Upload to Cloudinary
            result = cloudinary.uploader.upload(
                image_bytes,
                public_id=public_id,
                folder="driver_faces",
                overwrite=True,
                resource_type="image",
                transformation=[
                    {'width': 400, 'height': 400, 'crop': 'fill'},
                    {'quality': 'auto:good'}
                ]
            )
            
            url = result['secure_url']
            print(f"✅ Uploaded to Cloudinary: {public_id}")
            print(f"   URL: {url}")
            
            return url
            
        except Exception as e:
            print(f"❌ Cloudinary upload failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_driver_face_url(self, driver_id: str, image_number: int = 1) -> str:
        """Get URL for a driver's face image"""
        # Cloudinary URL pattern
        return f"https://res.cloudinary.com/{self.cloud_name}/image/upload/driver_faces/{driver_id}/face_{image_number}.jpg"
    
    def test_connection(self):
        """Test Cloudinary connection"""
        try:
            # Simple test: get account info
            account_info = cloudinary.api.ping()
            print(f"✅ Cloudinary connection successful!")
            print(f"   Status: {account_info.get('status')}")
            return True
        except Exception as e:
            print(f"❌ Cloudinary connection failed: {e}")
            return False

# Singleton instance
_cloudinary_storage = None

def get_cloudinary_storage():
    """Get or create Cloudinary storage instance"""
    global _cloudinary_storage
    if _cloudinary_storage is None:
        _cloudinary_storage = CloudinaryDriverStorage()
    return _cloudinary_storage