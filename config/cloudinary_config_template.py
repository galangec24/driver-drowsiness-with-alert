# config/cloudinary_config.py - TEMPLATE (DO NOT COMMIT REAL SECRETS!)
# Copy this to config/cloudinary_config_local.py and add your real secrets

import cloudinary
import cloudinary.uploader

class CloudinaryConfig:
    #  DO NOT COMMIT REAL VALUES TO GITHUB!
    # Add your real values in config/cloudinary_config_local.py
    
    CLOUD_NAME = "your_cloud_name_here"
    API_KEY = "your_api_key_here"
    API_SECRET = "your_api_secret_here"
    
    @classmethod
    def configure(cls):
        cloudinary.config(
            cloud_name=cls.CLOUD_NAME,
            api_key=cls.API_KEY,
            api_secret=cls.API_SECRET,
            secure=True
        )
