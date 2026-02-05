# create_app_icons.py - Run this once to create icons
from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, filename, color='#4e73df'):
    """Create app icon with truck/face symbol"""
    img = Image.new('RGB', (size, size), color=color)
    draw = ImageDraw.Draw(img)
    
    # Draw circle
    margin = size // 10
    draw.ellipse([margin, margin, size - margin, size - margin], fill='white')
    
    # Draw simple truck symbol or face
    if size >= 192:
        # For larger icons, draw more details
        draw.line([size//3, size//2, size*2//3, size//2], fill=color, width=size//20)
        draw.line([size//3, size//3, size//3, size*2//3], fill=color, width=size//20)
        draw.line([size*2//3, size//3, size*2//3, size*2//3], fill=color, width=size//20)
    
    # Save icon
    img.save(f'frontend/{filename}')
    print(f"Created {filename}")

# Create icons
create_icon(192, 'icon-192.png')
create_icon(512, 'icon-512.png')
print("✅ Icons created in frontend/ directory")