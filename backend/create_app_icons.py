# backend/create_app_icons.py
"""
PWA Icon Generator for Driver Drowsiness Alert System
Run this script to create mobile app icons
"""

from PIL import Image, ImageDraw, ImageFont
import os
import sys

def create_icon(size, filename, color='#4e73df'):
    """
    Create a PWA icon with driver/car symbol
    
    Args:
        size (int): Icon size in pixels (192 or 512)
        filename (str): Output filename
        color (str): Hex color code for the icon
    """
    print(f"🖌️  Creating {size}x{size} icon...")
    
    # Create new image with background color
    img = Image.new('RGB', (size, size), color=color)
    draw = ImageDraw.Draw(img)
    
    # Draw white circle in center
    margin = size // 8
    circle_radius = size - (2 * margin)
    draw.ellipse([margin, margin, size - margin, size - margin], 
                 fill='white', outline=color, width=size//40)
    
    # Draw driver/car symbol (more detailed for larger icons)
    if size >= 192:
        # Car body dimensions
        body_top = size // 3
        body_bottom = size * 2 // 3
        body_left = size // 4
        body_right = size * 3 // 4
        
        # Main car body
        draw.rectangle([body_left, body_top, body_right, body_bottom], 
                      fill=color, outline='white', width=size//50)
        
        # Car window
        window_top = body_top + (body_bottom - body_top) // 4
        window_bottom = body_top + (body_bottom - body_top) // 2
        window_left = body_left + (body_right - body_left) // 4
        window_right = body_right - (body_right - body_left) // 4
        draw.rectangle([window_left, window_top, window_right, window_bottom], 
                      fill='#e9ecef', outline='white', width=size//100)
        
        # Wheels
        wheel_size = size // 8
        wheel_y = body_bottom - wheel_size//4
        
        # Left wheel
        draw.ellipse([body_left - wheel_size//2, wheel_y - wheel_size//2,
                     body_left + wheel_size//2, wheel_y + wheel_size//2], 
                    fill='white', outline=color, width=size//80)
        
        # Right wheel
        draw.ellipse([body_right - wheel_size//2, wheel_y - wheel_size//2,
                     body_right + wheel_size//2, wheel_y + wheel_size//2], 
                    fill='white', outline=color, width=size//80)
        
        # Add a simple driver silhouette in window (for larger icons)
        if size >= 300:
            # Driver head
            head_center_x = (window_left + window_right) // 2
            head_center_y = window_top + (window_bottom - window_top) // 3
            head_radius = (window_right - window_left) // 6
            draw.ellipse([head_center_x - head_radius, head_center_y - head_radius,
                         head_center_x + head_radius, head_center_y + head_radius], 
                        fill=color)
    
    # Save to frontend directory
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    os.makedirs(frontend_dir, exist_ok=True)
    
    output_path = os.path.join(frontend_dir, filename)
    img.save(output_path, 'PNG', optimize=True)
    
    print(f"   ✅ Saved: {output_path}")
    return output_path

def create_screenshot_placeholder():
    """Create a placeholder screenshot for PWA manifest"""
    print("📱 Creating screenshot placeholder...")
    
    width, height = 1080, 1920
    img = Image.new('RGB', (width, height), color='#f5f7fa')
    draw = ImageDraw.Draw(img)
    
    # Header with gradient effect
    for i in range(200):
        alpha = i / 200
        color = (78, 115, 223)  # #4e73df in RGB
        draw.line([0, i, width, i], fill=color)
    
    # App title
    try:
        # Try to load a font
        font_paths = [
            'C:/Windows/Fonts/arial.ttf',
            '/System/Library/Fonts/Helvetica.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, 80)
                    break
                except:
                    continue
        
        if font:
            draw.text((width//2, 100), "DRIVER ALERT", 
                     fill='white', font=font, anchor='mm')
            draw.text((width//2, 200), "Drowsiness Monitoring System", 
                     fill='rgba(255,255,255,0.8)', font=ImageFont.truetype(font_path, 40), anchor='mm')
        else:
            # Fallback to default font
            draw.text((width//2, 100), "DRIVER ALERT", 
                     fill='white', anchor='mm')
    except:
        # Simple text if font loading fails
        draw.text((width//2, 100), "DRIVER ALERT", 
                 fill='white', anchor='mm')
    
    # Mock app interface
    # Card 1: Dashboard
    card1_left, card1_top = 50, 300
    card1_right, card1_bottom = width - 50, 500
    draw.rounded_rectangle([card1_left, card1_top, card1_right, card1_bottom], 
                          radius=20, fill='white', outline='#dee2e6', width=3)
    
    # Card 2: Driver List
    card2_top, card2_bottom = 550, 900
    draw.rounded_rectangle([card1_left, card2_top, card1_right, card2_bottom], 
                          radius=20, fill='white', outline='#dee2e6', width=3)
    
    # Card 3: Alerts
    card3_top, card3_bottom = 950, 1300
    draw.rounded_rectangle([card1_left, card3_top, card1_right, card3_bottom], 
                          radius=20, fill='white', outline='#dee2e6', width=3)
    
    # Footer
    draw.rectangle([0, height-100, width, height], fill='#4e73df')
    
    # Save screenshot
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    screenshot_path = os.path.join(frontend_dir, 'screenshot-mobile.png')
    img.save(screenshot_path, 'PNG', optimize=True)
    
    print(f"   ✅ Saved: {screenshot_path}")
    return screenshot_path

def create_favicon():
    """Create favicon.ico for browser tabs"""
    print("🔗 Creating favicon...")
    
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]
    images = []
    
    for size in sizes:
        img = Image.new('RGB', size, '#4e73df')
        draw = ImageDraw.Draw(img)
        
        # Simple circle for small icons
        margin = size[0] // 4
        draw.ellipse([margin, margin, size[0]-margin, size[1]-margin], 
                    fill='white')
        
        images.append(img)
    
    # Save as ICO
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    favicon_path = os.path.join(frontend_dir, 'favicon.ico')
    
    # For ICO format, we need to save the first image
    images[0].save(favicon_path, format='ICO', sizes=[(16, 16), (32, 32)])
    
    print(f"   ✅ Saved: {favicon_path}")
    return favicon_path

def verify_files():
    """Verify all required files exist"""
    print("\n🔍 Verifying generated files...")
    
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    required_files = ['icon-192.png', 'icon-512.png']
    
    all_ok = True
    for filename in required_files:
        filepath = os.path.join(frontend_dir, filename)
        if os.path.exists(filepath):
            print(f"   ✅ {filename}: Found")
            # Check file size
            size = os.path.getsize(filepath)
            print(f"       Size: {size:,} bytes")
        else:
            print(f"   ❌ {filename}: Missing!")
            all_ok = False
    
    return all_ok

def main():
    """Main function to generate all PWA assets"""
    print("=" * 60)
    print("🚗 DRIVER DROWSINESS ALERT SYSTEM - PWA ASSET GENERATOR")
    print("=" * 60)
    
    try:
        # Generate main icons
        icon_192 = create_icon(192, 'icon-192.png')
        icon_512 = create_icon(512, 'icon-512.png')
        
        # Generate additional assets (optional)
        create_screenshot_placeholder()
        create_favicon()
        
        # Verify files
        if verify_files():
            print("\n" + "=" * 60)
            print("🎉 SUCCESS! All PWA assets generated successfully!")
            print("=" * 60)
            
            # Show next steps
            frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
            print(f"\n📁 Assets location: {os.path.abspath(frontend_dir)}")
            print("\n📱 NEXT STEPS:")
            print("1. ✅ Icons are ready for PWA")
            print("2. Ensure manifest.json and service-worker.js are in frontend/")
            print("3. Start your Flask server:")
            print("   cd backend")
            print("   python app.py")
            print("4. Access on your computer: http://localhost:5000")
            print("5. On mobile (same WiFi): http://[YOUR-IP]:5000")
            print("6. Tap 'Add to Home Screen' in browser menu")
            print("\n💡 Tip: Check frontend/manifest.json references these icons")
            
        else:
            print("\n⚠️  Some files may be missing. Please check the output above.")
            
    except ImportError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Solution: Install Pillow library:")
        print("   pip install Pillow")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()