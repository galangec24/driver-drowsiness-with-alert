"""
Utility functions for drowsiness detection
"""

import numpy as np
import time

def detect_blink(ear, prev_ear, blink_thresh, detector_instance):
    """Improved blink detection with multiple blink support"""
    is_blink = False
    
    # Detect blink start (when EAR drops below threshold)
    if ear < blink_thresh and prev_ear >= blink_thresh:
        detector_instance.blink_counter += 1
        detector_instance.blink_start_frame = detector_instance.frame_count
    
    # Detect blink end (when EAR rises above threshold)
    elif ear >= blink_thresh and prev_ear < blink_thresh:
        if detector_instance.blink_counter > 0:
            blink_duration = detector_instance.frame_count - detector_instance.blink_start_frame
            if 1 <= blink_duration <= detector_instance.BLINK_FRAMES:
                is_blink = True
                detector_instance.total_blinks += 1
                detector_instance.last_blink_time = time.time()
                detector_instance.blink_history.append(detector_instance.last_blink_time)
            
            # Reset blink counter
            detector_instance.blink_counter = 0
    
    return is_blink

def calculate_blink_rate(blink_history):
    """Calculate blink rate per minute"""
    if len(blink_history) < 2:
        return 0
    
    # Calculate time span of recent blinks
    recent_blinks = list(blink_history)[-10:]
    if len(recent_blinks) < 2:
        return 0
    
    time_span = recent_blinks[-1] - recent_blinks[0]
    if time_span > 0:
        blinks_per_second = (len(recent_blinks) - 1) / time_span
        return blinks_per_second * 60  # Convert to per minute
    return 0