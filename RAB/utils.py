# utils.py

import exifread
from datetime import datetime
import logging
from PIL import Image
import rawpy
import numpy as np
import imageio
import subprocess

def allowed_file(filename):
    allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'dng', 'heic', 'mp4', 'mov', 'avi', 'wmv', 'flv', 'mkv', 'hevc'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_metadata(filepath):
    metadata = {
        'timestamp': None,
        'latitude': None,
        'longitude': None
    }
    try:
        with open(filepath, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        # Extract timestamp
        timestamp = tags.get('EXIF DateTimeOriginal')
        if timestamp:
            metadata['timestamp'] = datetime.strptime(str(timestamp), '%Y:%m:%d %H:%M:%S')
        # Extract GPS data
        gps_latitude = tags.get('GPS GPSLatitude')
        gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
        gps_longitude = tags.get('GPS GPSLongitude')
        gps_longitude_ref = tags.get('GPS GPSLongitudeRef')
        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = _convert_to_degrees(gps_latitude)
            if gps_latitude_ref.values[0] != 'N':
                lat = -lat
            lon = _convert_to_degrees(gps_longitude)
            if gps_longitude_ref.values[0] != 'E':
                lon = -lon
            metadata['latitude'] = lat
            metadata['longitude'] = lon
    except Exception as e:
        logging.error(f'Error extracting metadata from {filepath}: {str(e)}')
    return metadata

def _convert_to_degrees(value):
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)
    return d + (m / 60.0) + (s / 3600.0)

import os
import logging
from subprocess import run, PIPE
import rawpy
from PIL import Image


import rawpy
import numpy as np
from PIL import Image, ImageCms
import logging
import os
import rawpy
import numpy as np
from PIL import Image, ImageCms
import logging
import os


import rawpy
import numpy as np
from PIL import Image, ImageCms
import logging
import os
import tempfile

def generate_png_from_dng(dng_filepath, png_filepath):
    """
    Convert DNG files to PNG with robust error handling and verification
    
    Args:
        dng_filepath (str): Path to source DNG file
        png_filepath (str): Path where PNG should be saved
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    logger = logging.getLogger('dng_converter')
    
    if not os.path.exists(dng_filepath):
        logger.error(f"Source DNG file not found: {dng_filepath}")
        return False
        
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(png_filepath), exist_ok=True)
    
    # Use a temporary file to ensure atomic write
    temp_png = None
    try:
        # Create a temporary file in the same directory as the target
        temp_dir = os.path.dirname(png_filepath)
        temp_fd, temp_png = tempfile.mkstemp(suffix='.png', dir=temp_dir)
        os.close(temp_fd)  # Close the file descriptor
        
        logger.info(f"Starting DNG conversion: {dng_filepath}")
        
        # Read and process the DNG file
        with rawpy.imread(dng_filepath) as raw:
            logger.debug("Successfully opened DNG with rawpy")
            
            # Process with conservative parameters
            rgb_image = raw.postprocess(
                use_camera_wb=True,      # Use camera white balance
                half_size=False,         # Full resolution
                no_auto_bright=True,     # Disable auto brightness
                output_bps=8,            # 8 bits per sample
                bright=1.0,              # Normal brightness
                user_flip=0,             # No rotation
                gamma=(2.222, 4.5),      # Standard gamma curve
                highlight_mode=0,        # Clip highlights
                output_color=rawpy.ColorSpace.sRGB  # Use sRGB color space
            )
            logger.debug("Successfully processed DNG data")
        
        # Convert to PIL Image
        image = Image.fromarray(rgb_image)
        
        # Try to apply color profile
        try:
            # Check for Display P3 profile (Mac)
            p3_profile_path = os.getenv('COLOR_PROFILE_PATH')
            if os.path.exists(p3_profile_path):
                try:
                    p3_profile = ImageCms.ImageCmsProfile(p3_profile_path)
                    srgb_profile = ImageCms.createProfile('sRGB')
                    
                    # Convert from P3 to sRGB for wider compatibility
                    image = ImageCms.profileToProfile(
                        image,
                        p3_profile,
                        srgb_profile,
                        renderingIntent=ImageCms.Intent.PERCEPTUAL
                    )
                    logger.debug("Successfully applied color profile conversion")
                except Exception as color_error:
                    logger.warning(f"Color profile conversion failed, using default: {str(color_error)}")
        except Exception as profile_error:
            logger.warning(f"Color profile handling failed: {str(profile_error)}")
        
        # Save to temporary file first
        image.save(
            temp_png,
            "PNG",
            optimize=True,
            quality=95
        )
        
        # Verify the temporary file
        if not os.path.exists(temp_png):
            raise FileNotFoundError(f"Failed to create temporary PNG file")
            
        temp_size = os.path.getsize(temp_png)
        if temp_size == 0:
            raise ValueError(f"Created PNG file is empty")
            
        # Try opening the saved file to verify it's valid
        try:
            with Image.open(temp_png) as verify_img:
                verify_img.verify()
        except Exception as verify_error:
            raise ValueError(f"Created PNG file is invalid: {str(verify_error)}")
        
        # If all verifications pass, move the temporary file to the final location
        os.replace(temp_png, png_filepath)
        logger.info(f"Successfully created PNG file: {png_filepath} ({temp_size:,} bytes)")
        
        return True
        
    except rawpy.LibRawError as e:
        logger.error(f"LibRaw error processing DNG: {str(e)}")
        return False
        
    except (IOError, OSError) as e:
        logger.error(f"IO/OS error during DNG conversion: {str(e)}")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error during DNG conversion: {str(e)}")
        return False
        
    finally:
        # Clean up temporary file if it exists
        if temp_png and os.path.exists(temp_png):
            try:
                os.remove(temp_png)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file: {str(cleanup_error)}")


def extract_gpx_points(gpx):
    gpx_points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                gpx_points.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'time': point.time.replace(tzinfo=None) if point.time else None
                })
    return gpx_points



