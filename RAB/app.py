from flask import Flask, render_template, url_for, flash, redirect, request, jsonify, session, current_app, abort
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from datetime import datetime
import os
import logging
from .forms import GPXUploadForm, EmailForm, UploadForm, CommentForm, EditMetadataForm
from .extensions import db, login_manager
from flask_login import login_user, current_user, logout_user, login_required
from werkzeug.utils import secure_filename
from flask_migrate import Migrate
from PIL import Image
import rawpy
from .utils import allowed_file, extract_metadata, generate_png_from_dng, extract_gpx_points
import numpy as np
from .config import Config
import gpxpy
from collections import Counter, defaultdict
from geopy.distance import geodesic
from sqlalchemy import and_
from .models import Photo, Favorite, User, Comment
from flask_wtf.csrf import CSRFProtect
from flask import Flask
from flask_compress import Compress
from flask import send_file
from flask import request, jsonify, render_template
from datetime import datetime
from werkzeug.exceptions import BadRequest
from flask import flash, redirect, url_for
from concurrent.futures import ThreadPoolExecutor
from werkzeug.utils import secure_filename
import os
import logging
from PIL import Image
import io
from flask import current_app
import requests
from PIL import Image, ImageOps
import subprocess
from dotenv import load_dotenv
import shutil
import os
import mimetypes
from flask import request, send_file, make_response
from werkzeug.exceptions import RequestedRangeNotSatisfiable
import re

import os
import re
from flask import Response
from flask import request, Response, current_app
from functools import partial
from werkzeug.exceptions import ClientDisconnected

# Configure logging
logging.basicConfig(level=logging.DEBUG)
load_dotenv()  # Load environment variables from .env

def send_file_partial(path, start=0, end=None):
    """Send a file in chunks with proper range handling."""
    file_size = os.path.getsize(path)
    
    # If no end specified, send until the end of file
    if end is None:
        end = file_size - 1
    
    # Ensure end isn't beyond file size
    end = min(end, file_size - 1)
    
    # Calculate content length
    chunk_size = end - start + 1

    # Open the file and seek to start
    with open(path, 'rb') as f:
        f.seek(start)
        data = f.read(chunk_size)
        
        return data


def get_chunk_size(file_size, is_initial_request=False):
    """
    Calculate optimal chunk size based on file size and request type
    """
    if is_initial_request:
        return 1024 * 1024  # 1MB for initial load
    elif file_size > 1024 * 1024 * 1024:  # > 1GB
        return 5 * 1024 * 1024  # 5MB chunks
    else:
        return 2 * 1024 * 1024  # 2MB chunks


def get_mimetype(filename):
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        if filename.lower().endswith('.mov'):
            return 'video/quicktime'
        # Add other file type checks if necessary
        return 'application/octet-stream'
    return mime_type



def is_admin(user):
    """Check if the user has admin privileges"""
    admin_email = os.getenv('ADMIN_EMAIL')
    return user.email.lower() == admin_email.lower()

def can_modify_photo(user, photo):
    """Check if user can modify (edit/delete) a photo"""
    return is_admin(user) or photo.uploader == user


def get_photo_url(photo):
    """
    Get the appropriate URL for a photo, handling DNG, HEIC, and regular image files.
    
    Priority order for DNG:
    1. PNG converted version
    2. Preview JPG
    3. Original file
    
    Priority order for HEIC:
    1. JPG converted version
    2. Preview JPG
    3. Original file
    
    Priority order for other files:
    1. Preview version
    2. Original file
    """
    # Get lowercase filename for consistent extension checking
    filename_lower = photo.filename.lower()
    
    # Handle DNG files
    if filename_lower.endswith('.dng'):
        # First preference: PNG version
        if photo.png_filename:
            return url_for('static', filename=f'uploads/{photo.png_filename}')
        # Second preference: Preview JPG
        elif photo.preview_filename:
            preview_filename = photo.preview_filename.replace('.DNG', '.JPG').replace('.dng', '.jpg')
            return url_for('serve_preview', filename=preview_filename)
    
    # Handle HEIC files
    elif filename_lower.endswith(('.heic', '.heif')):
        # First preference: Converted JPG version
        if photo.converted_filename:
            return url_for('static', filename=f'uploads/{photo.converted_filename}')
        # Second preference: Preview JPG
        elif photo.preview_filename:
            preview_filename = photo.preview_filename.replace('.HEIC', '.JPG').replace('.heic', '.jpg')
            return url_for('serve_preview', filename=preview_filename)
    
    # Handle all other image files
    else:
        # For non-DNG/HEIC files, prefer preview if available
        if photo.preview_filename:
            return url_for('serve_preview', filename=photo.preview_filename)
    
    # Fall back to original file if no other version is available
    return url_for('static', filename=f'uploads/{photo.filename}')


def generate_video_preview(video_path, preview_path):
    """
    Generate a preview image from the first frame of a video using ffmpeg.
    Returns the preview filename if successful, None if failed.
    """
    logging.info(f"Attempting to generate video preview from: {video_path}")
    
    try:
        # Ensure the preview directory exists
        os.makedirs(os.path.dirname(preview_path), exist_ok=True)
        
        # First try to extract frame at 1 second
        command = [
            'ffmpeg',
            '-i', video_path,
            '-ss', '00:00:01.000',
            '-vframes', '1',
            '-vf', 'scale=800:-1',
            '-f', 'image2',
            '-y',  # Overwrite output file if exists
            preview_path
        ]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False  # Don't raise on error, we'll handle it
        )
        
        # If first attempt fails, try frame 0
        if result.returncode != 0:
            logging.warning("Failed to extract frame at 1 second, trying first frame")
            command[3] = '00:00:00.000'
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            logging.error(f"FFmpeg error: {result.stderr}")
            return None
        
        # Verify the preview file
        if os.path.exists(preview_path):
            if os.path.getsize(preview_path) > 0:
                try:
                    with Image.open(preview_path) as img:
                        img.verify()
                    logging.info("Successfully generated and verified video preview")
                    return os.path.basename(preview_path)
                except Exception as e:
                    logging.error(f"Preview verification failed: {str(e)}")
                    if os.path.exists(preview_path):
                        os.remove(preview_path)
            else:
                logging.error("Generated preview file is empty")
                os.remove(preview_path)
        
        return None
        
    except Exception as e:
        logging.error(f"Error in generate_video_preview: {str(e)}")
        if os.path.exists(preview_path):
            os.remove(preview_path)
        return None


class UploadHandler:
    def __init__(self):
        self.preview_size = (800, 800)
        self.logger = logging.getLogger('UploadHandler')
        self.allowed_video_extensions = {'.mp4', '.mov', '.avi', '.wmv', '.flv', '.mkv'}
        self.allowed_image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.heic', '.dng'}
    
    def process_upload(self, file, upload_folder, preview_folder, filename=None, file_already_saved=False):
        """
        Process file upload with improved video and error handling.
        
        Args:
            file: File object (FileStorage or file-like object)
            upload_folder: Path to save uploaded files
            preview_folder: Path to save preview images
            filename: Optional filename override
            file_already_saved: Whether the file is already saved to upload_folder
        """
        try:
            # Get/validate filename
            if not filename and hasattr(file, 'filename'):
                filename = file.filename
            elif not filename:
                raise ValueError("Filename must be provided if file object lacks filename attribute")
            
            filename = secure_filename(filename)
            filepath = os.path.join(upload_folder, filename)
            
            self.logger.info(f"Processing upload: {filename}")
            
            # Ensure directories exist
            os.makedirs(upload_folder, exist_ok=True)
            os.makedirs(preview_folder, exist_ok=True)
            
            # Check for existing file
            if os.path.exists(filepath) and not file_already_saved:
                return {
                    'success': True,
                    'existing': True,
                    'filename': filename,
                    'message': f"File '{filename}' already exists"
                }
            
            # Save file if needed
            if not file_already_saved:
                try:
                    if hasattr(file, 'save'):
                        file.save(filepath)
                    else:
                        with open(filepath, 'wb') as out_file:
                            shutil.copyfileobj(file, out_file)
                except Exception as e:
                    self.logger.error(f"Error saving file: {str(e)}")
                    return {'success': False, 'error': f"File save error: {str(e)}"}
            
            # Verify file
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                return {'success': False, 'error': 'File save verification failed'}
            
            # Determine file type
            ext = os.path.splitext(filename.lower())[1]
            is_video = ext in self.allowed_video_extensions
            is_image = ext in self.allowed_image_extensions
            
            preview_filename = None
            if is_video:
                self.logger.info("Processing video file")
                preview_path = os.path.join(preview_folder, f'preview_{os.path.splitext(filename)[0]}.jpg')
                preview_filename = generate_video_preview(filepath, preview_path)
                if not preview_filename:
                    self.logger.warning("Failed to generate video preview, continuing without preview")
            
            elif is_image:
                self.logger.info("Processing image file")
                preview_path = os.path.join(preview_folder, f'preview_{os.path.splitext(filename)[0]}.jpg')
                try:
                    with Image.open(filepath) as img:
                        img = ImageOps.exif_transpose(img)
                        img.thumbnail(self.preview_size, Image.LANCZOS)
                        img.save(preview_path, "JPEG", quality=85, optimize=True)
                    preview_filename = os.path.basename(preview_path)
                except Exception as e:
                    self.logger.error(f"Error generating image preview: {str(e)}")
                    preview_filename = None
            
            result = {
                'success': True,
                'filename': filename,
                'filepath': filepath,
                'preview_filename': preview_filename,
                'is_video': is_video,
                'is_image': is_image
            }
            
            self.logger.info(f"Upload processed successfully: {filename}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing upload: {str(e)}")
            return {'success': False, 'error': str(e)}

# Initialize the upload handler
upload_handler = UploadHandler()

def setup_upload_route(app):
    """Configure the upload route with the optimized handler"""
    upload_handler = UploadHandler()

def generate_preview(image_path, preview_path):
    try:
        # Log paths before attempting to create the directory
        logging.info(f"Attempting to create directory for preview: {os.path.dirname(preview_path)}")
        
        # Ensure the directory for the preview exists
        os.makedirs(os.path.dirname(preview_path), exist_ok=True)
        logging.info(f"Directory created or already exists: {os.path.dirname(preview_path)}")

        # Open the original image
        with Image.open(image_path) as img:
            # Calculate the new dimensions at 50% of the original size
            new_width = int(img.width * 0.5)
            new_height = int(img.height * 0.5)

            # Resize the image while maintaining the aspect ratio
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)

            # Save the resized preview image
            logging.info(f"Saving preview image at: {preview_path}")
            img_resized.save(preview_path, "JPEG", quality=85, optimize=True)
            logging.info(f"Preview generated at {preview_path}, size: {new_width}x{new_height}")
            
            return new_width, new_height
    except Exception as e:
        logging.error(f"Error generating preview for {image_path}: {str(e)}")
        raise


def cleanup_orphaned_files():
    upload_folder = current_app.config['UPLOAD_FOLDER']
    files_in_folder = set(os.listdir(upload_folder))
    files_in_db = set(photo.filename for photo in Photo.query.all())
    orphaned_files = files_in_folder - files_in_db

    for filename in orphaned_files:
        file_path = os.path.join(upload_folder, filename)
        try:
            os.remove(file_path)
            logging.info(f"Removed orphaned file: {file_path}")
        except Exception as e:
            logging.error(f"Error removing orphaned file {file_path}: {str(e)}")

    return list(orphaned_files)

def cleanup_broken_images():
    photos = Photo.query.all()
    removed_photos = []
    problematic_photos = []

    for photo in photos:
        file_exists = os.path.exists(photo.filepath)
        png_exists = False
        png_filepath = None
        if photo.png_filename:
            png_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], photo.png_filename)
            png_exists = os.path.exists(png_filepath)
        
        if not file_exists and not png_exists:
            try:
                db.session.delete(photo)
                removed_photos.append(photo.filename)
            except Exception as e:
                problematic_photos.append((photo.filename, f"Error marking for deletion: {str(e)}"))
        else:
            if file_exists:
                try:
                    with open(photo.filepath, 'rb') as f:
                        f.read(1)
                except IOError as e:
                    problematic_photos.append((photo.filename, f"Original file not readable: {str(e)}"))
            
            if png_exists:
                try:
                    with open(png_filepath, 'rb') as f:
                        f.read(1)
                except IOError as e:
                    problematic_photos.append((photo.png_filename, f"PNG file not readable: {str(e)}"))

    try:
        db.session.commit()
        result = f'Removed {len(removed_photos)} broken image entries from the database.'
        if problematic_photos:
            result += f' Found {len(problematic_photos)} problematic photos that may not display correctly.'
        return result
    except Exception as e:
        db.session.rollback()
        return f'Error during cleanup: {str(e)}'
    
def calculate_storage(photos):
    """Calculates the total storage used by the user's photos in megabytes (MB)."""
    total_storage = 0
    for photo in photos:
        try:
            total_storage += os.path.getsize(photo.filepath)
        except OSError as e:
            logging.error(f"Error getting file size for {photo.filepath}: {str(e)}")
    total_storage = total_storage / (1024 * 1024)  # Convert bytes to MB
    return total_storage

def tag_photos_with_gpx_points(photos, gpx_points, max_time_diff=300):
    if not gpx_points:
        logging.warning('No GPX points provided.')
        return

    # Sort GPX points by time
    gpx_points.sort(key=lambda x: x['time'])

    # Match each photo to the closest GPX point in time
    for photo in photos:
        if photo.timestamp:
            closest_point = min(gpx_points, key=lambda x: abs(x['time'] - photo.timestamp))
            time_diff = abs((closest_point['time'] - photo.timestamp).total_seconds())
            if time_diff <= max_time_diff:
                photo.latitude = closest_point['latitude']
                photo.longitude = closest_point['longitude']
                logging.info(f'Photo {photo.filename} tagged with location from GPX.')
            else:
                logging.warning(f'No GPX data close enough in time for photo {photo.filename}.')
        else:
            logging.warning(f'Photo {photo.filename} has no timestamp.')

def ensure_directory_exists(directory_path):
    """Ensures that the given directory exists, creates it if it doesn't."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f'Created directory: {directory_path}')


def calculate_photo_statistics(photos):
    # For the most frequent time of day
    time_counter = Counter()
    day_counter = Counter()
    
    for photo in photos:
        if photo.timestamp:
            # Extract hour and day from the photo's timestamp
            hour = photo.timestamp.hour
            day_of_week = photo.timestamp.strftime('%A')  # Monday, Tuesday, etc.
            time_counter[hour] += 1
            day_counter[day_of_week] += 1
    
    # Find the most frequent hour and day
    most_common_hour = time_counter.most_common(1)[0] if time_counter else (None, 0)
    most_common_day = day_counter.most_common(1)[0] if day_counter else (None, 0)

    # For location grouping within 4-block radius (~ 400 meters)
    location_clusters = []
    for photo in photos:
        if photo.latitude and photo.longitude:
            added_to_cluster = False
            for cluster in location_clusters:
                # Compare distance to the first point in each cluster
                if geodesic((photo.latitude, photo.longitude), (cluster[0]['latitude'], cluster[0]['longitude'])).meters < 400:
                    # If within 4 blocks (approx 400 meters), add to the cluster
                    cluster.append({'latitude': photo.latitude, 'longitude': photo.longitude, 'photo': photo})
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                location_clusters.append([{'latitude': photo.latitude, 'longitude': photo.longitude, 'photo': photo}])

    # Average the locations within each cluster
    popular_locations = []
    for cluster in location_clusters:
        avg_lat = sum([loc['latitude'] for loc in cluster]) / len(cluster)
        avg_lon = sum([loc['longitude'] for loc in cluster]) / len(cluster)
        popular_locations.append({
            'latitude': avg_lat,
            'longitude': avg_lon,
            'count': len(cluster),
            'photos': cluster
        })

    # Sort popular locations by most photos in each area
    popular_locations = sorted(popular_locations, key=lambda loc: loc['count'], reverse=True)

    return most_common_hour, most_common_day, popular_locations

# Initialize Flask extensions at the module level
mail = Mail()
migrate = Migrate()
s = None  # URLSafeTimedSerializer instance will be initialized in create_app()


csrf = CSRFProtect()

compress = Compress()

def create_app():
    app = Flask(__name__, static_folder=os.getenv('STATIC_FOLDER'))
    
    # Basic configurations
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
    app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 465))
    app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True'
    app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL') == 'True'
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')
    app.config['PREVIEW_FOLDER'] = os.getenv('PREVIEW_FOLDER')
    app.config['CHUNK_FOLDER'] = os.getenv('CHUNK_FOLDER')
    
    # Parse APPROVED_EMAILS into a list
    approved_emails_str = os.getenv('APPROVED_EMAILS', '')
    app.config['APPROVED_EMAILS'] = [email.strip() for email in approved_emails_str.split(',') if email.strip()]
    
    # Parse ALLOWED_EXTENSIONS into a set
    allowed_extensions_str = os.getenv('ALLOWED_EXTENSIONS', '')
    app.config['ALLOWED_EXTENSIONS'] = set(ext.strip() for ext in allowed_extensions_str.split(',') if ext.strip())
    
    # Other configurations
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16777216000))
    app.config['COLOR_PROFILE_PATH'] = os.getenv('COLOR_PROFILE_PATH')
    app.config['ADMIN_EMAIL'] = os.getenv('ADMIN_EMAIL')
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    csrf.init_app(app)
    compress.init_app(app)

    @app.template_filter('get_mimetype')
    def get_mimetype(filename):
        """Template filter to get MIME type from filename"""
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type is None:
            # Handle common video types that mimetypes might not detect
            if filename.lower().endswith('.mov'):
                return 'video/quicktime'
            elif filename.lower().endswith('.mkv'):
                return 'video/x-matroska'
            # Default fallback
            return 'video/mp4'
        return mime_type


    global s
    s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

    # Import models within app context
    with app.app_context():
        from .models import User, Photo, Comment, Favorite

    csrf.init_app(app)
    @app.route('/', methods=['GET'])
    @login_required
    def index():
        # Get filter parameters from request args
        show_favorites = request.args.get('favorites', 'false').lower() == 'true'
        uploader_id = request.args.get('uploader', type=int)
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')

        # Start building the query
        query = Photo.query

        # Apply filters
        if show_favorites:
            # Join with Favorite table to filter favorites of the current user
            query = query.join(Favorite).filter(Favorite.user_id == current_user.id)
        
        if uploader_id:
            query = query.filter(Photo.uploader_id == uploader_id)
        
        if date_from:
            try:
                date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
                query = query.filter(Photo.timestamp >= date_from_obj)
            except ValueError:
                flash('Invalid start date format. Use YYYY-MM-DD.', 'danger')
        
        if date_to:
            try:
                date_to_obj = datetime.strptime(date_to, '%Y-%m-%d')
                query = query.filter(Photo.timestamp <= date_to_obj)
            except ValueError:
                flash('Invalid end date format. Use YYYY-MM-DD.', 'danger')

        # Order by upload time descending
        photos = query.order_by(Photo.upload_time.desc()).all()

        # Get list of users for uploader filter
        users = User.query.all()

        return render_template('index.html', photos=photos, users=users, show_favorites=show_favorites, uploader_id=uploader_id, date_from=date_from, date_to=date_to)


    @app.route('/login', methods=['GET', 'POST'])
    @csrf.exempt
    def login():
        form = EmailForm()
        if form.validate_on_submit():
            email = form.email.data.strip()
            approved_emails = current_app.config.get('APPROVED_EMAILS', [])
            
            if not approved_emails:
                flash('No approved emails configured. Please contact administrator.', 'danger')
                current_app.logger.error('APPROVED_EMAILS configuration is empty')
                return render_template('login.html', form=form)
                
            if email in approved_emails:
                token = s.dumps(email, salt='email-confirm')
                link = url_for('confirm_email', token=token, _external=True)
                
                try:
                    msg = Message('Your Magic Login Link', 
                                sender=current_app.config['MAIL_USERNAME'],
                                recipients=[email])
                    msg.body = f'Click the link to log in: {link}'
                    mail.send(msg)
                    flash('A magic link has been sent to your email.', 'info')
                except Exception as e:
                    current_app.logger.error(f'Error sending email: {str(e)}')
                    flash('Error sending login email. Please try again later.', 'danger')
                    
                return redirect(url_for('login'))
            else:
                flash('Email not approved for access.', 'danger')
                current_app.logger.warning(f'Unauthorized login attempt from email: {email}')
                
        return render_template('login.html', form=form)

    @app.route('/confirm_email/<token>')
    @csrf.exempt
    def confirm_email(token):
        try:
            email = s.loads(token, salt='email-confirm', max_age=3600)
        except Exception as e:
            flash('The link is invalid or has expired.', 'danger')
            return redirect(url_for('login'))
        user = User.query.filter_by(email=email).first()
        if user is None:
            user = User(email=email)
            db.session.add(user)
            db.session.commit()
        login_user(user)
        flash('You have been logged in!', 'success')
        return redirect(url_for('index'))

    @app.route('/logout')
    @login_required
    @csrf.exempt
    def logout():
        logout_user()
        flash('You have been logged out.', 'info')
        return redirect(url_for('login'))

    @app.route('/upload', methods=['GET', 'POST'])
    @login_required
    @csrf.exempt
    def upload():
        logger = logging.getLogger('upload_route')
        logger.setLevel(logging.DEBUG)
        logger.info(f"Upload route accessed - Method: {request.method}")

        if request.method == 'GET':
            form = UploadForm()
            return render_template('upload.html', form=form)

        # POST: process single file upload
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        if not file or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file'}), 400

        # Ensure directories for uploads
        upload_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)

        filename = secure_filename(file.filename)
        existing_file_path = os.path.join(upload_folder, filename)

        # Check if file exists, handle .heic conversion cases
        if os.path.exists(existing_file_path):
            return jsonify({
                'success': True,
                'existing': True,
                'message': f"File '{filename}' already exists",
                'filename': filename
            })

        if filename.lower().endswith('.heic'):
            converted_filename = f"{os.path.splitext(filename)[0]}.jpg"
            converted_file_path = os.path.join(upload_folder, converted_filename)
            if os.path.exists(converted_file_path):
                return jsonify({
                    'success': True,
                    'existing': True,
                    'message': f"Converted file '{converted_filename}' already exists",
                    'filename': filename
                })

        try:
            # Process upload and save metadata
            result = upload_handler.process_upload(file, upload_folder, app.config['PREVIEW_FOLDER'])
            if result['success']:
                # Save upload metadata
                photo = Photo(
                    filename=result['filename'],
                    filepath=result['filepath'],
                    preview_filename=result.get('preview_filename'),
                    png_filename=result.get('png_filename'),
                    converted_filename=result.get('converted_filename'),
                    uploader_id=current_user.id,
                    upload_time=datetime.utcnow()
                )
                metadata = extract_metadata(result['filepath'])
                photo.timestamp = metadata.get('timestamp')
                photo.latitude = metadata.get('latitude')
                photo.longitude = metadata.get('longitude')
                db.session.add(photo)
                db.session.commit()

                # Update session['uploaded_files']
                if 'uploaded_files' not in session:
                    session['uploaded_files'] = []
                session['uploaded_files'].append(photo.id)
                session.modified = True  # Ensure the session is saved

                return jsonify({'success': True, 'message': 'File uploaded successfully'})
            else:
                return jsonify({'success': False, 'error': result.get('error', 'Unknown error')})
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            return jsonify({'success': False, 'error': str(e)})




    @app.route('/api/chunks', methods=['POST'])
    @login_required
    @csrf.exempt
    def upload_chunk():
        """Handle chunked file uploads with improved error handling."""
        try:
            # Validate request
            required_params = ['chunkIndex', 'totalChunks', 'filename']
            if not all(param in request.form for param in required_params) or 'file' not in request.files:
                return jsonify({'success': False, 'error': 'Missing required parameters'}), 400

            # Get request parameters
            chunk_index = int(request.form['chunkIndex'])
            total_chunks = int(request.form['totalChunks'])
            filename = secure_filename(request.form['filename'])
            chunk = request.files['file']

            # Setup paths
            upload_folder = current_app.config['UPLOAD_FOLDER']
            final_path = os.path.join(upload_folder, filename)
            chunk_folder = os.path.join(upload_folder, 'chunks', filename)

            # Check if final file exists
            if os.path.exists(final_path):
                return jsonify({
                    'success': True,
                    'existing': True,
                    'message': f"File '{filename}' already exists",
                    'filename': filename
                })

            # Ensure chunk directory exists
            os.makedirs(chunk_folder, exist_ok=True)
            chunk_path = os.path.join(chunk_folder, f'chunk_{chunk_index}')

            # Save the chunk
            chunk.save(chunk_path)

            # Check if all chunks received
            uploaded_chunks = len(os.listdir(chunk_folder))
            if uploaded_chunks == total_chunks:
                try:
                    # Assemble chunks
                    with open(final_path, 'wb') as final_file:
                        for i in range(total_chunks):
                            chunk_file_path = os.path.join(chunk_folder, f'chunk_{i}')
                            with open(chunk_file_path, 'rb') as chunk_file:
                                final_file.write(chunk_file.read())

                    # Clean up chunks
                    shutil.rmtree(chunk_folder)

                    # Process the assembled file
                    with open(final_path, 'rb') as assembled_file:
                        result = upload_handler.process_upload(
                            assembled_file,
                            upload_folder,
                            current_app.config['PREVIEW_FOLDER'],
                            filename=filename,
                            file_already_saved=True
                        )

                    if result['success']:
                        # Create database entry
                        try:
                            photo = Photo(
                                filename=result['filename'],
                                filepath=result['filepath'],
                                preview_filename=result.get('preview_filename'),
                                uploader_id=current_user.id,
                                upload_time=datetime.utcnow(),
                                is_video=result.get('is_video', False)
                            )

                            # Extract and save metadata
                            metadata = extract_metadata(final_path)
                            photo.timestamp = metadata.get('timestamp')
                            photo.latitude = metadata.get('latitude')
                            photo.longitude = metadata.get('longitude')

                            db.session.add(photo)
                            db.session.commit()

                            # Update session
                            if 'uploaded_files' not in session:
                                session['uploaded_files'] = []
                            session['uploaded_files'].append(photo.id)
                            session.modified = True

                            return jsonify({
                                'success': True, 
                                'message': 'File uploaded and processed successfully'
                            })

                        except Exception as e:
                            db.session.rollback()
                            logging.error(f"Database error: {str(e)}")
                            # Delete the file if database insertion fails
                            if os.path.exists(final_path):
                                os.remove(final_path)
                            return jsonify({
                                'success': False, 
                                'error': 'Database error occurred'
                            }), 500

                    else:
                        if os.path.exists(final_path):
                            os.remove(final_path)
                        return jsonify({
                            'success': False, 
                            'error': result.get('error', 'Processing error occurred')
                        })

                except Exception as e:
                    logging.error(f"Error assembling chunks: {str(e)}")
                    if os.path.exists(final_path):
                        os.remove(final_path)
                    if os.path.exists(chunk_folder):
                        shutil.rmtree(chunk_folder)
                    return jsonify({'success': False, 'error': str(e)}), 500

            return jsonify({
                'success': True, 
                'message': f'Chunk {chunk_index + 1} of {total_chunks} received'
            })

        except Exception as e:
            logging.error(f"Chunk upload error: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500


        
    @app.route('/preview_upload', methods=['GET', 'POST'])
    @login_required
    @csrf.exempt
    def preview_upload():
        if 'uploaded_files' not in session or not session['uploaded_files']:
            flash('No uploaded files to preview.', 'warning')
            return redirect(url_for('upload'))

        photos = Photo.query.filter(Photo.id.in_(session['uploaded_files'])).all()
        form = GPXUploadForm()

        if form.validate_on_submit():
            # Handle GPX file uploads
            if form.gpx_file.data:
                try:
                    all_gpx_points = []
                    for gpx_storage in form.gpx_file.data:
                        filename = gpx_storage.filename
                        if allowed_file(filename):
                            gpx = gpxpy.parse(gpx_storage.stream)
                            gpx_points = extract_gpx_points(gpx)
                            all_gpx_points.extend(gpx_points)
                        else:
                            flash(f'Invalid GPX file: {filename}', 'danger')
                    if all_gpx_points:
                        # Process GPX data
                        tag_photos_with_gpx_points(photos, all_gpx_points)
                        db.session.commit()
                        flash('Photos tagged with location data from Health data.', 'success')
                    else:
                        flash('No valid GPX data found.', 'danger')
                except Exception as e:
                    logging.error(f'Error processing GPX files: {e}')
                    flash('Error processing GPX files.', 'danger')
            else:
                flash('No GPX files uploaded.', 'danger')
            return redirect(url_for('preview_upload'))  # Redirect to avoid form resubmission

        elif request.method == 'POST':
            if 'batch_accept' in request.form:
                session.pop('uploaded_files', None)
                flash('All photos accepted.', 'success')
                return redirect(url_for('index'))

        return render_template('preview_upload.html', photos=photos, form=form)

    # Update the edit_metadata route
    @app.route('/edit_metadata/<int:photo_id>', methods=['GET', 'POST'])
    @login_required
    @csrf.exempt
    def edit_metadata(photo_id):
        photo = Photo.query.get_or_404(photo_id)
        if not can_modify_photo(current_user, photo):
            return jsonify({'success': False, 'message': 'You do not have permission to edit this photo.'}), 403

        if request.method == 'POST':
            try:
                if request.is_json:
                    data = request.get_json()
                else:
                    data = request.form.to_dict()

                photo.latitude = float(data['latitude']) if data.get('latitude') else None
                photo.longitude = float(data['longitude']) if data.get('longitude') else None
                photo.timestamp = datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None
                
                db.session.commit()
                flash('Metadata updated successfully.', 'success')
                return redirect(url_for('photo_view', photo_id=photo.id))
            except (ValueError, KeyError, BadRequest) as e:
                db.session.rollback()
                flash(f'Error updating metadata: {str(e)}', 'danger')
            except Exception as e:
                db.session.rollback()
                flash('An unexpected error occurred while updating metadata.', 'danger')
            return redirect(url_for('edit_metadata', photo_id=photo.id))

        form = EditMetadataForm(obj=photo)
        return render_template('edit_metadata.html', form=form, photo=photo)

    @app.route('/photo/<int:photo_id>', methods=['GET', 'POST'])
    @login_required
    @csrf.exempt
    def photo_view(photo_id):
        photo = Photo.query.get_or_404(photo_id)
        form = CommentForm()
        if form.validate_on_submit():
            comment = Comment(content=form.content.data, author=current_user, photo=photo)
            db.session.add(comment)
            db.session.commit()
            flash('Your memory has been added.', 'success')
            return redirect(url_for('photo_view', photo_id=photo_id))
        comments = Comment.query.filter_by(photo_id=photo_id).order_by(Comment.timestamp.desc()).all()
        is_favorite = Favorite.query.filter_by(user_id=current_user.id, photo_id=photo_id).first()

        # Get previous and next photos
        previous_photo = Photo.query.filter(Photo.id < photo_id).order_by(Photo.id.desc()).first()
        next_photo = Photo.query.filter(Photo.id > photo_id).order_by(Photo.id.asc()).first()

        return render_template(
            'photo.html',
            photo=photo,
            form=form,
            comments=comments,
            is_favorite=is_favorite,
            previous_photo=previous_photo,
            next_photo=next_photo
        )

    @app.route('/delete_comment/<int:comment_id>')
    @login_required
    @csrf.exempt

    def delete_comment(comment_id):
        comment = Comment.query.get_or_404(comment_id)
        if comment.author != current_user:
            flash('You can only delete your own comments.', 'danger')
            return redirect(url_for('photo_view', photo_id=comment.photo_id))
        db.session.delete(comment)
        db.session.commit()
        flash('Your comment has been deleted.', 'info')
        return redirect(url_for('photo_view', photo_id=comment.photo_id))

    @app.route('/favorite/<int:photo_id>')
    @login_required
    @csrf.exempt

    def favorite(photo_id):
        photo = Photo.query.get_or_404(photo_id)
        if not Favorite.query.filter_by(user_id=current_user.id, photo_id=photo_id).first():
            favorite = Favorite(user=current_user, photo=photo)
            db.session.add(favorite)
            db.session.commit()
            flash('Photo added to favorites.', 'success')
        else:
            flash('Photo is already in your favorites.', 'info')
        return redirect(url_for('photo_view', photo_id=photo_id))

    @app.route('/unfavorite/<int:photo_id>')
    @login_required
    @csrf.exempt

    def unfavorite(photo_id):
        favorite = Favorite.query.filter_by(user_id=current_user.id, photo_id=photo_id).first()
        if favorite:
            db.session.delete(favorite)
            db.session.commit()
            flash('Photo removed from favorites.', 'info')
        else:
            flash('Photo not in your favorites.', 'info')
        return redirect(url_for('photo_view', photo_id=photo_id))
    
     

    @app.route('/get_photos')
    @login_required
    def get_photos():
        try:
            # Get all filter parameters
            page = request.args.get('page', 1, type=int)
            show_favorites = request.args.get('favorites', 'false').lower() == 'true'
            uploader_id = request.args.get('uploader', type=int)
            date_from = request.args.get('date_from')
            date_to = request.args.get('date_to')
            photos_per_page = 20

            # Start building the query
            query = Photo.query

            # Apply filters
            if show_favorites:
                query = query.join(Photo.favorites).filter(Favorite.user_id == current_user.id)
            
            if uploader_id:
                query = query.filter(Photo.uploader_id == uploader_id)
                # Get total count for this user
                total_count = query.count()
                # If we're requesting a page that would be beyond this user's photos, return empty
                if (page - 1) * photos_per_page >= total_count:
                    return jsonify([])
            
            if date_from:
                try:
                    date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
                    query = query.filter(Photo.timestamp >= date_from_obj)
                except ValueError:
                    logging.error(f"Invalid date_from format: {date_from}")
            
            if date_to:
                try:
                    date_to_obj = datetime.strptime(date_to, '%Y-%m-%d')
                    query = query.filter(Photo.timestamp <= date_to_obj)
                except ValueError:
                    logging.error(f"Invalid date_to format: {date_to}")

            # Make the query distinct to avoid duplicates from joins
            query = query.distinct()
            
            # Apply ordering
            query = query.order_by(Photo.upload_time.desc())

            # Add debug logging for the SQL query
            logging.info(f"SQL Query: {query}")

            # Apply pagination
            paginated_photos = query.paginate(
                page=page,
                per_page=photos_per_page,
                error_out=False
            )

            # Get the items from pagination
            photos = paginated_photos.items

            # If no photos found for this page, return empty list
            if not photos:
                return jsonify([])

            photo_list = []
            current_user_id = None
            if uploader_id:
                current_user_id = uploader_id
                total_count = query.count()
            else:
                total_count = None

            for photo in photos:
                # Skip photos from other users if we're filtering by user
                if current_user_id and photo.uploader_id != current_user_id:
                    continue

                # Get appropriate URL using the get_photo_url function
                preview_url = get_photo_url(photo)

                photo_data = {
                    'id': photo.id,
                    'filename': photo.filename,
                    'preview_url': preview_url,
                    'png_url': url_for('static', filename=f'uploads/{photo.png_filename}') if photo.png_filename else None,
                    'converted_url': url_for('static', filename=f'uploads/{photo.converted_filename}') if photo.converted_filename else None,
                    'original_url': url_for('static', filename=f'uploads/{photo.filename}') 
                                if not photo.filename.lower().endswith(('.dng', '.heic')) else None,
                    'uploader_id': photo.uploader_id,
                    'timestamp': photo.timestamp.isoformat() if photo.timestamp else None,
                    'total_for_user': total_count,
                    'is_video': photo.is_video  # Include is_video property

                }
                photo_list.append(photo_data)

            # Add debug logging
            logging.info(f"Returning {len(photo_list)} photos for page {page}")
            if uploader_id:
                logging.info(f"Filtered for uploader_id {uploader_id}, total photos: {total_count}")

            return jsonify(photo_list)

        except Exception as e:
            logging.error(f"Error in get_photos: {str(e)}")
            return jsonify({'error': 'An error occurred while fetching photos'}), 500
                
    @app.route('/timeline')
    @login_required
    @csrf.exempt
    def timeline():
        return render_template('timeline.html')

    # Ensure all necessary fields are available when retrieving photos
    @app.route('/get_timeline_photos')
    @login_required
    @csrf.exempt
    def get_timeline_photos():
        photos = Photo.query.order_by(Photo.timestamp).all()
        
        location_clusters = []
        for photo in photos:
            if photo.latitude and photo.longitude and photo.timestamp:
                added_to_cluster = False
                for cluster in location_clusters:
                    # Group photos that are within 400 meters of each other
                    if geodesic((photo.latitude, photo.longitude), (cluster[0]['latitude'], cluster[0]['longitude'])).meters < 400:
                        cluster.append({
                            'latitude': photo.latitude,
                            'longitude': photo.longitude,
                            'filename': photo.png_filename if photo.png_filename else photo.filename,
                            'url': url_for('photo_view', photo_id=photo.id),
                            'timestamp': photo.timestamp.isoformat()  # Keep the timestamp for slider functionality
                        })
                        added_to_cluster = True
                        break
                if not added_to_cluster:
                    location_clusters.append([{
                        'latitude': photo.latitude,
                        'longitude': photo.longitude,
                        'filename': photo.png_filename if photo.png_filename else photo.filename,
                        'url': url_for('photo_view', photo_id=photo.id),
                        'timestamp': photo.timestamp.isoformat()  # Include the timestamp
                    }])

        # Format the data for JavaScript
        photo_list = []
        for cluster in location_clusters:
            avg_lat = sum([loc['latitude'] for loc in cluster]) / len(cluster)
            avg_lon = sum([loc['longitude'] for loc in cluster]) / len(cluster)
            photo_list.append({
                'latitude': avg_lat,
                'longitude': avg_lon,
                'count': len(cluster),
                'photos': cluster  # Pass the photos, each with its timestamp
            })
        
        return jsonify(photo_list)


    @app.route('/stats')
    @login_required
    @csrf.exempt
    def stats():
        try:
            total_photos = Photo.query.count()
            total_storage = 0
            for photo in Photo.query.all():
                try:
                    total_storage += os.path.getsize(photo.filepath)
                except OSError as e:
                    logging.error(f"Error getting file size for {photo.filepath}: {str(e)}")
            total_storage = total_storage / (1024 * 1024)  # Convert to MB

            # Calculate most popular photo times
            photos = Photo.query.all()
            time_counter = Counter()
            day_counter = Counter()
            for photo in photos:
                if photo.timestamp:
                    hour = photo.timestamp.hour
                    day_of_week = photo.timestamp.strftime('%A')
                    time_counter[hour] += 1
                    day_counter[day_of_week] += 1
            most_common_hour = time_counter.most_common(1)[0] if time_counter else (None, 0)
            most_common_day = day_counter.most_common(1)[0] if day_counter else (None, 0)

            # Popular locations with photo grouping and handling non-DNG versions
            location_clusters = []
            for photo in photos:
                if photo.latitude and photo.longitude:
                    added_to_cluster = False

                    # Use the converted PNG filename if the photo is a DNG
                    if photo.filename.lower().endswith('.dng'):
                        if photo.png_filename:
                            photo_url = url_for('static', filename=f'uploads/{photo.png_filename}')
                        else:
                            logging.warning(f"No converted PNG found for DNG {photo.filename}, skipping...")
                            continue  # Skip this photo if no converted image exists
                    else:
                        # Non-DNG files are valid
                        photo_url = url_for('static', filename=f'uploads/{photo.filename}')
                    
                    # Add to cluster only if there's a valid URL
                    for cluster in location_clusters:
                        if geodesic((photo.latitude, photo.longitude), (cluster[0]['latitude'], cluster[0]['longitude'])).meters < 400:
                            cluster.append({'latitude': photo.latitude, 'longitude': photo.longitude, 'photo': photo, 'photo_url': photo_url})
                            added_to_cluster = True
                            break
                    if not added_to_cluster:
                        location_clusters.append([{'latitude': photo.latitude, 'longitude': photo.longitude, 'photo': photo, 'photo_url': photo_url}])

            # Average locations for clusters and get photos
            popular_locations = []
            for cluster in location_clusters:
                avg_lat = sum([loc['latitude'] for loc in cluster]) / len(cluster)
                avg_lon = sum([loc['longitude'] for loc in cluster]) / len(cluster)
                photos_in_cluster = [{'filename': loc['photo_url'], 'photo_id': loc['photo'].id, 'latitude': loc['latitude'], 'longitude': loc['longitude']} for loc in cluster]
                popular_locations.append({
                    'latitude': avg_lat,
                    'longitude': avg_lon,
                    'count': len(cluster),
                    'photos': photos_in_cluster  # Include photos in this cluster
                })

            popular_locations = sorted(popular_locations, key=lambda loc: loc['count'], reverse=True)

            # Pass popular locations to the template for the map
            return render_template('stats.html', 
                                total_photos=total_photos, 
                                total_storage=total_storage,
                                popular_locations=popular_locations[:5],  # Limit to top 5 locations
                                most_common_hour=most_common_hour, 
                                most_common_day=most_common_day)
        except Exception as e:
            logging.error(f"Error in stats route: {str(e)}")
            flash('An error occurred while generating statistics.', 'danger')
            return redirect(url_for('index'))

    # Update the delete_photo route
    @app.route('/delete_photo/<int:photo_id>', methods=['POST'])
    @login_required
    @csrf.exempt
    def delete_photo(photo_id):
        photo = Photo.query.get_or_404(photo_id)
        
        if not can_modify_photo(current_user, photo):
            flash('You do not have permission to delete this photo.', 'danger')
            return redirect(url_for('photo_view', photo_id=photo_id))

        try:
            # Delete associated records first
            Favorite.query.filter_by(photo_id=photo_id).delete()
            Comment.query.filter_by(photo_id=photo_id).delete()

            # Delete the actual files
            if os.path.exists(photo.filepath):
                os.remove(photo.filepath)

            if photo.png_filename:
                png_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], photo.png_filename)
                if os.path.exists(png_filepath):
                    os.remove(png_filepath)

            if photo.preview_filename:
                preview_filepath = os.path.join(current_app.config['PREVIEW_FOLDER'], photo.preview_filename)
                if os.path.exists(preview_filepath):
                    os.remove(preview_filepath)

            # Delete the database entry
            db.session.delete(photo)
            db.session.commit()
            
            flash('Photo and all associated data deleted successfully.', 'success')
        except Exception as e:
            db.session.rollback()
            logging.error(f'Error deleting photo {photo.filename}: {str(e)}')
            flash('An error occurred while trying to delete the photo.', 'danger')

        return redirect(url_for('index'))
    
    @app.route('/photo/<int:photo_id>/permissions')
    @login_required
    def check_photo_permissions(photo_id):
        """Helper endpoint to check permissions for a photo"""
        photo = Photo.query.get_or_404(photo_id)
        return jsonify({
            'can_edit': can_modify_photo(current_user, photo),
            'is_admin': is_admin(current_user),
            'is_owner': photo.uploader == current_user
        })

    @app.route('/photo/<int:photo_id>/details')
    @login_required
    @csrf.exempt
    def photo_details(photo_id):
        photo = Photo.query.get_or_404(photo_id)
        # Ensure the user has permission to view the photo if necessary
        photo_data = {
            'id': photo.id,
            'filename': photo.filename,
            'png_filename': photo.png_filename,
            'latitude': photo.latitude,
            'longitude': photo.longitude,
            'timestamp': photo.timestamp.isoformat() if photo.timestamp else ''
        }
        return jsonify(photo_data)
    

    @app.route('/account', methods=['GET', 'POST'])
    @login_required
    @csrf.exempt
    def account():
        from .forms import GPXUploadForm
        from .models import Photo
        from collections import Counter
        from geopy.distance import geodesic

        # Get stats for the current user's photos
        photos = Photo.query.filter_by(uploader=current_user).all()
        total_photos = len(photos)
        total_storage = sum(
            os.path.getsize(photo.filepath) for photo in photos if os.path.exists(photo.filepath)
        ) / (1024 * 1024)  # Convert to MB

        # Calculate most popular photo times
        time_counter = Counter()
        day_counter = Counter()
        for photo in photos:
            if photo.timestamp:
                hour = photo.timestamp.hour
                day_of_week = photo.timestamp.strftime('%A')
                time_counter[hour] += 1
                day_counter[day_of_week] += 1
        most_common_hour = time_counter.most_common(1)[0] if time_counter else (None, 0)
        most_common_day = day_counter.most_common(1)[0] if day_counter else (None, 0)

        # Popular locations with photo grouping
        location_clusters = []
        for photo in photos:
            if photo.latitude and photo.longitude:
                added_to_cluster = False
                photo_url = url_for('static', filename=f'uploads/{photo.png_filename if photo.png_filename else photo.filename}')
                
                for cluster in location_clusters:
                    if geodesic((photo.latitude, photo.longitude), (cluster[0]['latitude'], cluster[0]['longitude'])).meters < 400:
                        cluster.append({
                            'latitude': photo.latitude,
                            'longitude': photo.longitude,
                            'id': photo.id,
                            'filename': photo_url
                        })
                        added_to_cluster = True
                        break
                if not added_to_cluster:
                    location_clusters.append([{
                        'latitude': photo.latitude,
                        'longitude': photo.longitude,
                        'id': photo.id,
                        'filename': photo_url
                    }])

        # Average locations for clusters and get photos
        popular_locations = []
        for cluster in location_clusters:
            avg_lat = sum([loc['latitude'] for loc in cluster]) / len(cluster)
            avg_lon = sum([loc['longitude'] for loc in cluster]) / len(cluster)
            popular_locations.append({
                'latitude': avg_lat,
                'longitude': avg_lon,
                'count': len(cluster),
                'photos': cluster
            })

        popular_locations = sorted(popular_locations, key=lambda loc: loc['count'], reverse=True)[:5]

        # Initialize duplicates as an empty list
        duplicates = []

        # Form for uploading GPX files
        form = GPXUploadForm()
        if form.validate_on_submit():
            if form.gpx_files.data:
                try:
                    all_gpx_points = []
                    for gpx_storage in form.gpx_files.data:
                        filename = gpx_storage.filename
                        if allowed_file(filename):
                            gpx = gpxpy.parse(gpx_storage.stream)
                            gpx_points = extract_gpx_points(gpx)
                            all_gpx_points.extend(gpx_points)
                        else:
                            flash(f'Invalid GPX file: {filename}', 'danger')
                    if all_gpx_points:
                        # Process GPX data
                        tag_photos_with_gpx_points(photos, all_gpx_points)
                        db.session.commit()
                        flash('Your photos have been tagged with location data from GPX files.', 'success')
                    else:
                        flash('No valid GPX data found.', 'danger')
                except Exception as e:
                    logging.error(f'Error processing GPX files: {e}')
                    flash('Error processing GPX files.', 'danger')
            else:
                flash('No GPX files uploaded.', 'danger')
            return redirect(url_for('account'))  # Redirect to avoid form resubmission

        return render_template(
            'account.html',
            total_photos=total_photos,
            total_storage=total_storage,
            popular_locations=popular_locations,
            most_common_hour=most_common_hour,
            most_common_day=most_common_day,
            form=form,
            include_leaflet=True,
            duplicates=duplicates  # Add this line
        )

    # Route to scan for duplicate photos
    @app.route('/scan_duplicates', methods=['POST'])
    @login_required
    @csrf.exempt
    def scan_duplicates():
        from .forms import GPXUploadForm
        form = GPXUploadForm()

        photos = Photo.query.filter_by(uploader=current_user).all()
        
        # Calculate most common day and hour using the updated function
        most_common_hour, most_common_day, popular_locations = calculate_photo_statistics(photos)
        
        # Convert popular_locations to a serializable format
        serializable_locations = []
        for loc in popular_locations:
            serializable_photos = []
            for photo in loc['photos']:
                serializable_photos.append({
                    'id': photo['photo'].id,
                    'filename': photo['photo'].filename,
                    'latitude': photo['latitude'],
                    'longitude': photo['longitude']
                })
            serializable_locations.append({
                'latitude': loc['latitude'],
                'longitude': loc['longitude'],
                'count': loc['count'],
                'photos': serializable_photos
            })
        
        # Dictionary to store photos by their filename
        photo_dict = defaultdict(list)

        # Group photos by filename and timestamp
        for photo in photos:
            photo_dict[photo.filename].append(photo)
        
        # Identify duplicates (more than one photo with the same filename)
        duplicates = []
        for filename, photo_list in photo_dict.items():
            if len(photo_list) > 1:
                duplicates.extend(photo_list)

        # Convert duplicates to a serializable format
        serializable_duplicates = [{
            'id': photo.id,
            'filename': photo.filename,
            'timestamp': photo.timestamp.isoformat() if photo.timestamp else None
        } for photo in duplicates]

        total_storage = calculate_storage(photos)  # Calculate total storage
        
        return render_template('account.html', 
                            total_photos=len(photos), 
                            total_storage=total_storage,
                            most_common_day=most_common_day, 
                            most_common_hour=most_common_hour,
                            popular_locations=serializable_locations[:5], 
                            form=form, 
                            duplicates=serializable_duplicates)



    # Route to delete all duplicate photos

    @app.route('/delete_duplicates', methods=['POST'])
    @login_required
    @csrf.exempt
    def delete_duplicates():
        admin_email = os.getenv('ADMIN_EMAIL')
        if current_user.email.lower() != admin_email.lower():
            abort(403)  

        photos = Photo.query.filter_by(uploader=current_user).all()

        # Dictionary to store photos by their filename
        photo_dict = defaultdict(list)

        # Group photos by filename
        for photo in photos:
            photo_dict[photo.filename].append(photo)

        deleted_count = 0
        errors = []

        for filename, photo_list in photo_dict.items():
            if len(photo_list) > 1:
                # Sort photos by ID (assuming older entries have lower IDs)
                photo_list.sort(key=lambda x: x.id)
                
                # Keep the first (oldest) photo
                file_to_keep = photo_list[0]
                
                # Remove duplicate database entries, but don't delete files yet
                for photo in photo_list[1:]:
                    try:
                        db.session.delete(photo)
                        deleted_count += 1
                    except Exception as e:
                        errors.append(f"Error deleting photo {photo.id}: {str(e)}")

        # Commit changes to the database
        try:
            db.session.commit()
            flash(f'Deleted {deleted_count} duplicate database entries, retaining one entry for each unique filename.', 'success')
            
            # Now, clean up orphaned files
            orphaned_files = cleanup_orphaned_files()
            if orphaned_files:
                flash(f'Cleaned up {len(orphaned_files)} orphaned files.', 'info')
            
            # Run cleanup_broken_images
            cleanup_result = cleanup_broken_images()
            flash(cleanup_result, 'info')
            
        except Exception as e:
            db.session.rollback()
            errors.append(f'Error during cleanup process: {str(e)}')

        if errors:
            error_message = "The following errors occurred during the cleanup process:\n" + "\n".join(errors)
            flash(error_message, 'danger')
            logging.error(error_message)
        else:
            flash('Cleanup process completed successfully.', 'success')

        # Return a redirect response directly
        return redirect(url_for('account'))

    @app.route('/cleanup_broken_images', methods=['POST'])
    @login_required
    @csrf.exempt
    def cleanup_broken_images():
        admin_email = os.getenv('ADMIN_EMAIL')
        if current_user.email.lower() != admin_email.lower():
            abort(403) 

        photos = Photo.query.all()
        removed_photos = []
        problematic_photos = []

        for photo in photos:
            # Check if the main file exists
            file_exists = os.path.exists(photo.filepath)
            
            # Check if the PNG exists, if applicable
            png_exists = False
            png_filepath = None
            if photo.png_filename:
                png_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], photo.png_filename)
                png_exists = os.path.exists(png_filepath)
            
            # Log the status of each photo
            logging.info(f"Photo ID: {photo.id}, Filename: {photo.filename}")
            logging.info(f"  Original file exists: {file_exists}")
            logging.info(f"  PNG file exists: {png_exists}")
            
            # If neither the original file nor the PNG exists, it's a broken reference
            if not file_exists and not png_exists:
                try:
                    db.session.delete(photo)
                    removed_photos.append(photo.filename)
                    logging.info(f"  Marked for deletion: {photo.filename}")
                except Exception as e:
                    logging.error(f"  Error marking for deletion: {photo.filename}: {str(e)}")
            else:
                # Check if the files are readable
                if file_exists:
                    try:
                        with open(photo.filepath, 'rb') as f:
                            f.read(1)  # Try to read 1 byte
                        logging.info(f"  Original file is readable: {photo.filepath}")
                    except IOError as e:
                        problematic_photos.append((photo.filename, f"Original file not readable: {str(e)}"))
                        logging.warning(f"  Original file not readable: {photo.filepath}: {str(e)}")
                
                if png_exists:
                    try:
                        with open(png_filepath, 'rb') as f:
                            f.read(1)  # Try to read 1 byte
                        logging.info(f"  PNG file is readable: {png_filepath}")
                    except IOError as e:
                        problematic_photos.append((photo.png_filename, f"PNG file not readable: {str(e)}"))
                        logging.warning(f"  PNG file not readable: {png_filepath}: {str(e)}")

        # Commit the changes to remove broken references
        try:
            db.session.commit()
            result = f'Removed {len(removed_photos)} broken image entries from the database.'
            if problematic_photos:
                result += f' Found {len(problematic_photos)} problematic photos that may not display correctly.'
            for filename, error in problematic_photos:
                result += f'\nProblematic photo: {filename} - Error: {error}'
            return result
        except Exception as e:
            db.session.rollback()
            logging.error(f'Error committing cleanup changes: {str(e)}')
            return 'An error occurred during cleanup. Please try again.'


    @app.route('/check_photo_status/<int:photo_id>')
    @login_required
    @csrf.exempt
    def check_photo_status(photo_id):
        photo = Photo.query.get_or_404(photo_id)
        file_exists = os.path.exists(photo.filepath)
        file_readable = False
        
        if file_exists:
            try:
                with open(photo.filepath, 'rb') as f:
                    f.read(1)
                file_readable = True
            except IOError:
                file_readable = False
        
        return jsonify({
            'exists': file_exists,
            'readable': file_readable
        })
    
    @app.route('/recent_comments')
    @login_required
    @csrf.exempt
    def recent_comments():
        # Fetch the 20 most recent comments
        recent_comments = Comment.query.order_by(Comment.timestamp.desc()).limit(20).all()
        
        # For each comment, we need to fetch the associated photo
        comments_with_photos = []
        for comment in recent_comments:
            photo = Photo.query.get(comment.photo_id)
            comments_with_photos.append({
                'comment': comment,
                'photo': photo
            })
        
        return render_template('recent_comments.html', comments_with_photos=comments_with_photos)

    @app.route('/static/uploads/<filename>')
    @login_required
    def serve_upload(filename):
        """Serve original uploaded files with proper MIME type handling"""
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(upload_path):
            logging.error(f"Upload file not found: {upload_path}")
            abort(404)
            
        # Determine correct MIME type
        if filename.lower().endswith('.dng'):
            mime_type = 'image/x-adobe-dng'
        elif filename.lower().endswith('.png'):
            mime_type = 'image/png'
        elif filename.lower().endswith(('.jpg', '.jpeg')):
            mime_type = 'image/jpeg'
        else:
            mime_type = None
            
        try:
            return send_file(upload_path, mimetype=mime_type)
        except Exception as e:
            logging.error(f"Error serving upload {upload_path}: {str(e)}")
            abort(500)

    @app.route('/previews/<filename>')
    @login_required
    def serve_preview(filename):
        """Serve preview files with proper path handling and fallbacks"""
        # Handle DNG preview conversion
        preview_filename = filename.replace('.DNG', '.JPG').replace('.dng', '.jpg')
        
        # Use the PREVIEW_FOLDER environment variable
        preview_path = os.path.join(app.config['PREVIEW_FOLDER'], preview_filename)
        
        logging.info(f"Attempting to serve preview: {preview_path}")
        
        if os.path.exists(preview_path):
            try:
                return send_file(preview_path, mimetype='image/jpeg')
            except Exception as e:
                logging.error(f"Error serving preview {preview_path}: {str(e)}")
                abort(500)
        
        # If preview doesn't exist, try to find PNG version
        photo = Photo.query.filter_by(preview_filename=filename).first()
        if photo and photo.png_filename:
            png_path = os.path.join(app.config['UPLOAD_FOLDER'], photo.png_filename)
            if os.path.exists(png_path):
                try:
                    return send_file(png_path, mimetype='image/png')
                except Exception as e:
                    logging.error(f"Error serving PNG fallback {png_path}: {str(e)}")
                    abort(500)
        
        logging.error(f"No valid preview found for: {filename}")
        abort(404)


    @app.route('/stream/<path:filename>')
    @login_required
    def stream_video(filename):
        try:
            video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(video_path):
                abort(404)
            
            file_size = os.path.getsize(video_path)
            logging.info(f"Streaming video {filename}, size: {file_size}")
            range_header = request.headers.get('Range', None)
            logging.info(f"Range header: {range_header}")
            if range_header:
                # Parse the Range header
                range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
                    if end >= file_size:
                        end = file_size - 1
                    chunk_size = end - start + 1
                    logging.info(f"Sending bytes {start}-{end}/{file_size}, chunk_size: {chunk_size}")
                    with open(video_path, 'rb') as f:
                        f.seek(start)
                        data = f.read(chunk_size)
                    response = Response(data, 206, mimetype=get_mimetype(filename), direct_passthrough=True)
                    response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
                    response.headers.add('Accept-Ranges', 'bytes')
                    response.headers.add('Content-Length', str(chunk_size))
                    return response
                else:
                    logging.error(f"Invalid Range header: {range_header}")
                    return Response(status=416)
            else:
                # No Range header; send the entire file
                logging.info("No Range header, sending entire file")
                return send_file(video_path, mimetype=get_mimetype(filename))
        except Exception as e:
            logging.error(f"Error streaming {filename}: {str(e)}")
            return 'Error streaming video', 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)