# celery_worker.py

from celery import Celery
from flask import Flask
from .config import Config
from .extensions import db
from .models import Photo
from .utils import extract_metadata, generate_png_from_dng
import os
import logging
import numpy as np
import imageio

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

# Initialize Flask app for Celery
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions with app
db.init_app(app)

celery = make_celery(app)

@celery.task(bind=True)
def process_uploaded_file(self, photo_id):
    logging.info(f'Starting process_uploaded_file for Photo ID {photo_id}')
    try:
        with app.app_context():
            photo = Photo.query.get(photo_id)
            if not photo:
                logging.error(f'Photo with id {photo_id} not found.')
                return

            logging.info(f'Processing DNG file: {photo.filepath}')
            
            # If it's a DNG file, generate a PNG version for display
            if photo.filename.lower().endswith('.dng'):
                try:
                    upload_folder = app.config['UPLOAD_FOLDER']
                    png_filename = os.path.splitext(photo.filename)[0] + '.png'
                    png_filepath = os.path.join(upload_folder, png_filename)
                    
                    logging.info(f'Generating PNG from DNG: {png_filepath}')
                    generate_png_from_dng(photo.filepath, png_filepath)
                    
                    if os.path.exists(png_filepath):
                        photo.png_filename = png_filename
                        logging.info(f'Generated PNG for Photo ID {photo_id}: {png_filename}')
                    else:
                        raise FileNotFoundError(f"PNG file not found after generation: {png_filepath}")
                        
                except Exception as e:
                    logging.error(f'Error generating PNG from DNG {photo.filename}: {str(e)}')
                    photo.png_filename = None

            # Extract metadata after PNG generation
            logging.info(f'Extracting metadata for Photo ID {photo_id}.')
            metadata = extract_metadata(photo.filepath)
            if metadata:
                photo.timestamp = metadata.get('timestamp')
                photo.latitude = metadata.get('latitude')
                photo.longitude = metadata.get('longitude')
                logging.info(f'Updated photo metadata: timestamp={photo.timestamp}, '
                           f'latitude={photo.latitude}, longitude={photo.longitude}')
            
            db.session.commit()
            logging.info(f'Photo ID {photo_id} processing completed successfully.')
            
    except Exception as e:
        logging.error(f'Error processing uploaded file {photo_id}: {str(e)}')
        db.session.rollback()
    finally:
        db.session.close()