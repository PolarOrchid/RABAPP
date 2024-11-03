# RAB/__init__.py

import os
from flask import Flask, redirect, url_for, flash
from flask_compress import Compress
from flask_mail import Mail
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from itsdangerous import URLSafeTimedSerializer
from dotenv import load_dotenv
from werkzeug.exceptions import Unauthorized

from .config import Config
from .extensions import db, login_manager

# Load environment variables
load_dotenv()

# Initialize extensions at module level
mail = Mail()
migrate = Migrate()
compress = Compress()
csrf = CSRFProtect()
s = URLSafeTimedSerializer(Config.SECRET_KEY)

def create_app(config_class=Config):
    """Application factory function"""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    
    # Initialize CSRF with exempt views
    csrf.init_app(app)
    
    # Configure login manager before other extensions
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    # Custom unauthorized handler that works with flask-login
    @login_manager.unauthorized_handler
    def unauthorized_handler():
        flash('Please log in to access this page.', 'info')
        return redirect(url_for('login', next=request.url))
    
    # Handle 401 errors globally
    @app.errorhandler(401)
    def unauthorized_error(error):
        flash('Please log in to access this page.', 'info')
        return redirect(url_for('login'))
        
    # Handle Unauthorized exception
    @app.errorhandler(Unauthorized)
    def unauthorized_exception(error):
        flash('Please log in to access this page.', 'info')
        return redirect(url_for('login'))
    
    # Initialize remaining extensions
    mail.init_app(app)
    compress.init_app(app)
    
    # Import models within app context
    with app.app_context():
        from .models import User, Photo, Comment, Favorite
        migrate.init_app(app, db)
        
        # Create required directories
        required_dirs = [
            os.getenv('UPLOAD_FOLDER'),
            os.getenv('PREVIEW_FOLDER'),
            os.getenv('CHUNK_FOLDER')
        ]
        
        for directory in required_dirs:
            if directory:
                os.makedirs(directory, exist_ok=True)
    
    # CSRF exempt the login view
    csrf.exempt(app.view_functions['login'])
    
    return app