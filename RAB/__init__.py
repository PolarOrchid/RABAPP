# RAB/__init__.py

from flask import Flask
from .config import Config
from .extensions import db, login_manager
from flask_mail import Mail
from itsdangerous import URLSafeTimedSerializer
from flask_migrate import Migrate

mail = Mail()
s = URLSafeTimedSerializer(Config.SECRET_KEY)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    mail.init_app(app)

    # Import models to ensure they are registered with SQLAlchemy
    from .models import User, Photo, Comment, Favorite

    # Initialize Flask-Migrate after models are imported
    migrate = Migrate(app, db)

    return app
