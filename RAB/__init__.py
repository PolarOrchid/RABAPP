from flask import Flask
from .config import Config
from .extensions import db, login_manager
from flask_mail import Mail
from itsdangerous import URLSafeTimedSerializer
from flask_migrate import Migrate  # Add this line to import Flask-Migrate
from .app import create_app

mail = Mail()
s = URLSafeTimedSerializer(Config.SECRET_KEY)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    mail.init_app(app)

    # Initialize Flask-Migrate
    migrate = Migrate(app, db)  # Add this line to initialize Flask-Migrate

    with app.app_context():
        db.create_all()

    return app
