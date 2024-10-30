# models.py

# models.py
from .extensions import db, login_manager
from flask_login import UserMixin
from datetime import datetime

class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    photos = db.relationship('Photo', backref='uploader', lazy=True)
    comments = db.relationship('Comment', backref='author', lazy=True)
    favorites = db.relationship('Favorite', backref='user', lazy=True)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

class Photo(db.Model):
    __tablename__ = 'photo'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    filepath = db.Column(db.String(200), nullable=False)
    preview_filename = db.Column(db.String(128), nullable=True)  # Changed to nullable
    png_filename = db.Column(db.String(100), nullable=True)
    upload_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=True)
    uploader_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    time_diff = db.Column(db.Float, nullable=True)
    converted_filename = db.Column(db.String(255), nullable=True)
    is_video = db.Column(db.Boolean, default=False, nullable=False)
    
    # Relationships
    comments = db.relationship('Comment', backref='photo', lazy=True, 
                             cascade="all, delete-orphan")
    favorites = db.relationship('Favorite', backref='photo', lazy=True,
                              cascade="all, delete-orphan")

class Comment(db.Model):
    __tablename__ = 'comment'
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    photo_id = db.Column(db.Integer, db.ForeignKey('photo.id'), nullable=False)
    author_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Favorite(db.Model):
    __tablename__ = 'favorite'
    id = db.Column(db.Integer, primary_key=True)
    photo_id = db.Column(db.Integer, db.ForeignKey('photo.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)