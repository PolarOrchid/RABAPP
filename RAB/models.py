# models.py

from .extensions import db, login_manager
from flask_login import UserMixin
from datetime import datetime

# User model
class User(db.Model, UserMixin):
    __tablename__ = 'user'  # Specify table name to avoid conflicts
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    photos = db.relationship('Photo', backref='uploader', lazy=True)
    comments = db.relationship('Comment', backref='author', lazy=True)
    favorites = db.relationship('Favorite', backref='user', lazy=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Photo model with cascade delete for favorites
class Photo(db.Model):
    __tablename__ = 'photo'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    filepath = db.Column(db.String(200), nullable=False)
    preview_filename = db.Column(db.String(128), nullable=False)  # Store preview image path

    png_filename = db.Column(db.String(100), nullable=True)  # Field for storing PNG version of DNG
    upload_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=True)
    uploader_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    comments = db.relationship('Comment', backref='photo', lazy=True)
    time_diff = db.Column(db.Float, nullable=True)  # Time difference in seconds

    # Cascade deletion of favorites when a photo is deleted
    favorites = db.relationship('Favorite', cascade="all, delete-orphan", backref='photo', lazy=True)
    converted_filename = db.Column(db.String(255))

# Comment model
class Comment(db.Model):
    __tablename__ = 'comment'
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    photo_id = db.Column(db.Integer, db.ForeignKey('photo.id'), nullable=False)
    author_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Favorite model
class Favorite(db.Model):
    __tablename__ = 'favorite'
    id = db.Column(db.Integer, primary_key=True)
    photo_id = db.Column(db.Integer, db.ForeignKey('photo.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
