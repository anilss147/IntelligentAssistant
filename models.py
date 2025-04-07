"""
Database models for Jarvis AI Assistant.
"""

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db
import time

class User(UserMixin, db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    phone = db.Column(db.String(15), unique=True, nullable=True)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.Integer, default=lambda: int(time.time()))
    last_login = db.Column(db.Integer, nullable=True)
    
    # Relationship with Memory entries
    memories = db.relationship('Memory', backref='user', lazy=True)
    
    # Relationship with Conversation entries
    conversations = db.relationship('Conversation', backref='user', lazy=True)
    
    def __init__(self, email=None, phone=None, password=None, is_admin=False):
        self.email = email
        self.phone = phone
        if password:
            self.set_password(password)
        self.is_admin = is_admin
    
    def set_password(self, password):
        """Set password hash."""
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        """Check password hash."""
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        """Update last login timestamp."""
        self.last_login = int(time.time())
        db.session.commit()
    
    @property
    def identity(self):
        """Return user's identity (email or phone)."""
        return self.email if self.email else self.phone
    
    def __repr__(self):
        return f'<User {self.identity}>'


class Memory(db.Model):
    """Model for storing long-term memories."""
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    embedding = db.Column(db.Text, nullable=True)  # Store as serialized vector
    meta_data = db.Column(db.JSON, nullable=True)  # Renamed from 'metadata' as it's a reserved name
    created_at = db.Column(db.Integer, default=lambda: int(time.time()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def __repr__(self):
        return f'<Memory {self.id}: {self.content[:30]}...>'


class Conversation(db.Model):
    """Model for storing conversation history."""
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.Integer, default=lambda: int(time.time()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def __repr__(self):
        return f'<Conversation {self.id}: {self.role} - {self.content[:30]}...>'


class Reminder(db.Model):
    """Model for storing reminders."""
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    scheduled_time = db.Column(db.Integer, nullable=True)
    is_completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.Integer, default=lambda: int(time.time()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def __repr__(self):
        return f'<Reminder {self.id}: {self.text[:30]}...>'