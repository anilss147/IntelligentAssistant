"""
Flask application for Jarvis AI Assistant.

This file provides a web interface for the Jarvis AI Assistant.
"""

import os
import logging
import time
import json
import re
from flask import Flask, render_template, request, jsonify, Response, send_from_directory, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.security import generate_password_hash, check_password_hash
from email_validator import validate_email, EmailNotValidError

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.secret_key = os.environ.get("SESSION_SECRET", "jarvis-assistant-secret-key")

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize SQLAlchemy
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Import components
from components.task_executor import TaskExecutor
from config import Config

# Initialize configuration and task executor
config = Config()
task_executor = TaskExecutor(config)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))

# Create forms for user authentication
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, ValidationError
from wtforms.validators import DataRequired, Email, EqualTo, Length, Regexp

class LoginForm(FlaskForm):
    """Form for user login."""
    identity = StringField('Email or Phone', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    """Form for user registration."""
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = StringField('Phone (Indian Format)', validators=[
        Regexp(r'^(\+91|91)?[6-9][0-9]{9}$', message='Please enter a valid Indian phone number.')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, message='Password must be at least 8 characters long.')
    ])
    password2 = PasswordField(
        'Confirm Password', validators=[DataRequired(), EqualTo('password', message='Passwords must match.')]
    )
    submit = SubmitField('Register')
    
    def validate_email(self, field):
        from models import User
        if not field.data and not self.phone.data:
            raise ValidationError('Either Email or Phone is required.')
        if field.data and User.query.filter_by(email=field.data).first():
            raise ValidationError('Email already registered.')
    
    def validate_phone(self, field):
        from models import User
        if not field.data and not self.email.data:
            raise ValidationError('Either Email or Phone is required.')
        if field.data and User.query.filter_by(phone=field.data).first():
            raise ValidationError('Phone number already registered.')

# Create authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login route."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        from models import User
        
        # Check if identity is email or phone
        identity = form.identity.data
        if '@' in identity:
            user = User.query.filter_by(email=identity).first()
        else:
            # Clean phone number format
            phone = identity.strip()
            if phone.startswith('+91'):
                phone = phone[3:]
            elif phone.startswith('91'):
                phone = phone[2:]
            user = User.query.filter_by(phone=phone).first()
        
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login'))
        
        login_user(user, remember=form.remember_me.data)
        user.update_last_login()
        
        # Redirect to the page the user was trying to access
        next_page = request.args.get('next')
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('index')
        
        return redirect(next_page)
    
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
@login_required
def logout():
    """User logout route."""
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration route."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        from models import User
        
        # Clean phone number format if provided
        phone = None
        if form.phone.data:
            phone = form.phone.data.strip()
            if phone.startswith('+91'):
                phone = phone[3:]
            elif phone.startswith('91'):
                phone = phone[2:]
        
        # Check if this is the admin user (anilsherikar207@gmail.com)
        is_admin = form.email.data == 'anilsherikar207@gmail.com'
        
        user = User(
            email=form.email.data if form.email.data else None,
            phone=phone,
            password=form.password.data,
            is_admin=is_admin
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Congratulations, you are now a registered user!', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', title='Register', form=form)

@app.route('/profile')
@login_required
def profile():
    """User profile page."""
    return render_template('profile.html', title='Profile')

@app.route('/')
def index():
    """Render the main page."""
    # Require login for main interface
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
        
    app_config = config.get_app_config()
    assistant_name = app_config["name"]
    assistant_prompt = app_config["assistant_prompt"]
    
    return render_template(
        'index.html',
        title=f"{assistant_name} - AI Assistant",
        assistant_name=assistant_name,
        assistant_prompt=assistant_prompt
    )

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

# Add datetime filter to Jinja environment
@app.template_filter('datetime')
def _jinja2_filter_datetime(timestamp, fmt=None):
    if timestamp is None:
        return ''
    if fmt:
        return time.strftime(fmt, time.localtime(timestamp))
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))

# Conversation storage
conversation_history = []

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """Process chat messages."""
    global conversation_history
    data = request.json
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Add to conversation history
    conversation_history.append({"role": "user", "content": user_input, "timestamp": int(time.time())})
    
    # Generate response (simple mock response for now)
    if "time" in user_input.lower():
        result = task_executor.execute_command("get_time")
        if result["success"]:
            time_info = result["time"]
            response = f"The current time is {time_info['formatted']} on {time_info['date']} ({time_info['day_of_week']})."
        else:
            response = "Sorry, I couldn't get the current time."
    elif "system" in user_input.lower() or "computer" in user_input.lower():
        result = task_executor.execute_command("get_system_info")
        if result["success"]:
            sys_info = result["system_info"]
            response = f"You're running {sys_info['platform']} {sys_info['platform_release']} on {sys_info['architecture']} architecture."
        else:
            response = "Sorry, I couldn't get system information."
    else:
        response = f"I understand you're asking about '{user_input}'. As a simple demo, I can only respond to basic queries about time and system info."
    
    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": response, "timestamp": int(time.time())})
    
    return jsonify({
        "response": response,
        "timestamp": int(time.time())
    })

@app.route('/api/command', methods=['POST'])
@login_required
def execute_command():
    """Execute a command."""
    data = request.json
    command_type = data.get('command_type', '')
    params = data.get('params', {})
    
    if not command_type:
        return jsonify({"error": "No command type provided"}), 400
    
    result = task_executor.execute_command(command_type, params)
    
    return jsonify(result)

@app.route('/api/conversation/clear', methods=['POST'])
@login_required
def clear_conversation():
    """Clear conversation history."""
    global conversation_history
    conversation_history = []
    
    return jsonify({
        "success": True
    })

@app.route('/api/conversation/history', methods=['GET'])
@login_required
def get_conversation_history():
    """Get conversation history."""
    limit = request.args.get('limit', None, type=int)
    
    if limit:
        history = conversation_history[-limit:]
    else:
        history = conversation_history
    
    return jsonify({
        "history": history,
        "count": len(history)
    })

@app.route('/api/assistant/info', methods=['GET'])
def get_assistant_info():
    """Get information about the assistant."""
    return jsonify({
        "name": config.get_app_config()["name"],
        "prompt": config.get_app_config()["assistant_prompt"]
    })

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "timestamp": int(time.time())
    })

# Handle cleanup on shutdown
@app.teardown_appcontext
def shutdown_cleanup(exception=None):
    """Clean up resources on shutdown."""
    if task_executor is not None:
        task_executor.cleanup()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
