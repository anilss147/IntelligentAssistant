"""
Jarvis AI Assistant - Main Entry Point

This is the main entry point for the Jarvis AI Assistant application.
It initializes all components and starts the application.
"""

from app import app, db

# Make sure all database tables are created
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)