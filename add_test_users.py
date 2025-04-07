"""
Script to add test users to the database.
This creates 5 test users and 1 admin user.
"""

from app import app, db
import sys
import time

def add_test_users():
    """Create and add test users to the database."""
    
    # Import User model inside app context to avoid metadata errors
    with app.app_context():
        # Import all models to ensure tables are created
        from models import User, Memory, Conversation, Reminder
        
        # Create all tables in the database
        print("Creating database tables...")
        db.create_all()
        print("Database tables created.")
        
        # List of test users to create
        users = [
            # Regular users
            {
                'email': 'user1@example.com',
                'phone': '9876543210',
                'password': 'Password123',
                'is_admin': False
            },
            {
                'email': 'user2@example.com',
                'phone': '9876543211',
                'password': 'Password123',
                'is_admin': False
            },
            {
                'email': 'user3@example.com',
                'phone': '9876543212',
                'password': 'Password123',
                'is_admin': False
            },
            {
                'email': 'user4@example.com',
                'phone': '9876543213',
                'password': 'Password123',
                'is_admin': False
            },
            {
                'email': 'user5@example.com',
                'phone': '9876543214',
                'password': 'Password123',
                'is_admin': False
            },
            # Admin user
            {
                'email': 'anilsherikar207@gmail.com',
                'phone': '9876543215',
                'password': 'AdminPass123',
                'is_admin': True
            }
        ]
        
        created_count = 0
        skipped_count = 0
        
        for user_data in users:
            try:
                # Check if user already exists
                existing_user = User.query.filter(
                    (User.email == user_data['email']) | 
                    (User.phone == user_data['phone'])
                ).first()
                
                if existing_user:
                    print(f"Skipping user with email {user_data['email']} - already exists")
                    skipped_count += 1
                    continue
                
                # Create new user
                new_user = User(
                    email=user_data['email'],
                    phone=user_data['phone'],
                    password=user_data['password'],
                    is_admin=user_data['is_admin']
                )
                
                db.session.add(new_user)
                created_count += 1
                print(f"Created user: {user_data['email']} ({'Admin' if user_data['is_admin'] else 'Regular'})")
            except Exception as e:
                print(f"Error creating user {user_data['email']}: {str(e)}")
        
        # Commit changes to database
        try:
            db.session.commit()
            print("Database changes committed successfully.")
        except Exception as e:
            db.session.rollback()
            print(f"Error committing changes to database: {str(e)}")
    
        print(f"\nSummary: Created {created_count} users, Skipped {skipped_count} users")
        print("\nTest User Credentials:")
        print("----------------------")
        print("Regular Users:")
        print("Username: user1@example.com (or any user1-5)")
        print("Password: Password123")
        print("\nAdmin User:")
        print("Username: anilsherikar207@gmail.com")
        print("Password: AdminPass123")

if __name__ == "__main__":
    add_test_users()