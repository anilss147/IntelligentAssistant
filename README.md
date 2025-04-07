# Jarvis AI Assistant

An offline Jarvis-like AI assistant using Python, local LLMs, and open-source tools for voice interaction and task execution.

## Features

- **Completely Offline** - No internet connection required for core functionality
- **Voice Interaction** - Speak to your assistant and hear its responses
- **Local Language Models** - Uses locally run LLMs for intelligence
- **Task Execution** - Open apps, set reminders, answer questions, and more
- **Memory** - Remembers past conversations and learns from them
- **Customizable** - Easy to configure and extend with new capabilities
- **User Authentication** - Secure login and registration system

## Deployment with GitHub Actions

This project includes GitHub Actions workflow for continuous integration and deployment.

### Setup GitHub Repository

1. Create a new repository on GitHub
2. Push your code to the repository
3. GitHub Actions will automatically run the workflow defined in `.github/workflows/python-app.yml`

### Required Secrets for GitHub Actions

For deployment, add these secrets to your GitHub repository:

- `SESSION_SECRET`: A random string for session security
- `DATABASE_URL`: Your production database URL
- `TWILIO_ACCOUNT_SID`: Your Twilio account SID (if using SMS)
- `TWILIO_AUTH_TOKEN`: Your Twilio auth token (if using SMS)
- `TWILIO_PHONE_NUMBER`: Your Twilio phone number (if using SMS)

### Deployment Options

The workflow can be configured to deploy to platforms like Heroku or any other PaaS by updating the deployment section in the workflow file.

### Running Locally

```bash
python main.py
```

The application will be available at `http://localhost:5000`
