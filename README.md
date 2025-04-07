# Jarvis AI Assistant

An offline Jarvis-like AI assistant using Python, local LLMs, and open-source tools for voice interaction and task execution.

## Features

- **Completely Offline** - No internet connection required for core functionality
- **Voice Interaction** - Speak to your assistant and hear its responses
- **Local Language Models** - Uses locally run LLMs for intelligence
- **Task Execution** - Open apps, set reminders, answer questions, and more
- **Memory** - Remembers past conversations and learns from them
- **Customizable** - Easy to configure and extend with new capabilities

## Requirements

- Python 3.8+
- 8+ GB of RAM (16+ GB recommended for larger models)
- GPU with 4+ GB VRAM (optional, but recommended for faster response)
- Microphone and speakers/headphones

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/jarvis-ai-assistant.git
   cd jarvis-ai-assistant
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required models:
   ```bash
   python main.py --download-models
   ```

## Usage

### Starting the Assistant

```bash
python main.py
