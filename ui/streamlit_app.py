"""
Streamlit UI for Jarvis AI Assistant.

This file contains the Streamlit-based user interface for the Jarvis AI Assistant.
"""

import os
import sys
import logging
import time
import threading
import streamlit as st
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import components
from components.llm_handler import LLMHandler
from components.speech_processor import SpeechProcessor
from components.memory import Memory
from components.task_executor import TaskExecutor
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Jarvis AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Bootstrap CSS
st.markdown("""
<link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet" />
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: rgba(var(--bs-tertiary-bg-rgb), 0.2);
    border-left: 5px solid var(--bs-tertiary-color);
}
.chat-message.assistant {
    background-color: rgba(var(--bs-secondary-bg-rgb), 0.2);
    border-left: 5px solid var(--bs-info);
}
.chat-message .message-content {
    display: flex;
    margin-bottom: 0;
}
.chat-message .avatar {
    width: 20%;
    display: flex;
    align-items: center;
    justify-content: center;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .content {
    width: 80%;
    padding-left: 1rem;
}
.speech-button {
    border-radius: 50%;
    width: 60px;
    height: 60px;
    font-size: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.status-indicator {
    display: flex;
    align-items: center;
    padding: 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.8rem;
}
.status-indicator.listening {
    background-color: rgba(var(--bs-danger-rgb), 0.1);
    color: var(--bs-danger);
}
.status-indicator.thinking {
    background-color: rgba(var(--bs-warning-rgb), 0.1);
    color: var(--bs-warning);
}
.status-indicator.speaking {
    background-color: rgba(var(--bs-success-rgb), 0.1);
    color: var(--bs-success);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_components():
    """Initialize and cache the assistant components."""
    logger.info("Initializing assistant components...")
    config = Config()
    llm = LLMHandler(config)
    speech = SpeechProcessor(config)
    memory = Memory(config)
    executor = TaskExecutor(config)
    
    return config, llm, speech, memory, executor

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    if "recording" not in st.session_state:
        st.session_state["recording"] = False
    
    if "status" not in st.session_state:
        st.session_state["status"] = "idle"  # idle, listening, thinking, speaking
    
    if "history_loaded" not in st.session_state:
        st.session_state["history_loaded"] = False

def get_avatar(role):
    """Get avatar for chat message."""
    if role == "user":
        return "https://api.dicebear.com/7.x/personas/svg?seed=user"
    else:
        return "https://api.dicebear.com/7.x/bottts/svg?seed=jarvis"

def display_message(role, content):
    """Display a chat message."""
    avatar_url = get_avatar(role)
    
    message_html = f"""
    <div class="chat-message {role}">
        <div class="message-content">
            <div class="avatar">
                <img src="{avatar_url}" alt="{role}">
            </div>
            <div class="content">
                <p>{content}</p>
            </div>
        </div>
    </div>
    """
    
    st.markdown(message_html, unsafe_allow_html=True)

def display_chat_history():
    """Display chat history from session state."""
    for message in st.session_state["messages"]:
        display_message(message["role"], message["content"])

def start_recording():
    """Start recording audio."""
    if st.session_state["recording"]:
        return
    
    st.session_state["recording"] = True
    st.session_state["status"] = "listening"
    
    # Start recording in a thread
    def record_thread():
        speech.start_recording()
    
    threading.Thread(target=record_thread).start()

def stop_recording():
    """Stop recording and process audio."""
    if not st.session_state["recording"]:
        return
    
    st.session_state["recording"] = False
    st.session_state["status"] = "thinking"
    
    # Stop recording and get transcription
    transcription = speech.stop_recording()
    
    if transcription:
        # Add user message to session
        st.session_state["messages"].append({"role": "user", "content": transcription})
        
        # Add to memory
        memory.add_to_conversation("user", transcription)
        
        # Get conversation history
        conversation_history = memory.get_conversation_history()
        
        # Generate response
        st.session_state["status"] = "thinking"
        response = llm.generate_response(transcription, conversation_history)
        
        # Add assistant message to session
        st.session_state["messages"].append({"role": "assistant", "content": response})
        
        # Add to memory
        memory.add_to_conversation("assistant", response)
        
        # Speak response
        st.session_state["status"] = "speaking"
        threading.Thread(target=lambda: speech.speak(response)).start()
    
    st.session_state["status"] = "idle"

def process_text_input():
    """Process text input."""
    user_input = st.session_state["text_input"]
    if not user_input:
        return
    
    # Clear input
    st.session_state["text_input"] = ""
    
    # Add user message to session
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Add to memory
    memory.add_to_conversation("user", user_input)
    
    # Get conversation history
    conversation_history = memory.get_conversation_history()
    
    # Generate response
    st.session_state["status"] = "thinking"
    response = llm.generate_response(user_input, conversation_history)
    
    # Add assistant message to session
    st.session_state["messages"].append({"role": "assistant", "content": response})
    
    # Add to memory
    memory.add_to_conversation("assistant", response)
    
    # Speak response
    st.session_state["status"] = "speaking"
    threading.Thread(target=lambda: speech.speak(response)).start()
    
    st.session_state["status"] = "idle"

def load_conversation_history():
    """Load conversation history from memory."""
    if st.session_state["history_loaded"]:
        return
    
    conversation_history = memory.get_conversation_history()
    
    if conversation_history:
        st.session_state["messages"] = [
            {"role": entry["role"], "content": entry["content"]}
            for entry in conversation_history
        ]
    
    st.session_state["history_loaded"] = True

def clear_conversation():
    """Clear conversation history."""
    memory.clear_conversation_history()
    st.session_state["messages"] = []
    st.session_state["history_loaded"] = True

def show_status_indicator():
    """Show status indicator."""
    status = st.session_state["status"]
    
    if status == "listening":
        st.markdown("""
        <div class="status-indicator listening">
            <i class="fas fa-microphone"></i> Listening...
        </div>
        """, unsafe_allow_html=True)
    elif status == "thinking":
        st.markdown("""
        <div class="status-indicator thinking">
            <i class="fas fa-brain"></i> Thinking...
        </div>
        """, unsafe_allow_html=True)
    elif status == "speaking":
        st.markdown("""
        <div class="status-indicator speaking">
            <i class="fas fa-comment"></i> Speaking...
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function for Streamlit UI."""
    # Initialize components
    config, llm, speech, memory, executor = initialize_components()
    
    # Initialize session state
    initialize_session_state()
    
    # Load conversation history
    load_conversation_history()
    
    # Sidebar
    with st.sidebar:
        st.title("Jarvis AI Assistant")
        st.markdown("Your offline AI assistant")
        
        st.header("Settings")
        
        # Speech settings
        st.subheader("Speech")
        voices = speech.list_available_voices()
        if voices:
            voice_name = st.selectbox("TTS Voice", options=voices, index=0)
            if st.button("Apply Voice"):
                speech.change_voice(voice_name)
        
        speech_rate = st.slider("Speech Rate", min_value=100, max_value=250, value=int(config.get_speech_config()["tts_rate"]), step=5)
        if st.button("Apply Rate"):
            speech.set_speech_rate(speech_rate)
        
        speech_volume = st.slider("Speech Volume", min_value=0.0, max_value=1.0, value=float(config.get_speech_config()["tts_volume"]), step=0.1)
        if st.button("Apply Volume"):
            speech.set_speech_volume(speech_volume)
        
        # LLM settings
        st.subheader("Language Model")
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=float(config.get_llm_config()["temperature"]), step=0.1)
        max_tokens = st.slider("Max Tokens", min_value=50, max_value=1000, value=int(config.get_llm_config()["max_new_tokens"]), step=50)
        
        if st.button("Apply LLM Settings"):
            llm.update_model_parameters(temperature=temperature, max_new_tokens=max_tokens)
        
        # Actions
        st.subheader("Actions")
        if st.button("Clear Conversation"):
            clear_conversation()
    
    # Main chat container
    chat_container = st.container()
    
    with chat_container:
        # Welcome message
        if not st.session_state["messages"]:
            welcome_message = config.get_app_config()["assistant_prompt"]
            st.session_state["messages"].append({"role": "assistant", "content": welcome_message})
            memory.add_to_conversation("assistant", welcome_message)
        
        # Display chat history
        display_chat_history()
        
        # Status indicator
        status_placeholder = st.empty()
        with status_placeholder:
            show_status_indicator()
    
    # Input container
    input_container = st.container()
    
    with input_container:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.text_input("Type your message...", key="text_input", on_change=process_text_input)
        
        with col2:
            # Centered recording button
            st.markdown("""
            <div style="display: flex; justify-content: center;">
            """, unsafe_allow_html=True)
            
            if st.session_state["recording"]:
                if st.button("🛑", key="stop_recording", help="Stop recording"):
                    stop_recording()
            else:
                if st.button("🎤", key="start_recording", help="Start recording"):
                    start_recording()
            
            st.markdown("""
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
