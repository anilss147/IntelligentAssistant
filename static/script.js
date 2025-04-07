// Jarvis AI Assistant - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const messageForm = document.getElementById('messageForm');
    const userInput = document.getElementById('userInput');
    const messagesContainer = document.getElementById('messagesContainer');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const statusIndicator = document.getElementById('statusIndicator');
    const micButton = document.getElementById('micButton');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    const sidebarClose = document.getElementById('sidebarClose');
    const commandButtons = document.querySelectorAll('.command-btn');

    // Initialize
    loadConversationHistory();

    // Event Listeners
    messageForm.addEventListener('submit', handleSubmitMessage);
    clearChatBtn.addEventListener('click', clearConversation);
    commandButtons.forEach(button => {
        button.addEventListener('click', () => {
            userInput.value = button.getAttribute('data-command');
            messageForm.dispatchEvent(new Event('submit'));
            toggleSidebar(false);
        });
    });

    // Mobile sidebar handling
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => toggleSidebar(true));
    }
    if (sidebarClose) {
        sidebarClose.addEventListener('click', () => toggleSidebar(false));
    }

    // Function to toggle sidebar on mobile
    function toggleSidebar(show) {
        if (window.innerWidth < 768) {
            if (show) {
                sidebar.classList.add('show');
                // Add backdrop
                const backdrop = document.createElement('div');
                backdrop.classList.add('sidebar-backdrop');
                backdrop.addEventListener('click', () => toggleSidebar(false));
                document.body.appendChild(backdrop);
            } else {
                sidebar.classList.remove('show');
                // Remove backdrop
                const backdrop = document.querySelector('.sidebar-backdrop');
                if (backdrop) {
                    backdrop.remove();
                }
            }
        }
    }

    // Handle message submission
    function handleSubmitMessage(e) {
        e.preventDefault();
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message to UI
        addMessageToUI('user', message);
        
        // Clear input
        userInput.value = '';

        // Update status
        updateStatus('thinking');

        // Send to backend
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        })
        .then(response => response.json())
        .then(data => {
            // Add assistant response to UI
            addMessageToUI('assistant', data.response);
            updateStatus('ready');
        })
        .catch(error => {
            console.error('Error:', error);
            addMessageToUI('assistant', 'Sorry, there was an error processing your request.');
            updateStatus('ready');
        });
    }

    // Add message to UI
    function addMessageToUI(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message', role);

        // Avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.classList.add('avatar');
        const icon = document.createElement('i');
        icon.classList.add('bi', role === 'user' ? 'bi-person' : 'bi-robot');
        avatarDiv.appendChild(icon);

        // Content
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('content');
        contentDiv.innerHTML = `<p>${formatMessage(content)}</p>`;

        // Append to message
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);

        // Add to container
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Format message with markdown-like syntax
    function formatMessage(text) {
        // Convert URLs to links
        text = text.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');
        
        // Convert *text* to bold
        text = text.replace(/\*([^*]+)\*/g, '<strong>$1</strong>');
        
        // Convert _text_ to italic
        text = text.replace(/\_([^_]+)\_/g, '<em>$1</em>');
        
        // Convert `code` to code spans
        text = text.replace(/\`([^\`]+)\`/g, '<code>$1</code>');
        
        // Handle line breaks
        text = text.replace(/\n/g, '<br>');
        
        return text;
    }

    // Update status indicator
    function updateStatus(status) {
        if (status === 'thinking') {
            statusIndicator.innerHTML = `
                <span class="text-warning">Thinking</span>
                <div class="typing-indicator ms-2">
                    <span></span><span></span><span></span>
                </div>
            `;
        } else if (status === 'listening') {
            statusIndicator.innerHTML = '<span class="text-danger">Listening...</span>';
        } else {
            statusIndicator.innerHTML = '<span class="text-info">Ready</span>';
        }
    }

    // Load conversation history from the server
    function loadConversationHistory() {
        fetch('/api/conversation/history')
            .then(response => response.json())
            .then(data => {
                if (data.history && data.history.length > 0) {
                    // Clear existing messages
                    while (messagesContainer.childNodes.length > 1) {
                        messagesContainer.removeChild(messagesContainer.lastChild);
                    }
                    
                    // Add messages from history
                    data.history.forEach(msg => {
                        addMessageToUI(msg.role, msg.content);
                    });
                }
            })
            .catch(error => {
                console.error('Error loading conversation history:', error);
            });
    }

    // Clear conversation history
    function clearConversation() {
        fetch('/api/conversation/clear', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Keep only the first message (welcome message)
                    while (messagesContainer.childNodes.length > 1) {
                        messagesContainer.removeChild(messagesContainer.lastChild);
                    }
                }
            })
            .catch(error => {
                console.error('Error clearing conversation:', error);
            });
    }

    // Handle Enter key to submit form
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            messageForm.dispatchEvent(new Event('submit'));
        }
    });
});
