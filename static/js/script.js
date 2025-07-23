document.addEventListener('DOMContentLoaded', function() {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    // Function to append a message to the chat window
    function appendMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`, 'mb-2');
        messageDiv.innerHTML = `<p>${message}</p>`;
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll to bottom
    }

    // Function to send a message to the Flask backend
    async function sendMessage() {
        const query = userInput.value.trim();
        if (query === "") {
            return; // Don't send empty messages
        }

        appendMessage('user', query); // Display user's message immediately
        userInput.value = ''; // Clear input field

        try {
            // Show a typing indicator (optional but good for UX)
            const typingIndicator = document.createElement('div');
            typingIndicator.classList.add('message', 'bot-message', 'mb-2', 'typing-indicator');
            typingIndicator.innerHTML = '<p>Bot is typing...</p>';
            chatWindow.appendChild(typingIndicator);
            chatWindow.scrollTop = chatWindow.scrollHeight;

            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            });

            // Remove typing indicator
            chatWindow.removeChild(typingIndicator);

            if (response.ok) {
                const data = await response.json();
                appendMessage('bot', data.answer);
            } else {
                console.error('Error:', response.statusText);
                appendMessage('bot', 'Oops! Something went wrong. Please try again.');
            }
        } catch (error) {
            console.error('Network error:', error);
            appendMessage('bot', 'A network error occurred. Please check your connection.');
        }
    }

    // Event listeners
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});