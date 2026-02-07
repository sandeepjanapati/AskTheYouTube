/**
 * AskTheYouTube - Frontend Logic
 * Handles API communication, State Management, and UI Rendering.
 */

// CONFIGURATION
// TODO: Replace with your actual Cloud Run URL after backend deployment
// const API_BASE_URL = "http://localhost:8080";
const API_BASE_URL = "https://asktheyoutube-backend-923028230772.us-central1.run.app";

// STATE MANAGEMENT
const state = {
    currentVideoId: null,
    chatHistory: [],
    isProcessing: false
};

// DOM ELEMENTS
const elements = {
    // Sections
    videoSection: document.getElementById('video-section'),
    statusSection: document.getElementById('status-section'),
    chatSection: document.getElementById('chat-section'),

    // Inputs & Buttons
    urlInput: document.getElementById('youtube-url'),
    processBtn: document.getElementById('process-btn'),
    queryInput: document.getElementById('query-input'),
    sendBtn: document.getElementById('send-btn'),

    // Status Elements
    statusText: document.getElementById('status-text'),
    errorBox: document.getElementById('error-box'),
    errorMessage: document.getElementById('error-message'),
    retryBtn: document.getElementById('retry-btn'),

    newChatBtn: document.getElementById('new-chat-btn'),

    // Chat Area
    chatHistoryContainer: document.getElementById('chat-history')
};

// --- INITIALIZATION ---

document.addEventListener('DOMContentLoaded', () => {
    loadSession(); // Check if we have a saved session
});

function loadSession() {
    const savedVideoId = sessionStorage.getItem('atyt_video_id');
    const savedHistory = sessionStorage.getItem('atyt_history');

    if (savedVideoId) {
        state.currentVideoId = savedVideoId;
        state.chatHistory = savedHistory ? JSON.parse(savedHistory) : [];

        // Restore UI to Chat Mode
        showSection('chat');

        // Render saved messages
        // Clear default welcome message if history exists
        if (state.chatHistory.length > 0) {
            // Remove the hardcoded welcome message from HTML to avoid duplicates or keep it at top
            // For this implementation, we simply append history below it.
        }

        state.chatHistory.forEach(msg => {
            appendMessageToUI(msg.role, msg.content, false); // false = don't save again
        });

        scrollToBottom();
    }
}

function saveState() {
    sessionStorage.setItem('atyt_video_id', state.currentVideoId);
    sessionStorage.setItem('atyt_history', JSON.stringify(state.chatHistory));
}

// --- VIEW CONTROLLER ---

function showSection(sectionName) {
    // Hide all
    elements.videoSection.classList.add('hidden');
    elements.statusSection.classList.add('hidden');
    elements.chatSection.classList.add('hidden');

    // Show Target
    if (sectionName === 'video') elements.videoSection.classList.remove('hidden');
    if (sectionName === 'status') elements.statusSection.classList.remove('hidden');
    if (sectionName === 'chat') elements.chatSection.classList.remove('hidden');

    // --- Toggle New Chat Button ---
    // Only show button if NOT in video input section
    if (sectionName === 'video') {
        elements.newChatBtn.classList.add('hidden');
    } else {
        elements.newChatBtn.classList.remove('hidden');
    }
}

function showError(msg) {
    elements.errorBox.classList.remove('hidden');
    elements.errorMessage.textContent = msg;
    // Hide loader
    document.querySelector('.loader').classList.add('hidden');
    elements.statusText.classList.add('hidden');
}

function resetStatusUI() {
    elements.errorBox.classList.add('hidden');
    document.querySelector('.loader').classList.remove('hidden');
    elements.statusText.classList.remove('hidden');
    elements.statusText.textContent = "Processing video...";
}

// --- EVENT LISTENERS ---

// 1. Process Video
elements.processBtn.addEventListener('click', handleProcessVideo);

// 2. Retry Logic
elements.retryBtn.addEventListener('click', () => {
    showSection('video');
    resetStatusUI();
});

// 3. Send Message (Click)
elements.sendBtn.addEventListener('click', handleSendMessage);

// 4. Send Message (Enter Key)
elements.queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevent newline
        handleSendMessage();
    }
});

// 5. Auto-resize Textarea
elements.queryInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if (this.value === '') this.style.height = 'auto';

    // Enable/Disable button
    elements.sendBtn.disabled = this.value.trim().length === 0;
});

// 6. New Chat / Reset
elements.newChatBtn.addEventListener('click', handleNewChat);


// --- LOGIC HANDLERS ---

async function handleProcessVideo() {
    const url = elements.urlInput.value.trim();

    if (API_BASE_URL.includes("YOUR_CLOUD_RUN_API_URL")) {
        alert("Configuration Error: Please open app.js and replace 'YOUR_CLOUD_RUN_API_URL' with your actual Cloud Run URL.");
        return;
    }

    if (!url) {
        alert("Please enter a YouTube URL");
        return;
    }

    // Update UI State
    state.isProcessing = true;
    showSection('status');
    elements.statusText.textContent = "Analyzing transcript (this may take a moment for long videos)...";

    try {
        const response = await fetch(`${API_BASE_URL}/process-video`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });


        // --- SAFE RESPONSE HANDLING START ---
        // 1. Check if response is JSON
        const contentType = response.headers.get("content-type");
        let data;

        if (contentType && contentType.indexOf("application/json") !== -1) {
            // It is JSON, parse it safely
            data = await response.json();
        } else {
            // It is NOT JSON (likely HTML error or empty), read as text
            const text = await response.text();

            // If response was not OK, throw the text as error
            if (!response.ok) {
                throw new Error(text || `Server Error: ${response.status} ${response.statusText}`);
            }

            // If response WAS OK but not JSON, this is unexpected for your API
            throw new Error("Invalid Server Response: Expected JSON but got text/html.");
        }
        // 2. Handle API Level Errors (Non-200 status codes)
        if (!response.ok) {
            throw new Error(data.detail || "Failed to process video");
        }
        // --- SAFE RESPONSE HANDLING END ---

        // Success
        state.currentVideoId = data.video_id;
        state.isProcessing = false;

        saveState();
        showSection('chat');

    } catch (error) {
        console.error("Processing Error:", error);
        state.isProcessing = false;
        let uiMessage = error.message;
        if (uiMessage.includes("Unexpected end of JSON")) {
            uiMessage = "Server returned an empty response. Check your API URL and Backend Logs.";
        }
        showError(uiMessage);
    }
}

async function handleSendMessage() {
    const query = elements.queryInput.value.trim();

    if (!query || state.isProcessing) return;

    // 1. UI Updates immediately
    appendMessageToUI('user', query);
    elements.queryInput.value = '';
    elements.queryInput.style.height = 'auto';
    elements.sendBtn.disabled = true;

    // Add User msg to history
    state.chatHistory.push({ role: 'user', content: query });
    saveState();

    // 2. Show Typing Indicator
    const typingId = showTypingIndicator();
    scrollToBottom();

    // 3. API Call
    try {
        const payload = {
            query: query,
            video_id: state.currentVideoId,
            history: state.chatHistory.slice(0, -1) // Send history excluding current query (optional, depends on backend logic)
        };

        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        // Remove typing indicator
        removeTypingIndicator(typingId);

        if (!response.ok) {
            throw new Error("Failed to get response");
        }

        const data = await response.json();
        const botResponse = data.response; // Assuming backend returns { "response": "..." }

        // 4. Render Bot Response
        appendMessageToUI('model', botResponse);

        // Update History
        state.chatHistory.push({ role: 'model', content: botResponse });
        saveState();

    } catch (error) {
        removeTypingIndicator(typingId);
        appendMessageToUI('model', "**Error:** I couldn't reach the server. Please try again.");
    }
}

function handleNewChat() {
    // 1. Clear Session Storage
    sessionStorage.removeItem('atyt_video_id');
    sessionStorage.removeItem('atyt_history');

    // 2. Reset State
    state.currentVideoId = null;
    state.chatHistory = [];
    state.isProcessing = false;

    // 3. Reset UI Inputs
    elements.urlInput.value = '';
    elements.queryInput.value = '';
    elements.sendBtn.disabled = true;

    // 4. Reset Chat History UI
    // We recreate the default welcome message
    elements.chatHistoryContainer.innerHTML = `
        <div class="message bot-message">
            <div class="message-content">
                <p><strong>Video Loaded!</strong> <br> I've analyzed the transcript. You can now ask me anything about this video.</p>
            </div>
        </div>
    `;

    // 5. Navigate back to Home
    showSection('video');
}


// --- DOM HELPERS ---

function appendMessageToUI(role, text, animate = true) {
    const isUser = role === 'user';

    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message');
    msgDiv.classList.add(isUser ? 'user-message' : 'bot-message');

    // Markdown Parsing for Bot
    // using 'marked' library included in index.html
    const contentHtml = isUser ? text : marked.parse(text);

    msgDiv.innerHTML = `<div class="message-content">${contentHtml}</div>`;

    elements.chatHistoryContainer.appendChild(msgDiv);
    scrollToBottom();
}

function showTypingIndicator() {
    const id = 'typing-' + Date.now();

    const msgDiv = document.createElement('div');
    msgDiv.id = id;
    msgDiv.classList.add('message', 'bot-message');
    msgDiv.innerHTML = `
        <div class="message-content">
            <i class="fa-solid fa-circle-notch fa-spin"></i> Thinking...
        </div>
    `;

    elements.chatHistoryContainer.appendChild(msgDiv);
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    elements.chatHistoryContainer.scrollTop = elements.chatHistoryContainer.scrollHeight;
}