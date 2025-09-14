class RAGChatApp {
    constructor() {
        this.isSystemReady = false;
        this.isProcessing = false;
        this.chatMessages = document.getElementById('chat-messages');
        this.userInput = document.getElementById('user-input');
        this.sendBtn = document.getElementById('send-btn');
        this.statusIndicator = document.getElementById('status-indicator');
        this.statusText = document.getElementById('status-text');
        
        this.initializeEventListeners();
        this.checkSystemStatus();
        this.startStatusPolling();
    }

    initializeEventListeners() {
        // Send button click
        this.sendBtn.addEventListener('click', () => this.handleSendMessage());
        
        // Enter key to send message
        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendMessage();
            }
        });
        
        // Auto-resize textarea
        this.userInput.addEventListener('input', () => this.autoResizeTextarea());
        
        // Info button
        document.getElementById('info-btn').addEventListener('click', () => this.showSystemInfo());
        
        // Close modal when clicking outside
        window.addEventListener('click', (e) => {
            const modal = document.getElementById('info-modal');
            if (e.target === modal) {
                this.closeModal();
            }
        });
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            this.updateStatusIndicator(status);
            
            if (status.ready && !this.isSystemReady) {
                this.isSystemReady = true;
                this.enableInput();
                this.showSystemReadyMessage();
            }
        } catch (error) {
            console.error('Error checking system status:', error);
            this.updateStatusIndicator({
                ready: false,
                status: 'Connection Error'
            });
        }
    }

    startStatusPolling() {
        // Poll status every 2 seconds until system is ready
        const pollInterval = setInterval(() => {
            if (this.isSystemReady) {
                clearInterval(pollInterval);
                return;
            }
            this.checkSystemStatus();
        }, 2000);
    }

    updateStatusIndicator(status) {
        this.statusText.textContent = status.status;
        
        // Remove all status classes
        this.statusIndicator.classList.remove('ready', 'loading', 'error');
        
        // Add appropriate class
        if (status.ready) {
            this.statusIndicator.classList.add('ready');
        } else if (status.status.includes('Error')) {
            this.statusIndicator.classList.add('error');
        } else {
            this.statusIndicator.classList.add('loading');
        }
    }

    enableInput() {
        this.userInput.disabled = false;
        this.sendBtn.disabled = false;
        this.userInput.placeholder = "Ask me anything about your documents...";
    }

    showSystemReadyMessage() {
        const readyMessage = this.createBotMessage(
            "🎉 System is ready! I've successfully loaded and processed your documents. You can now ask me questions about their content. What would you like to know?"
        );
        this.chatMessages.appendChild(readyMessage);
        this.scrollToBottom();
    }

    async handleSendMessage() {
        if (!this.isSystemReady || this.isProcessing) return;
        
        const query = this.userInput.value.trim();
        if (!query) return;
        
        // Add user message to chat
        this.addUserMessage(query);
        this.userInput.value = '';
        this.autoResizeTextarea();
        
        // Show thinking animation
        const thinkingElement = this.showThinkingAnimation();
        
        // Disable input while processing
        this.isProcessing = true;
        this.userInput.disabled = true;
        this.sendBtn.disabled = true;
        
        try {
            // Send query to API
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query, top_k: 3 })
            });
            
            const result = await response.json();
            
            // Remove thinking animation
            this.removeThinkingAnimation(thinkingElement);
            
            if (response.ok) {
                // Add bot response
                this.addBotMessage(result.answer, result.retrieved_docs);
            } else {
                // Handle error
                this.addBotMessage(
                    `❌ Sorry, I encountered an error: ${result.error || 'Unknown error'}`,
                    []
                );
            }
        } catch (error) {
            console.error('Error sending query:', error);
            this.removeThinkingAnimation(thinkingElement);
            this.addBotMessage(
                "❌ Sorry, I'm having trouble connecting to the server. Please try again.",
                []
            );
        } finally {
            // Re-enable input
            this.isProcessing = false;
            if (this.isSystemReady) {
                this.userInput.disabled = false;
                this.sendBtn.disabled = false;
                this.userInput.focus();
            }
        }
    }

    addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'user-message';
        messageElement.innerHTML = `
            <div class="user-avatar">
                <i class="fas fa-user"></i>
            </div>
            <div class="message-content">
                ${this.formatText(message)}
            </div>
        `;
        this.chatMessages.appendChild(messageElement);
        this.scrollToBottom();
    }

    addBotMessage(message, sources = []) {
        const botMessage = this.createBotMessage(message, sources);
        this.chatMessages.appendChild(botMessage);
        this.scrollToBottom();
    }

    createBotMessage(message, sources = []) {
        const messageElement = document.createElement('div');
        messageElement.className = 'bot-message';
        
        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            sourcesHtml = `
                <div class="source-documents">
                    <h4><i class="fas fa-book"></i> Source Documents (${sources.length})</h4>
                    ${sources.map((source, index) => `
                        <div class="source-doc" onclick="showSourcePreview('${this.escapeHtml(source.content)}', '${this.escapeHtml(source.source)}')">
                            <div class="source-doc-header">
                                <span class="source-title">${source.source}</span>
                                <span class="similarity-score">${(source.similarity_score * 100).toFixed(1)}% match</span>
                            </div>
                            <div class="source-preview">${source.content}</div>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        messageElement.innerHTML = `
            <div class="bot-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                ${this.formatText(message)}
                ${sourcesHtml}
            </div>
        `;
        
        return messageElement;
    }

    showThinkingAnimation() {
        const thinkingElement = document.createElement('div');
        thinkingElement.className = 'thinking-animation';
        thinkingElement.innerHTML = `
            <div class="bot-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="thinking-content">
                <i class="fas fa-brain" style="color: #4f46e5;"></i>
                <span>Thinking and analyzing your documents...</span>
                <div class="thinking-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        this.chatMessages.appendChild(thinkingElement);
        this.scrollToBottom();
        return thinkingElement;
    }

    removeThinkingAnimation(element) {
        if (element && element.parentNode) {
            element.parentNode.removeChild(element);
        }
    }

    autoResizeTextarea() {
        this.userInput.style.height = 'auto';
        this.userInput.style.height = Math.min(this.userInput.scrollHeight, 120) + 'px';
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }

    formatText(text) {
        // Simple text formatting
        return text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async showSystemInfo() {
        const modal = document.getElementById('info-modal');
        const modalBody = document.getElementById('modal-body');
        
        modal.style.display = 'block';
        modalBody.innerHTML = `
            <div class="loading-spinner">
                <i class="fas fa-spinner fa-spin"></i>
                Loading system information...
            </div>
        `;
        
        try {
            const response = await fetch('/api/system-info');
            const info = await response.json();
            
            if (response.ok) {
                modalBody.innerHTML = `
                    <div class="info-grid">
                        <div class="info-card">
                            <h4>System Status</h4>
                            <p>${info.status}</p>
                        </div>
                        <div class="info-card">
                            <h4>Documents Loaded</h4>
                            <p>${info.documents_loaded}</p>
                        </div>
                        <div class="info-card">
                            <h4>Text Chunks</h4>
                            <p>${info.chunks_created}</p>
                        </div>
                        <div class="info-card">
                            <h4>Vectors Stored</h4>
                            <p>${info.vectors_stored}</p>
                        </div>
                        <div class="info-card">
                            <h4>Embedding Model</h4>
                            <p>${info.embedding_model}</p>
                        </div>
                        <div class="info-card">
                            <h4>Embedding Dimension</h4>
                            <p>${info.embedding_dimension}</p>
                        </div>
                        <div class="info-card">
                            <h4>Data Directory</h4>
                            <p>${info.data_directory}</p>
                        </div>
                        <div class="info-card">
                            <h4>Collection Name</h4>
                            <p>${info.collection_name}</p>
                        </div>
                        <div class="info-card">
                            <h4>Chunk Size</h4>
                            <p>${info.chunk_size} characters</p>
                        </div>
                        <div class="info-card">
                            <h4>Chunk Overlap</h4>
                            <p>${info.chunk_overlap} characters</p>
                        </div>
                    </div>
                `;
            } else {
                modalBody.innerHTML = `
                    <div class="loading-spinner">
                        <i class="fas fa-exclamation-triangle"></i>
                        Error loading system information: ${info.error}
                    </div>
                `;
            }
        } catch (error) {
            modalBody.innerHTML = `
                <div class="loading-spinner">
                    <i class="fas fa-exclamation-triangle"></i>
                    Error connecting to server
                </div>
            `;
        }
    }

    closeModal() {
        document.getElementById('info-modal').style.display = 'none';
    }
}

// Global functions
function setQuery(query) {
    document.getElementById('user-input').value = query;
    document.getElementById('user-input').focus();
}

function showSourcePreview(content, source) {
    alert(`Source: ${source}\n\nContent:\n${content}`);
}

function closeModal() {
    app.closeModal();
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new RAGChatApp();
});
