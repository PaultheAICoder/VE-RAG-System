"""Gradio theme and custom CSS for AI Ready RAG UI."""

import gradio as gr

# Custom theme based on Gradio's Soft theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
).set(
    # Colors
    body_background_fill="#f9fafb",
    block_background_fill="#ffffff",
    block_border_color="#e5e7eb",
    # Primary button
    button_primary_background_fill="#3b82f6",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
    # Secondary button
    button_secondary_background_fill="#f3f4f6",
    button_secondary_text_color="#374151",
    # Input
    input_background_fill="#ffffff",
    input_border_color="#e5e7eb",
    # Shadows
    block_shadow="0 1px 3px rgba(0,0,0,0.1)",
)

custom_css = """
/* Message bubbles */
.user-message {
    background-color: #3b82f6;
    color: white;
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin-left: auto;
    max-width: 80%;
}

.assistant-message {
    background-color: #f3f4f6;
    color: #111827;
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin-right: auto;
    max-width: 80%;
}

/* Confidence bar */
.confidence-bar {
    height: 0.5rem;
    border-radius: 9999px;
    background-color: #e5e7eb;
    overflow: hidden;
    margin-top: 0.5rem;
}

.confidence-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.3s ease;
}

.confidence-high { background-color: #10b981; }
.confidence-medium { background-color: #f59e0b; }
.confidence-low { background-color: #ef4444; }

/* Routing alert */
.routing-alert {
    background-color: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin-top: 0.5rem;
}

/* Error banner */
.error-banner {
    background-color: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* Loading spinner */
.loading-spinner {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #6b7280;
    padding: 1rem;
}

/* Disabled state for Phase 2 elements */
.phase2-disabled {
    opacity: 0.5;
    cursor: not-allowed;
    pointer-events: none;
}

/* Sidebar session list */
.session-item {
    padding: 0.5rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: background-color 0.15s;
}

.session-item:hover {
    background-color: #e5e7eb;
}

.session-item.active {
    background-color: #dbeafe;
}

/* Citation expandable */
.citation-header {
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.citation-content {
    padding: 0.5rem;
    background-color: #f9fafb;
    border-radius: 0.375rem;
    margin-top: 0.25rem;
    font-size: 0.875rem;
}

/* Login container */
.login-container {
    max-width: 400px;
    margin: 0 auto;
    padding: 2rem;
}

/* Header bar */
.header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background-color: #ffffff;
    border-bottom: 1px solid #e5e7eb;
}

/* Sidebar */
.sidebar {
    width: 250px;
    background-color: #f3f4f6;
    border-right: 1px solid #e5e7eb;
    padding: 1rem;
    height: 100%;
    overflow-y: auto;
}

/* Chat area */
.chat-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    background-color: #ffffff;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
}

.chat-input-area {
    border-top: 1px solid #e5e7eb;
    padding: 1rem;
}

/* Sidebar responsive (future) */
@media (max-width: 768px) {
    .sidebar-collapsible {
        /* TODO: Phase 2 - mobile responsive */
    }
}

/* Status indicators for OCR availability */
.status-available {
    color: #10b981;
    font-weight: 600;
}

.status-unavailable {
    color: #ef4444;
    font-weight: 600;
}

/* Health status badges */
.health-healthy {
    background-color: rgba(16, 185, 129, 0.1);
    color: #10b981;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
}

.health-unhealthy {
    background-color: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
}

/* Architecture info section spacing */
.architecture-section {
    margin-bottom: 1rem;
}

.architecture-label {
    font-weight: 600;
    color: #374151;
}

/* Reindex status card styles (#72) */
.reindex-status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 600;
}

.status-pending { background-color: #e5e7eb; color: #374151; }
.status-running { background-color: #dbeafe; color: #1d4ed8; }
.status-paused { background-color: #fef3c7; color: #92400e; }
.status-completed { background-color: #d1fae5; color: #065f46; }
.status-failed { background-color: #fee2e2; color: #991b1b; }
.status-aborted { background-color: #e5e7eb; color: #374151; }

/* Progress bar for reindex */
.reindex-progress-bar {
    height: 0.75rem;
    border-radius: 9999px;
    background-color: #e5e7eb;
    overflow: hidden;
    margin: 0.5rem 0;
}

.reindex-progress-fill {
    height: 100%;
    border-radius: 9999px;
    background-color: #3b82f6;
    transition: width 0.3s ease;
}

/* Auto-skip badge */
.auto-skip-badge {
    display: inline-block;
    padding: 0.125rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 500;
    background-color: #fef3c7;
    color: #92400e;
    margin-left: 0.5rem;
}

/* Failure list styling */
.failure-item {
    padding: 0.5rem;
    border-bottom: 1px solid #e5e7eb;
}

.failure-item:last-child {
    border-bottom: none;
}

.failure-filename {
    font-weight: 600;
    color: #374151;
}

.failure-error {
    font-size: 0.875rem;
    color: #ef4444;
    margin-top: 0.25rem;
}
"""
