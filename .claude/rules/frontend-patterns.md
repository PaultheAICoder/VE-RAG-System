# Gradio UI Patterns - AI Ready RAG

## Architecture

- **Framework**: Gradio 5.0+ mounted on FastAPI at `/app`
- **No separate build**: Gradio renders server-side
- **State**: `gr.State` for per-session user data

## Directory Structure

```
ai_ready_rag/ui/
├── gradio_app.py       # Main app factory (create_app)
├── components.py       # Reusable UI helpers
├── document_components.py  # Document management UI
├── api_client.py       # Stateless API client
└── theme.py           # Custom theme and CSS
```

## Key Patterns

### Session State
```python
# Per-user auth state - isolated per browser session
auth_state = gr.State(value={
    "token": None,
    "user": None,
    "is_authenticated": False,
})
```

### API Client (Stateless)
```python
# Token passed per-request, not stored in client
class GradioAPIClient:
    @staticmethod
    def send_message(token: str, session_id: str, content: str):
        with httpx.Client(base_url=BASE_URL) as client:
            response = client.post(
                f"/api/chat/sessions/{session_id}/messages",
                json={"content": content},
                headers={"Authorization": f"Bearer {token}"},
            )
```

### Event Handlers
```python
def handle_action(auth: dict, input_value: str) -> tuple:
    """Handler returns tuple matching outputs."""
    token = auth.get("token")
    if not token:
        return "Not authenticated", gr.update()

    # Do work...
    return result, gr.update(value=new_value)

# Wire up
button.click(
    fn=handle_action,
    inputs=[auth_state, input_box],
    outputs=[status_text, output_component],
)
```

### Loading States
```python
# Chain with .then() for loading indicator
send_btn.click(
    fn=show_loading,
    outputs=[loading_indicator],
).then(
    fn=do_work,
    inputs=[...],
    outputs=[..., loading_indicator],
)
```

### Conditional Visibility
```python
# Admin-only sections
doc_mgmt_accordion = gr.Accordion(
    "Document Management",
    open=False,
    visible=False,  # Set visible=True on admin login
)

# In handler
is_admin = user.get("role") == "admin"
return gr.update(visible=is_admin)
```

## Component Patterns

### DataFrames
```python
document_table = gr.Dataframe(
    headers=["ID", "Filename", "Status"],
    datatype=["str", "str", "str"],
    interactive=False,
    wrap=True,
)

# Return list of lists
return [["id1", "file.pdf", "ready"], ["id2", "doc.txt", "pending"]]
```

### Dropdowns with Choices
```python
tag_dropdown = gr.Dropdown(
    label="Tags",
    multiselect=True,
    choices=[],  # Populated dynamically
)

# Update choices
return gr.update(choices=[("Display", "value"), ...])
```

### File Upload
```python
file_input = gr.File(
    label="Select File",
    file_types=[".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".html", ".csv"],
)

# In handler, file.name gives the path
file_path = file.name if hasattr(file, "name") else str(file)
```

## Error Handling

```python
try:
    result = GradioAPIClient.some_action(token, ...)
    return f"Success: {result}"
except httpx.HTTPStatusError as e:
    if e.response.status_code == 401:
        return "Session expired. Please log in again."
    elif e.response.status_code == 403:
        return "Permission denied."
    else:
        return f"Error: {e.response.text}"
except Exception as e:
    return f"Unexpected error: {e}"
```

## Testing Gradio

Gradio UI is tested via API integration tests, not direct UI tests.
The API client methods can be unit tested separately.
