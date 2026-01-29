"""Document management UI components for Gradio."""

from typing import Any

import gradio as gr

from ai_ready_rag.ui.api_client import GradioAPIClient


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_status_badge(status: str) -> str:
    """Format status as colored text."""
    colors = {
        "pending": "orange",
        "processing": "blue",
        "ready": "green",
        "failed": "red",
    }
    color = colors.get(status, "gray")
    return f"<span style='color: {color}; font-weight: bold;'>{status}</span>"


def create_documents_tab() -> tuple[Any, ...]:
    """Create the Documents tab for admin users.

    Returns:
        Tuple of Gradio components for external event wiring.
    """
    with gr.Column():
        gr.Markdown("## Document Management")

        # Upload section
        with gr.Accordion("Upload Document", open=True):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Select Document",
                        file_types=[
                            ".pdf",
                            ".docx",
                            ".xlsx",
                            ".pptx",
                            ".txt",
                            ".md",
                            ".html",
                            ".csv",
                        ],
                    )
                with gr.Column(scale=1):
                    tag_dropdown = gr.Dropdown(
                        label="Tags (required)",
                        multiselect=True,
                        choices=[],
                        interactive=True,
                    )

            with gr.Row():
                title_input = gr.Textbox(
                    label="Title (optional)",
                    placeholder="Document title...",
                )
                description_input = gr.Textbox(
                    label="Description (optional)",
                    placeholder="Brief description...",
                )

            upload_btn = gr.Button("Upload Document", variant="primary")
            upload_status = gr.Markdown("")

        # Document list section
        gr.Markdown("### Documents")

        with gr.Row():
            status_filter = gr.Dropdown(
                label="Status",
                choices=["All", "pending", "processing", "ready", "failed"],
                value="All",
                scale=1,
            )
            search_input = gr.Textbox(
                label="Search",
                placeholder="Search filename or title...",
                scale=2,
            )
            refresh_btn = gr.Button("Refresh", scale=1)

        # Document table
        document_table = gr.Dataframe(
            headers=["ID", "Filename", "Type", "Size", "Status", "Tags", "Uploaded"],
            datatype=["str", "str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )

        # Action buttons row
        with gr.Row():
            selected_doc_id = gr.Textbox(
                label="Selected Document ID",
                placeholder="Click a row to select...",
                interactive=True,
            )
            delete_btn = gr.Button("Delete", variant="stop")
            reprocess_btn = gr.Button("Reprocess")

        # Tag update section
        with gr.Accordion("Update Tags", open=False):
            update_tag_dropdown = gr.Dropdown(
                label="New Tags",
                multiselect=True,
                choices=[],
                interactive=True,
            )
            update_tags_btn = gr.Button("Update Tags")

        action_status = gr.Markdown("")

    return (
        file_input,
        tag_dropdown,
        title_input,
        description_input,
        upload_btn,
        upload_status,
        status_filter,
        search_input,
        refresh_btn,
        document_table,
        selected_doc_id,
        delete_btn,
        reprocess_btn,
        update_tag_dropdown,
        update_tags_btn,
        action_status,
    )


def load_tags(auth_state: dict[str, Any]) -> list[tuple[str, str]]:
    """Load tags from API for dropdown.

    Returns:
        List of (display_name, id) tuples for dropdown choices.
    """
    token = auth_state.get("token")
    if not token:
        return []

    try:
        tags = GradioAPIClient.get_tags(token)
        return [(t["display_name"], t["id"]) for t in tags]
    except Exception:
        return []


def load_documents(
    auth_state: dict[str, Any],
    status_filter: str = "All",
    search: str = "",
) -> list[list[str]]:
    """Load documents from API for table display.

    Returns:
        List of rows for dataframe.
    """
    token = auth_state.get("token")
    if not token:
        return []

    try:
        params: dict[str, Any] = {"limit": 50, "offset": 0}
        if status_filter and status_filter != "All":
            params["status_filter"] = status_filter
        if search:
            params["search"] = search

        result = GradioAPIClient.list_documents(
            token,
            limit=50,
            status_filter=status_filter if status_filter != "All" else None,
            search=search if search else None,
        )

        rows = []
        for doc in result.get("documents", []):
            # Format tags as comma-separated names
            tags = ", ".join(t["name"] for t in doc.get("tags", []))

            # Format uploaded date
            uploaded_at = doc.get("uploaded_at", "")
            if uploaded_at:
                # Truncate to date only
                uploaded_at = uploaded_at[:10]

            rows.append(
                [
                    doc.get("id", "")[:8] + "...",  # Truncated ID
                    doc.get("original_filename", ""),
                    doc.get("file_type", ""),
                    format_file_size(doc.get("file_size", 0)),
                    doc.get("status", ""),
                    tags,
                    uploaded_at,
                ]
            )

        return rows

    except Exception as e:
        return [[f"Error loading documents: {e}", "", "", "", "", "", ""]]


def handle_upload(
    auth_state: dict[str, Any],
    file: Any,
    tag_ids: list[str],
    title: str,
    description: str,
) -> str:
    """Handle document upload.

    Returns:
        Status message.
    """
    token = auth_state.get("token")
    if not token:
        return "Not authenticated. Please log in."

    if not file:
        return "Please select a file to upload."

    if not tag_ids:
        return "Please select at least one tag."

    try:
        # file is a Gradio file object with .name attribute
        file_path = file.name if hasattr(file, "name") else str(file)

        result = GradioAPIClient.upload_document(
            token=token,
            file_path=file_path,
            tag_ids=tag_ids,
            title=title if title else None,
            description=description if description else None,
        )

        return f"Uploaded successfully: {result.get('original_filename', 'document')} (Status: {result.get('status', 'pending')})"

    except Exception as e:
        return f"Upload failed: {e}"


def handle_delete(auth_state: dict[str, Any], document_id: str) -> str:
    """Handle document deletion.

    Returns:
        Status message.
    """
    token = auth_state.get("token")
    if not token:
        return "Not authenticated. Please log in."

    if not document_id:
        return "Please select a document to delete."

    # Extract full ID if truncated (we stored truncated in table)
    # For now, require user to enter full ID or we need to track separately
    try:
        GradioAPIClient.delete_document(token, document_id)
        return "Document deleted successfully."
    except Exception as e:
        return f"Delete failed: {e}"


def handle_reprocess(auth_state: dict[str, Any], document_id: str) -> str:
    """Handle document reprocessing.

    Returns:
        Status message.
    """
    token = auth_state.get("token")
    if not token:
        return "Not authenticated. Please log in."

    if not document_id:
        return "Please select a document to reprocess."

    try:
        result = GradioAPIClient.reprocess_document(token, document_id)
        return f"Document queued for reprocessing. Status: {result.get('status', 'pending')}"
    except Exception as e:
        return f"Reprocess failed: {e}"


def handle_tag_update(
    auth_state: dict[str, Any],
    document_id: str,
    tag_ids: list[str],
) -> str:
    """Handle document tag update.

    Returns:
        Status message.
    """
    token = auth_state.get("token")
    if not token:
        return "Not authenticated. Please log in."

    if not document_id:
        return "Please select a document."

    if not tag_ids:
        return "Please select at least one tag."

    try:
        result = GradioAPIClient.update_document_tags(token, document_id, tag_ids)
        new_tags = ", ".join(t["name"] for t in result.get("tags", []))
        return f"Tags updated: {new_tags}"
    except Exception as e:
        return f"Tag update failed: {e}"
