"""Main Gradio Blocks application for AI Ready RAG."""

from datetime import datetime

import gradio as gr
import httpx

from ai_ready_rag.ui.api_client import GradioAPIClient, handle_api_error
from ai_ready_rag.ui.components import parse_assistant_response
from ai_ready_rag.ui.document_components import (
    get_document_choices,
    handle_bulk_delete,
    handle_delete,
    handle_reprocess,
    handle_upload,
    load_documents,
    load_tags,
)
from ai_ready_rag.ui.theme import custom_css, theme


def _format_date(iso_string: str | None) -> str:
    """Format ISO date string for display."""
    if not iso_string:
        return ""
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.strftime("%m/%d %H:%M")
    except Exception:
        return iso_string[:10] if iso_string else ""


def _format_sessions_for_display(sessions: list[dict]) -> list[list[str]]:
    """Format sessions list for Dataframe display."""
    return [
        [
            s.get("title") or f"Chat {s.get('id', '')[:8]}...",
            _format_date(s.get("updated_at")),
        ]
        for s in sessions
    ]


def _format_messages_for_chatbot(messages: list[dict]) -> list[dict]:
    """Format messages list for Gradio Chatbot display.

    Returns messages in Gradio 5.0 format: list of dicts with 'role' and 'content'.
    Assistant messages are formatted with citations, confidence, and routing info.
    """
    chat_history = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            chat_history.append(
                {
                    "role": "user",
                    "content": msg.get("content", ""),
                }
            )
        elif role == "assistant":
            chat_history.append(
                {
                    "role": "assistant",
                    "content": parse_assistant_response(msg),
                }
            )
    return chat_history


def create_app() -> gr.Blocks:
    """Create and return the Gradio app instance.

    The app uses gr.State for per-session authentication state,
    ensuring user isolation in multi-user environments.
    """
    with gr.Blocks(theme=theme, css=custom_css, title="AI Ready RAG") as app:
        # Per-user auth state - isolated per browser session
        auth_state = gr.State(
            value={
                "token": None,
                "user": None,
                "is_authenticated": False,
            }
        )

        # Session state for chat
        session_state = gr.State(
            value={
                "sessions": [],
                "active_session_id": None,
                "has_more_sessions": False,
                "sessions_offset": 0,
                "messages": [],
            }
        )

        # ========== LOGIN CONTAINER ==========
        with gr.Column(visible=True, elem_classes=["login-container"]) as login_container:
            gr.Markdown("# AI Ready RAG")
            gr.Markdown("Sign in to access your documents and chat.")

            with gr.Column():
                login_email = gr.Textbox(
                    label="Email",
                    placeholder="you@example.com",
                    type="email",
                )
                login_password = gr.Textbox(
                    label="Password",
                    placeholder="Enter your password",
                    type="password",
                )
                login_btn = gr.Button("Login", variant="primary")
                login_error = gr.Markdown(visible=False, elem_classes=["error-banner"])

        # ========== MAIN APP CONTAINER ==========
        with gr.Column(visible=False) as main_container:
            # Header bar
            with gr.Row(elem_classes=["header-bar"]):
                gr.Markdown("**AI Ready RAG**")
                with gr.Row():
                    user_display = gr.Markdown("")
                    logout_btn = gr.Button("Logout", variant="secondary", size="sm")

            # Main content area with sidebar and chat
            with gr.Row():
                # Sidebar
                with gr.Column(scale=1, min_width=250, elem_classes=["sidebar"]):
                    start_new_btn = gr.Button("New", variant="secondary", size="sm")

                    gr.Markdown("### Recent Chats")
                    sessions_list = gr.Dataframe(
                        headers=["Title", "Updated"],
                        datatype=["str", "str"],
                        column_count=(2, "fixed"),
                        interactive=False,
                    )
                    load_more_btn = gr.Button(
                        "Load More",
                        variant="secondary",
                        size="sm",
                        visible=False,
                    )

                    # Admin sections container (visibility controlled on login)
                    with gr.Column(visible=False) as admin_sections:
                        gr.Markdown("---")
                        with gr.Accordion(
                            "Document Management",
                            open=False,
                        ):
                            # Upload section
                            doc_file_input = gr.File(
                                label="Select Files",
                                file_types=[
                                    ".pdf",
                                    ".docx",
                                    ".txt",
                                    ".md",
                                    ".csv",
                                    ".xlsx",
                                    ".pptx",
                                    ".html",
                                    ".htm",
                                ],
                                file_count="multiple",
                            )
                            doc_tag_dropdown = gr.Dropdown(
                                label="Tags (required)",
                                multiselect=True,
                                choices=[],
                            )
                            doc_title_input = gr.Textbox(
                                label="Title (optional)",
                                placeholder="Document title...",
                            )
                            doc_upload_btn = gr.Button("Upload", variant="primary", size="sm")
                            doc_upload_status = gr.Markdown("")

                            gr.Markdown("---")

                            # Document list controls
                            doc_refresh_btn = gr.Button("Refresh List", size="sm")
                            doc_table = gr.Dataframe(
                                headers=["ID", "Filename", "Status"],
                                datatype=["str", "str", "str"],
                                interactive=False,
                            )

                            gr.Markdown("---")
                            gr.Markdown("**Select Documents**")

                            # Multi-select checkboxes for documents
                            doc_checkboxes = gr.CheckboxGroup(
                                label="Select documents to delete",
                                choices=[],
                                value=[],
                            )

                            # Selection controls
                            with gr.Row():
                                doc_select_all_btn = gr.Button("Select All", size="sm")
                                doc_clear_selection_btn = gr.Button("Clear Selection", size="sm")

                            # Bulk action buttons
                            with gr.Row():
                                doc_delete_selected_btn = gr.Button(
                                    "Delete Selected", variant="stop", size="sm"
                                )
                                doc_delete_all_btn = gr.Button(
                                    "Delete All", variant="stop", size="sm"
                                )

                            doc_action_status = gr.Markdown("")

                            gr.Markdown("---")
                            gr.Markdown("**Single Document Actions**")

                            # Single document action section (legacy)
                            doc_selected_id = gr.Textbox(
                                label="Document ID (for single actions)",
                                placeholder="Enter full document ID...",
                            )
                            with gr.Row():
                                doc_delete_btn = gr.Button("Delete", size="sm")
                                doc_reprocess_btn = gr.Button("Reprocess", size="sm")

                        # Admin System Architecture Section
                        with gr.Accordion(
                            "Model Architecture",
                            open=False,
                        ):
                            arch_refresh_btn = gr.Button("Refresh Architecture Info", size="sm")
                            arch_status = gr.Markdown(
                                "Click 'Refresh' to load architecture information."
                            )

                            # Document Parsing subsection
                            gr.Markdown("### Document Parsing")
                            arch_doc_parsing = gr.Markdown("")

                            # Embeddings subsection
                            gr.Markdown("### Embeddings")
                            arch_embeddings = gr.Markdown("")

                            # Chat Model subsection
                            gr.Markdown("### Chat Model")
                            arch_chat_model = gr.Markdown("")

                            # Infrastructure subsection
                            gr.Markdown("### Infrastructure")
                            arch_infrastructure = gr.Markdown("")

                        # Admin OCR Status Section
                        with gr.Accordion(
                            "OCR Status",
                            open=False,
                        ):
                            ocr_refresh_btn = gr.Button("Refresh OCR Status", size="sm")

                            # OCR status display
                            ocr_tesseract = gr.Markdown("")
                            ocr_easyocr = gr.Markdown("")

                        # Processing Options Section (#12)
                        with gr.Accordion(
                            "Processing Options",
                            open=False,
                        ):
                            proc_refresh_btn = gr.Button("Load Settings", size="sm")
                            proc_status = gr.Markdown("")

                            proc_enable_ocr = gr.Checkbox(
                                label="Enable OCR",
                                value=True,
                            )
                            proc_force_ocr = gr.Checkbox(
                                label="Force Full Page OCR",
                                value=False,
                            )
                            proc_ocr_lang = gr.Dropdown(
                                label="OCR Language",
                                choices=["English", "Spanish", "French", "German", "Chinese"],
                                value="English",
                            )
                            proc_table_mode = gr.Radio(
                                label="Table Extraction Mode",
                                choices=["accurate", "fast"],
                                value="accurate",
                            )
                            proc_image_desc = gr.Checkbox(
                                label="Include Image Descriptions",
                                value=False,
                            )
                            proc_query_routing = gr.Radio(
                                label="Query Routing Mode",
                                choices=["retrieve_only", "retrieve_and_direct"],
                                value="retrieve_only",
                                info="retrieve_only: Always search documents. retrieve_and_direct: Allow direct LLM responses for general questions.",
                            )
                            proc_save_btn = gr.Button("Save Settings", variant="primary", size="sm")

                        # Knowledge Base Statistics Section (#13)
                        with gr.Accordion(
                            "Knowledge Base Statistics",
                            open=False,
                        ):
                            kb_refresh_btn = gr.Button("Refresh Stats", size="sm")
                            kb_status = gr.Markdown("")

                            kb_stats_display = gr.Markdown("Click 'Refresh Stats' to load.")

                            with gr.Accordion("Indexed Files", open=False):
                                kb_file_list = gr.Markdown("No files loaded.")

                            kb_clear_btn = gr.Button(
                                "Clear Knowledge Base",
                                variant="stop",
                                size="sm",
                            )

                        # Model Configuration Section (#14, #21)
                        with gr.Accordion(
                            "Model Configuration",
                            open=False,
                        ):
                            model_refresh_btn = gr.Button("Load Models", size="sm")
                            model_status = gr.Markdown("")

                            # Embedding Model Section
                            gr.Markdown("### Embedding Model (Vectorization)")
                            embed_current = gr.Markdown("**Current**: Not loaded")
                            embed_dropdown = gr.Dropdown(
                                label="Select Embedding Model",
                                choices=[],
                                interactive=True,
                            )
                            embed_warning = gr.Markdown(
                                "⚠️ **Warning**: Changing the embedding model invalidates all existing vectors. "
                                "Documents must be re-indexed.",
                                visible=False,
                            )
                            embed_apply_btn = gr.Button(
                                "Apply Embedding Change",
                                variant="stop",
                                size="sm",
                            )

                            gr.Markdown("---")

                            # Chat Model Section
                            gr.Markdown("### Chat Model (RAG)")
                            model_current = gr.Markdown("**Current**: Not loaded")
                            model_dropdown = gr.Dropdown(
                                label="Select Chat Model",
                                choices=[],
                                interactive=True,
                            )
                            model_apply_btn = gr.Button(
                                "Apply Chat Model Change",
                                variant="primary",
                                size="sm",
                            )

                # Chat area
                with gr.Column(scale=3, elem_classes=["chat-area"]):
                    chat_display = gr.Chatbot(
                        label="Chat",
                        height=400,
                        show_label=False,
                        elem_classes=["chat-messages"],
                    )

                    # Loading indicator (hidden by default)
                    loading_indicator = gr.Markdown(
                        "Thinking...",
                        visible=False,
                        elem_classes=["loading-spinner"],
                    )

                    # Error banner (hidden by default)
                    chat_error = gr.Markdown(
                        visible=False,
                        elem_classes=["error-banner"],
                    )

                    # Input area
                    with gr.Row(elem_classes=["chat-input-area"]):
                        msg_input = gr.Textbox(
                            placeholder="Type your message...",
                            show_label=False,
                            scale=4,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

        # ========== EVENT HANDLERS ==========

        def handle_login(email: str, password: str, auth: dict, sess: dict) -> tuple:
            """Handle login form submission and load sessions."""
            if not email or not password:
                return (
                    auth,
                    sess,
                    gr.update(visible=True, value="Please enter email and password."),
                    gr.update(visible=True),  # login_container
                    gr.update(visible=False),  # main_container
                    "",  # user_display
                    gr.update(value=[]),  # sessions_list
                    gr.update(visible=False),  # load_more_btn
                    gr.update(visible=False),  # admin_sections
                    gr.update(choices=[]),  # doc_tag_dropdown
                )

            try:
                # Authenticate
                result = GradioAPIClient.login(email, password)
                token = result.get("access_token")
                user = result.get("user", {})

                new_auth = {
                    "token": token,
                    "user": user,
                    "is_authenticated": True,
                }

                # Load initial sessions
                sessions_response = GradioAPIClient.get_sessions(token, limit=20, offset=0)
                sessions = sessions_response.get("sessions", [])
                total = sessions_response.get("total", 0)
                has_more = len(sessions) < total

                new_sess = {
                    "sessions": sessions,
                    "active_session_id": None,
                    "has_more_sessions": has_more,
                    "sessions_offset": 20,
                    "messages": [],
                }

                user_name = user.get("display_name", email)
                sessions_display = _format_sessions_for_display(sessions)

                # Check if admin for document management visibility
                is_admin = user.get("role") == "admin"

                # Load tags for admin document dropdown
                tag_choices = []
                if is_admin:
                    tag_choices = load_tags(new_auth)

                return (
                    new_auth,
                    new_sess,
                    gr.update(visible=False),  # login_error
                    gr.update(visible=False),  # login_container
                    gr.update(visible=True),  # main_container
                    f"**{user_name}**",  # user_display
                    gr.update(value=sessions_display),  # sessions_list
                    gr.update(visible=has_more),  # load_more_btn
                    gr.update(visible=is_admin),  # admin_sections
                    gr.update(choices=tag_choices),  # doc_tag_dropdown
                )
            except httpx.HTTPStatusError as e:
                # For login, 401 means invalid credentials, not session expired
                if e.response.status_code == 401:
                    error_msg = "Invalid email or password."
                else:
                    error_info = handle_api_error(e)
                    error_msg = error_info["message"]
                return (
                    auth,
                    sess,
                    gr.update(visible=True, value=f"**Error:** {error_msg}"),
                    gr.update(visible=True),  # login_container
                    gr.update(visible=False),  # main_container
                    "",  # user_display
                    gr.update(value=[]),  # sessions_list
                    gr.update(visible=False),  # load_more_btn
                    gr.update(visible=False),  # admin_sections
                    gr.update(choices=[]),  # doc_tag_dropdown
                )
            except Exception as e:
                return (
                    auth,
                    sess,
                    gr.update(visible=True, value=f"**Error:** {e!s}"),
                    gr.update(visible=True),  # login_container
                    gr.update(visible=False),  # main_container
                    "",  # user_display
                    gr.update(value=[]),  # sessions_list
                    gr.update(visible=False),  # load_more_btn
                    gr.update(visible=False),  # admin_sections
                    gr.update(choices=[]),  # doc_tag_dropdown
                )

        def handle_logout(auth: dict, sess: dict) -> tuple:
            """Handle logout button click."""
            token = auth.get("token")
            if token:
                try:
                    GradioAPIClient.logout(token)
                except Exception:
                    pass  # Ignore logout errors

            # Reset states
            new_auth = {
                "token": None,
                "user": None,
                "is_authenticated": False,
            }
            new_sess = {
                "sessions": [],
                "active_session_id": None,
                "has_more_sessions": False,
                "sessions_offset": 0,
                "messages": [],
            }

            return (
                new_auth,
                new_sess,
                gr.update(visible=True),  # login_container
                gr.update(visible=False),  # main_container
                "",  # user_display
                gr.update(visible=False),  # login_error
                gr.update(value=[]),  # sessions_list
                gr.update(visible=False),  # load_more_btn
                [],  # chat_display
                gr.update(visible=False),  # admin_sections
                gr.update(choices=[]),  # doc_tag_dropdown
            )

        def handle_start_new(auth: dict, sess: dict) -> tuple:
            """Clear active session to allow starting fresh conversation."""
            if not auth.get("token"):
                return sess, [], gr.update(visible=False)

            new_sess = {
                **sess,
                "active_session_id": None,
                "messages": [],
            }
            return new_sess, [], gr.update(visible=False)

        def handle_load_more(auth: dict, sess: dict) -> tuple:
            """Load more sessions (pagination)."""
            token = auth.get("token")
            if not token:
                return sess, gr.update(), gr.update(visible=False)

            try:
                offset = sess.get("sessions_offset", 0)
                response = GradioAPIClient.get_sessions(token, limit=20, offset=offset)

                new_sessions = response.get("sessions", [])
                total = response.get("total", 0)

                # Append to existing sessions
                all_sessions = sess.get("sessions", []) + new_sessions
                new_offset = offset + len(new_sessions)
                has_more = new_offset < total

                new_sess = {
                    **sess,
                    "sessions": all_sessions,
                    "sessions_offset": new_offset,
                    "has_more_sessions": has_more,
                }

                sessions_display = _format_sessions_for_display(all_sessions)

                return (
                    new_sess,
                    gr.update(value=sessions_display),
                    gr.update(visible=has_more),
                )
            except Exception:
                return sess, gr.update(), gr.update(visible=False)

        def handle_session_select(evt: gr.SelectData, auth: dict, sess: dict) -> tuple:
            """Handle clicking on a session in the list."""
            token = auth.get("token")
            sessions = sess.get("sessions", [])

            if not token or evt.index[0] >= len(sessions):
                return sess, [], gr.update(visible=False)

            selected_session = sessions[evt.index[0]]
            session_id = selected_session.get("id")

            try:
                # Load messages for this session
                response = GradioAPIClient.get_messages(token, session_id)
                messages = response.get("messages", [])

                # Format for Chatbot - list of message dicts with role/content
                chat_history = _format_messages_for_chatbot(messages)

                new_sess = {
                    **sess,
                    "active_session_id": session_id,
                    "messages": messages,
                }

                return new_sess, chat_history, gr.update(visible=False)
            except httpx.HTTPStatusError as e:
                error_info = handle_api_error(e)
                return (
                    sess,
                    [],
                    gr.update(visible=True, value=f"**Error:** {error_info['message']}"),
                )
            except Exception as e:
                return sess, [], gr.update(visible=True, value=f"**Error:** {e!s}")

        def send_message(message: str, history: list, auth: dict, sess: dict) -> tuple:
            """Send message to RAG and get response.

            Auto-creates a session when sending the first message with no active session.
            """
            if not message or not message.strip():
                return (
                    "",
                    history,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    sess,
                    gr.update(),  # sessions_list unchanged
                )

            token = auth.get("token")
            if not token:
                return (
                    message,
                    history,
                    gr.update(visible=False),
                    gr.update(
                        visible=True,
                        value="**Session expired.** Please log out and log in again.",
                    ),
                    sess,
                    gr.update(),  # sessions_list unchanged
                )

            # Get or auto-create session
            session_id = sess.get("active_session_id")
            sessions_update = gr.update()  # Default: no change

            if not session_id:
                # Auto-create session with first message as title (truncated)
                try:
                    title = message[:30] + "..." if len(message) > 30 else message
                    new_session = GradioAPIClient.create_session(token, title=title)
                    session_id = new_session.get("id")
                    sessions = [new_session] + sess.get("sessions", [])
                    sess = {
                        **sess,
                        "sessions": sessions,
                        "active_session_id": session_id,
                    }
                    sessions_update = gr.update(value=_format_sessions_for_display(sessions))
                except httpx.HTTPStatusError as e:
                    error_info = handle_api_error(e)
                    return (
                        message,
                        history,
                        gr.update(visible=False),
                        gr.update(visible=True, value=f"**Error:** {error_info['message']}"),
                        sess,
                        gr.update(),  # sessions_list unchanged
                    )
                except Exception as e:
                    return (
                        message,
                        history,
                        gr.update(visible=False),
                        gr.update(visible=True, value=f"**Error creating session:** {e!s}"),
                        sess,
                        gr.update(),  # sessions_list unchanged
                    )

            # Add user message to history immediately
            history = history or []
            history.append({"role": "user", "content": message})

            try:
                # Call the RAG API
                response = GradioAPIClient.send_message(token, session_id, message)

                # Format assistant response with citations, confidence, routing
                assistant_msg = response.get("assistant_message", {})
                formatted_response = parse_assistant_response(assistant_msg)

                # Add assistant response to history
                history.append({"role": "assistant", "content": formatted_response})

                # Update session state with new messages
                new_sess = {
                    **sess,
                    "messages": sess.get("messages", [])
                    + [
                        response.get("user_message", {}),
                        assistant_msg,
                    ],
                }

                return (
                    "",  # Clear input
                    history,
                    gr.update(visible=False),  # Hide loading
                    gr.update(visible=False),  # Hide error
                    new_sess,
                    sessions_update,  # Update sessions list if created
                )
            except httpx.HTTPStatusError as e:
                error_info = handle_api_error(e)

                # Remove the pending user message on error
                if history and history[-1].get("role") == "user":
                    history = history[:-1]

                # Handle 401 specially
                if error_info.get("action") == "redirect_to_login":
                    return (
                        message,
                        history,
                        gr.update(visible=False),
                        gr.update(
                            visible=True,
                            value="**Session expired.** Please log out and log in again.",
                        ),
                        sess,
                        sessions_update,  # Preserve any session created before error
                    )

                return (
                    message,  # Keep message for retry
                    history,
                    gr.update(visible=False),
                    gr.update(visible=True, value=f"**Error:** {error_info['message']}"),
                    sess,
                    sessions_update,  # Preserve any session created before error
                )
            except Exception as e:
                # Remove the pending user message on error
                if history and history[-1].get("role") == "user":
                    history = history[:-1]

                return (
                    message,  # Keep message for retry
                    history,
                    gr.update(visible=False),
                    gr.update(visible=True, value=f"**Error:** {e!s}"),
                    sess,
                    sessions_update,  # Preserve any session created before error
                )

        def show_loading() -> gr.update:
            """Show loading indicator."""
            return gr.update(visible=True, value="Thinking...")

        def hide_loading() -> gr.update:
            """Hide loading indicator."""
            return gr.update(visible=False)

        # ========== CONNECT EVENT HANDLERS ==========

        # Login
        login_btn.click(
            fn=handle_login,
            inputs=[login_email, login_password, auth_state, session_state],
            outputs=[
                auth_state,
                session_state,
                login_error,
                login_container,
                main_container,
                user_display,
                sessions_list,
                load_more_btn,
                admin_sections,
                doc_tag_dropdown,
            ],
        )

        login_password.submit(
            fn=handle_login,
            inputs=[login_email, login_password, auth_state, session_state],
            outputs=[
                auth_state,
                session_state,
                login_error,
                login_container,
                main_container,
                user_display,
                sessions_list,
                load_more_btn,
                admin_sections,
                doc_tag_dropdown,
            ],
        )

        # Logout
        logout_btn.click(
            fn=handle_logout,
            inputs=[auth_state, session_state],
            outputs=[
                auth_state,
                session_state,
                login_container,
                main_container,
                user_display,
                login_error,
                sessions_list,
                load_more_btn,
                chat_display,
                admin_sections,
                doc_tag_dropdown,
            ],
        )

        # New Chat
        start_new_btn.click(
            fn=handle_start_new,
            inputs=[auth_state, session_state],
            outputs=[session_state, chat_display, chat_error],
        )

        # Load More Sessions
        load_more_btn.click(
            fn=handle_load_more,
            inputs=[auth_state, session_state],
            outputs=[session_state, sessions_list, load_more_btn],
        )

        # Session Selection
        sessions_list.select(
            fn=handle_session_select,
            inputs=[auth_state, session_state],
            outputs=[session_state, chat_display, chat_error],
        )

        # Send Message - with loading indicator
        send_btn.click(
            fn=show_loading,
            outputs=[loading_indicator],
        ).then(
            fn=send_message,
            inputs=[msg_input, chat_display, auth_state, session_state],
            outputs=[
                msg_input,
                chat_display,
                loading_indicator,
                chat_error,
                session_state,
                sessions_list,
            ],
        )

        msg_input.submit(
            fn=show_loading,
            outputs=[loading_indicator],
        ).then(
            fn=send_message,
            inputs=[msg_input, chat_display, auth_state, session_state],
            outputs=[
                msg_input,
                chat_display,
                loading_indicator,
                chat_error,
                session_state,
                sessions_list,
            ],
        )

        # ========== DOCUMENT MANAGEMENT EVENT HANDLERS ==========

        def refresh_doc_list(auth: dict) -> tuple:
            """Refresh document list and checkbox choices."""
            docs = load_documents(auth)
            choices = get_document_choices(auth)
            return docs, gr.update(choices=choices, value=[])

        def do_upload(auth: dict, files, tag_ids: list, title: str) -> tuple:
            """Handle document upload for single or multiple files."""
            # Handle None or empty
            if files is None:
                docs = load_documents(auth)
                choices = get_document_choices(auth)
                return (
                    "No files selected.",
                    docs,
                    None,
                    gr.update(value=[]),
                    "",
                    gr.update(choices=choices, value=[]),
                )

            # Normalize to list (gr.File with file_count="multiple" returns list)
            if not isinstance(files, list):
                files = [files]

            if len(files) == 0:
                docs = load_documents(auth)
                choices = get_document_choices(auth)
                return (
                    "No files selected.",
                    docs,
                    None,
                    gr.update(value=[]),
                    "",
                    gr.update(choices=choices, value=[]),
                )

            # Process each file and collect results
            results = []
            success_count = 0
            fail_count = 0

            for file in files:
                status = handle_upload(auth, file, tag_ids, title, "")
                results.append(status)
                # Check if upload succeeded (status contains "Uploaded:" or "Ready")
                if "Uploaded:" in status or "Ready" in status:
                    success_count += 1
                else:
                    fail_count += 1

            # Build summary status
            if len(files) == 1:
                final_status = results[0]
            else:
                summary = (
                    f"**Batch Upload Complete**: {success_count} succeeded, {fail_count} failed\n\n"
                )
                final_status = summary + "\n".join(f"- {r}" for r in results)

            docs = load_documents(auth)
            choices = get_document_choices(auth)
            return (
                final_status,
                docs,
                None,
                gr.update(value=[]),
                "",
                gr.update(choices=choices, value=[]),
            )

        def do_delete(auth: dict, doc_id: str) -> tuple:
            """Handle document deletion."""
            status = handle_delete(auth, doc_id)
            docs = load_documents(auth)
            choices = get_document_choices(auth)
            return status, docs, gr.update(choices=choices, value=[])

        def do_reprocess(auth: dict, doc_id: str) -> tuple:
            """Handle document reprocessing."""
            status = handle_reprocess(auth, doc_id)
            docs = load_documents(auth)
            return status, docs

        def do_select_all(auth: dict) -> gr.update:
            """Select all documents in checkbox group."""
            choices = get_document_choices(auth)
            all_ids = [doc_id for _, doc_id in choices]
            return gr.update(value=all_ids)

        def do_clear_selection() -> gr.update:
            """Clear all selections."""
            return gr.update(value=[])

        def do_delete_selected(auth: dict, selected_ids: list) -> tuple:
            """Delete selected documents."""
            if not selected_ids:
                return "No documents selected.", gr.update(), gr.update()
            status = handle_bulk_delete(auth, selected_ids)
            docs = load_documents(auth)
            choices = get_document_choices(auth)
            return status, docs, gr.update(choices=choices, value=[])

        def do_delete_all(auth: dict) -> tuple:
            """Delete all documents."""
            choices = get_document_choices(auth)
            if not choices:
                return "No documents to delete.", gr.update(), gr.update()
            all_ids = [doc_id for _, doc_id in choices]
            status = handle_bulk_delete(auth, all_ids)
            docs = load_documents(auth)
            new_choices = get_document_choices(auth)
            return status, docs, gr.update(choices=new_choices, value=[])

        # ========== ARCHITECTURE INFO EVENT HANDLERS ==========

        def load_architecture_info(auth: dict) -> tuple:
            """Load architecture information from API."""
            token = auth.get("token")
            if not token:
                return (
                    "Not authenticated.",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                )

            try:
                data = GradioAPIClient.get_architecture_info(token)

                # Format Document Parsing
                doc = data.get("document_parsing", {})
                doc_parsing_md = f"""
**Engine**: {doc.get("engine", "Unknown")} ({doc.get("type", "")})
**Version**: {doc.get("version", "Unknown")}
**Capabilities**:
{chr(10).join(f"- {cap}" for cap in doc.get("capabilities", []))}
"""

                # Format Embeddings
                emb = data.get("embeddings", {})
                embeddings_md = f"""
**Model**: {emb.get("model", "Unknown")}
**Dimensions**: {emb.get("dimensions", "Unknown")}
**Vector Store**: {emb.get("vector_store", "Unknown")}
**URL**: {emb.get("vector_store_url", "Unknown")}
"""

                # Format Chat Model
                chat = data.get("chat_model", {})
                chat_model_md = f"""
**Model**: {chat.get("name", "Unknown")}
**Provider**: {chat.get("provider", "Unknown")}
**Capabilities**:
{chr(10).join(f"- {cap}" for cap in chat.get("capabilities", []))}
"""

                # Format Infrastructure
                infra = data.get("infrastructure", {})
                ollama_status = infra.get("ollama_status", "unknown")
                vector_status = infra.get("vector_db_status", "unknown")

                ollama_badge = "healthy" if ollama_status == "healthy" else "unhealthy"
                vector_badge = "healthy" if vector_status == "healthy" else "unhealthy"

                infrastructure_md = f"""
**Ollama URL**: {infra.get("ollama_url", "Unknown")}
**Ollama Status**: <span class="health-{ollama_badge}">{ollama_status}</span>
**Vector DB Status**: <span class="health-{vector_badge}">{vector_status}</span>
**Profile**: {data.get("profile", "Unknown")}
"""

                # Format OCR Status
                ocr = data.get("ocr_status", {})
                tess = ocr.get("tesseract", {})
                easy = ocr.get("easyocr", {})

                tess_available = tess.get("available", False)
                tess_version = tess.get("version", "N/A")
                tess_status_class = "available" if tess_available else "unavailable"
                tess_status_text = "Available" if tess_available else "Not Available"
                tesseract_md = f"""
**Tesseract OCR**: <span class="status-{tess_status_class}">{tess_status_text}</span>
**Version**: {tess_version if tess_available else "N/A"}
"""

                easy_available = easy.get("available", False)
                easy_version = easy.get("version", "N/A")
                easy_status_class = "available" if easy_available else "unavailable"
                easy_status_text = "Available" if easy_available else "Not Available"
                easyocr_md = f"""
**EasyOCR**: <span class="status-{easy_status_class}">{easy_status_text}</span>
**Version**: {easy_version if easy_available else "N/A"}
"""

                return (
                    "Architecture info loaded successfully.",
                    doc_parsing_md,
                    embeddings_md,
                    chat_model_md,
                    infrastructure_md,
                    tesseract_md,
                    easyocr_md,
                )

            except httpx.HTTPStatusError as e:
                error_msg = f"Failed to load architecture info (HTTP {e.response.status_code})"
                return (error_msg, "", "", "", "", "", "")
            except Exception as e:
                error_msg = f"Error: {e}"
                return (error_msg, "", "", "", "", "", "")

        # Document refresh
        doc_refresh_btn.click(
            fn=refresh_doc_list,
            inputs=[auth_state],
            outputs=[doc_table, doc_checkboxes],
        )

        # Document upload
        doc_upload_btn.click(
            fn=do_upload,
            inputs=[auth_state, doc_file_input, doc_tag_dropdown, doc_title_input],
            outputs=[
                doc_upload_status,
                doc_table,
                doc_file_input,
                doc_tag_dropdown,
                doc_title_input,
                doc_checkboxes,
            ],
        )

        # Document delete (single)
        doc_delete_btn.click(
            fn=do_delete,
            inputs=[auth_state, doc_selected_id],
            outputs=[doc_action_status, doc_table, doc_checkboxes],
        )

        # Document reprocess
        doc_reprocess_btn.click(
            fn=do_reprocess,
            inputs=[auth_state, doc_selected_id],
            outputs=[doc_action_status, doc_table],
        )

        # Bulk delete - Select All
        doc_select_all_btn.click(
            fn=do_select_all,
            inputs=[auth_state],
            outputs=[doc_checkboxes],
        )

        # Bulk delete - Clear Selection
        doc_clear_selection_btn.click(
            fn=do_clear_selection,
            inputs=[],
            outputs=[doc_checkboxes],
        )

        # Bulk delete - Delete Selected
        doc_delete_selected_btn.click(
            fn=do_delete_selected,
            inputs=[auth_state, doc_checkboxes],
            outputs=[doc_action_status, doc_table, doc_checkboxes],
        )

        # Bulk delete - Delete All
        doc_delete_all_btn.click(
            fn=do_delete_all,
            inputs=[auth_state],
            outputs=[doc_action_status, doc_table, doc_checkboxes],
        )

        # ========== ARCHITECTURE INFO EVENT HANDLERS ==========

        # Architecture refresh button - loads both architecture and OCR sections
        arch_refresh_btn.click(
            fn=load_architecture_info,
            inputs=[auth_state],
            outputs=[
                arch_status,
                arch_doc_parsing,
                arch_embeddings,
                arch_chat_model,
                arch_infrastructure,
                ocr_tesseract,
                ocr_easyocr,
            ],
        )

        # OCR refresh button - same handler, same endpoint
        ocr_refresh_btn.click(
            fn=load_architecture_info,
            inputs=[auth_state],
            outputs=[
                arch_status,
                arch_doc_parsing,
                arch_embeddings,
                arch_chat_model,
                arch_infrastructure,
                ocr_tesseract,
                ocr_easyocr,
            ],
        )

        # ========== PROCESSING OPTIONS EVENT HANDLERS (#12) ==========

        def load_processing_options(auth: dict) -> tuple:
            """Load processing options from API."""
            token = auth.get("token")
            if not token:
                return (
                    "Not authenticated.",
                    False,
                    False,
                    "English",
                    "accurate",
                    False,
                    "retrieve_only",
                )

            try:
                data = GradioAPIClient.get_processing_options(token)
                return (
                    "Settings loaded.",
                    data.get("enable_ocr", True),
                    data.get("force_full_page_ocr", False),
                    data.get("ocr_language", "English"),
                    data.get("table_extraction_mode", "accurate"),
                    data.get("include_image_descriptions", False),
                    data.get("query_routing_mode", "retrieve_only"),
                )
            except httpx.HTTPStatusError as e:
                return (
                    f"Failed to load (HTTP {e.response.status_code})",
                    False,
                    False,
                    "English",
                    "accurate",
                    False,
                    "retrieve_only",
                )
            except Exception as e:
                return (f"Error: {e}", False, False, "English", "accurate", False, "retrieve_only")

        def save_processing_options(
            auth: dict,
            enable_ocr: bool,
            force_ocr: bool,
            ocr_lang: str,
            table_mode: str,
            image_desc: bool,
            query_routing: str,
        ) -> str:
            """Save processing options to API."""
            token = auth.get("token")
            if not token:
                return "Not authenticated."

            try:
                options = {
                    "enable_ocr": enable_ocr,
                    "force_full_page_ocr": force_ocr,
                    "ocr_language": ocr_lang,
                    "table_extraction_mode": table_mode,
                    "include_image_descriptions": image_desc,
                    "query_routing_mode": query_routing,
                }
                GradioAPIClient.update_processing_options(token, options)
                return "Settings saved successfully."
            except httpx.HTTPStatusError as e:
                return f"Failed to save (HTTP {e.response.status_code})"
            except Exception as e:
                return f"Error: {e}"

        proc_refresh_btn.click(
            fn=load_processing_options,
            inputs=[auth_state],
            outputs=[
                proc_status,
                proc_enable_ocr,
                proc_force_ocr,
                proc_ocr_lang,
                proc_table_mode,
                proc_image_desc,
                proc_query_routing,
            ],
        )

        proc_save_btn.click(
            fn=save_processing_options,
            inputs=[
                auth_state,
                proc_enable_ocr,
                proc_force_ocr,
                proc_ocr_lang,
                proc_table_mode,
                proc_image_desc,
                proc_query_routing,
            ],
            outputs=[proc_status],
        )

        # ========== KNOWLEDGE BASE STATISTICS EVENT HANDLERS (#13) ==========

        def load_kb_stats(auth: dict) -> tuple:
            """Load knowledge base statistics from API."""
            token = auth.get("token")
            if not token:
                return ("Not authenticated.", "No data.", "No files.")

            try:
                data = GradioAPIClient.get_kb_stats(token)
                total_chunks = data.get("total_chunks", 0)
                unique_files = data.get("unique_files", 0)
                files = data.get("files", [])

                stats_md = f"""
**Total Chunks**: {total_chunks}
**Unique Files**: {unique_files}
"""

                if files:
                    # Show first 10 files (filename only), indicate if more
                    # Files are FileStats dicts with 'filename' key
                    file_list = files[:10]
                    files_md = "\n".join(f"- {f.get('filename', 'Unknown')}" for f in file_list)
                    if len(files) > 10:
                        files_md += f"\n\n*...and {len(files) - 10} more files*"
                else:
                    files_md = "*No files indexed.*"

                return ("Stats loaded.", stats_md, files_md)
            except httpx.HTTPStatusError as e:
                return (
                    f"Failed to load (HTTP {e.response.status_code})",
                    "Error loading stats.",
                    "Error loading files.",
                )
            except Exception as e:
                return (f"Error: {e}", "Error loading stats.", "Error loading files.")

        def clear_kb(auth: dict) -> tuple:
            """Clear knowledge base after confirmation."""
            token = auth.get("token")
            if not token:
                return ("Not authenticated.", "No data.", "No files.")

            try:
                result = GradioAPIClient.clear_knowledge_base(token)
                deleted_chunks = result.get("deleted_chunks", 0)
                deleted_files = result.get("deleted_files", 0)

                status = f"Cleared {deleted_chunks} chunks from {deleted_files} files."
                return (status, "**Total Chunks**: 0\n**Unique Files**: 0", "*No files indexed.*")
            except httpx.HTTPStatusError as e:
                return (f"Failed to clear (HTTP {e.response.status_code})", "Error.", "Error.")
            except Exception as e:
                return (f"Error: {e}", "Error.", "Error.")

        kb_refresh_btn.click(
            fn=load_kb_stats,
            inputs=[auth_state],
            outputs=[kb_status, kb_stats_display, kb_file_list],
        )

        kb_clear_btn.click(
            fn=clear_kb,
            inputs=[auth_state],
            outputs=[kb_status, kb_stats_display, kb_file_list],
        )

        # ========== MODEL CONFIGURATION EVENT HANDLERS (#14, #21) ==========

        def load_models(auth: dict) -> tuple:
            """Load available models from API (both embedding and chat)."""
            token = auth.get("token")
            if not token:
                return (
                    "Not authenticated.",
                    "**Current**: Unknown",
                    gr.update(choices=[]),
                    gr.update(visible=False),
                    "**Current**: Unknown",
                    gr.update(choices=[]),
                )

            try:
                data = GradioAPIClient.get_available_models(token)
                current_chat = data.get("current_chat_model", "Unknown")
                current_embed = data.get("current_embedding_model", "Unknown")

                # Get filtered model lists
                embedding_models = data.get("embedding_models", [])
                chat_models = data.get("chat_models", [])

                # Format embedding model choices
                embed_choices = [
                    (m.get("display_name", m.get("name", "")), m.get("name", ""))
                    for m in embedding_models
                ]

                # Format chat model choices
                chat_choices = [
                    (m.get("display_name", m.get("name", "")), m.get("name", ""))
                    for m in chat_models
                ]

                return (
                    "Models loaded.",
                    f"**Current**: {current_embed}",
                    gr.update(choices=embed_choices, value=current_embed),
                    gr.update(visible=True),  # Show warning
                    f"**Current**: {current_chat}",
                    gr.update(choices=chat_choices, value=current_chat),
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    return (
                        "Ollama unavailable.",
                        "**Current**: Ollama offline",
                        gr.update(choices=[]),
                        gr.update(visible=False),
                        "**Current**: Ollama offline",
                        gr.update(choices=[]),
                    )
                return (
                    f"Failed to load (HTTP {e.response.status_code})",
                    "**Current**: Error",
                    gr.update(choices=[]),
                    gr.update(visible=False),
                    "**Current**: Error",
                    gr.update(choices=[]),
                )
            except Exception as e:
                return (
                    f"Error: {e}",
                    "**Current**: Error",
                    gr.update(choices=[]),
                    gr.update(visible=False),
                    "**Current**: Error",
                    gr.update(choices=[]),
                )

        def apply_embedding_change(auth: dict, selected_model: str) -> tuple:
            """Apply the selected embedding model (with re-indexing warning)."""
            token = auth.get("token")
            if not token:
                return ("Not authenticated.", "**Current**: Unknown")

            if not selected_model:
                return ("No model selected.", "**Current**: Unknown")

            try:
                result = GradioAPIClient.change_embedding_model(
                    token, selected_model, confirm_reindex=True
                )
                new_model = result.get("current_model", selected_model)
                docs_affected = result.get("documents_affected", 0)
                return (
                    f"Embedding model changed to {new_model}. "
                    f"⚠️ {docs_affected} documents need re-indexing!",
                    f"**Current**: {new_model}",
                )
            except httpx.HTTPStatusError as e:
                return (
                    f"Failed to change model (HTTP {e.response.status_code})",
                    "**Current**: Error",
                )
            except Exception as e:
                return (f"Error: {e}", "**Current**: Error")

        def apply_chat_model_change(auth: dict, selected_model: str) -> tuple:
            """Apply the selected chat model."""
            token = auth.get("token")
            if not token:
                return ("Not authenticated.", "**Current**: Unknown")

            if not selected_model:
                return ("No model selected.", "**Current**: Unknown")

            try:
                result = GradioAPIClient.change_chat_model(token, selected_model)
                new_model = result.get("current_model", selected_model)
                return (f"Chat model changed to {new_model}.", f"**Current**: {new_model}")
            except httpx.HTTPStatusError as e:
                return (
                    f"Failed to change model (HTTP {e.response.status_code})",
                    "**Current**: Error",
                )
            except Exception as e:
                return (f"Error: {e}", "**Current**: Error")

        model_refresh_btn.click(
            fn=load_models,
            inputs=[auth_state],
            outputs=[
                model_status,
                embed_current,
                embed_dropdown,
                embed_warning,
                model_current,
                model_dropdown,
            ],
        )

        embed_apply_btn.click(
            fn=apply_embedding_change,
            inputs=[auth_state, embed_dropdown],
            outputs=[model_status, embed_current],
        )

        model_apply_btn.click(
            fn=apply_chat_model_change,
            inputs=[auth_state, model_dropdown],
            outputs=[model_status, model_current],
        )

    return app


# Export the app factory
__all__ = ["create_app"]
