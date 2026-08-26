"""
app.py — Streamlit Web Interface (Main Entry Point)

Module: Module 11 (Streamlit UI)

Responsibility:
    - Multi-User Authentication (Login & Sign Up tabs)
    - UUID-Based Chat Session Management with dynamic naming
    - Strict Session-Level Document Isolation
    - Strict Chat Guardrail: Enforces that documents must be uploaded in THIS specific chat
    - Light-themed, clean SaaS styling
"""

from pathlib import Path
import streamlit as st

from src.rag_pipeline import ask_question, ingest_document
from src.vector_store import load_vector_store
from src.user_manager import (
    create_user,
    authenticate_user,
    create_chat_session,
    get_recent_chats,
    get_chat_messages,
    get_chat_title,
    update_chat_title,
    save_chat_message,
    delete_all_chats,
)


# ============================================================================
# 1. PAGE SETUP & LIGHT THEME STYLING
# ============================================================================

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def apply_clean_light_css():
    """
    Inject clean, light SaaS CSS.
    Keeps sidebar toggle and standard Streamlit layout containers intact.
    Hides only the top-right hamburger menu and footer.
    """
    st.markdown(
        """
        <style>
        /* ── Google Fonts Inter ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #1F2937;
        }

        /* ── Hide hamburger menu & footer only ── */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* ── Clean Light Background ── */
        .stApp {
            background-color: #F9FAFB;
        }

        /* ── Global Text Styling ── */
        h1, h2, h3, h4, h5, h6 {
            color: #111827 !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }
        p, span, li, label, div {
            color: #1F2937;
        }

        /* ── Sidebar Styling ── */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E5E7EB;
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 1.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* ── Modern Buttons ── */
        .stButton > button,
        button[data-testid="stBaseButton-secondary"] {
            background-color: #FFFFFF !important;
            border: 1px solid #D1D5DB !important;
            border-radius: 8px !important;
            color: #1F2937 !important;
            font-weight: 500;
            font-size: 0.875rem;
            padding: 0.5rem 1rem;
            transition: all 0.15s ease-in-out;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        .stButton > button:hover,
        button[data-testid="stBaseButton-secondary"]:hover {
            background-color: #F3F4F6 !important;
            border-color: #9CA3AF !important;
            color: #111827 !important;
        }

        /* ── Chat Messages ── */
        div[data-testid="stChatMessage"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 12px !important;
            padding: 1rem 1.25rem !important;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }

        /* ── Chat Input ── */
        div[data-testid="stChatInput"] {
            padding-top: 0.75rem;
        }
        div[data-testid="stChatInput"] textarea {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.92rem;
            color: #1F2937 !important;
            background-color: #FFFFFF !important;
            border: 1px solid #D1D5DB !important;
            border-radius: 12px !important;
            padding: 0.75rem 1rem;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
        }
        div[data-testid="stChatInput"] textarea:focus {
            border-color: #4F46E5 !important;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
        }

        /* ── Text Input (Auth / Forms) ── */
        div[data-testid="stTextInput"] input {
            color: #1F2937 !important;
            background-color: #FFFFFF !important;
            border: 1px solid #D1D5DB !important;
            border-radius: 8px !important;
            padding: 0.6rem 0.9rem;
        }

        /* ── File Uploader ── */
        div[data-testid="stFileUploader"] {
            background-color: #FFFFFF;
            border: 1px dashed #D1D5DB;
            border-radius: 10px;
            padding: 0.75rem;
        }

        /* ── Expander ── */
        details[data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 8px;
            margin-top: 0.5rem;
        }

        /* ── Tabs Styling ── */
        div[data-testid="stTabs"] button[role="tab"] {
            font-size: 1rem;
            font-weight: 600;
            color: #6B7280;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #111827 !important;
            border-bottom-color: #4F46E5 !important;
        }

        /* ── Dividers ── */
        hr {
            border: none;
            border-top: 1px solid #E5E7EB;
            margin: 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_clean_light_css()


# ============================================================================
# 2. STATE MANAGEMENT & SESSION HELPERS
# ============================================================================

if "logged_in_username" not in st.session_state:
    st.session_state.logged_in_username = None

if "logged_in_name" not in st.session_state:
    st.session_state.logged_in_name = None

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None


def has_session_documents(username: str, session_id: str) -> bool:
    """
    Check if the Chroma vector store contains any chunks for this specific user & session.
    """
    try:
        vs = load_vector_store()
        res = vs._collection.get(
            where={
                "$and": [
                    {"username": {"$eq": username.strip()}},
                    {"session_id": {"$eq": session_id.strip()}},
                ]
            },
            limit=1,
        )
        return len(res.get("ids", [])) > 0
    except Exception:
        return False


# ============================================================================
# 3. AUTHENTICATION SCREEN (LOGIN / SIGN UP)
# ============================================================================

if st.session_state.logged_in_username is None:
    st.markdown("")
    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.title("🧠 Knowledge Assistant")
        st.markdown("Your personal AI assistant grounded in your uploaded documents.")
        st.markdown("")

        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        # ── Login Tab ──
        with tab_login:
            st.markdown("#### Sign In")
            login_username = st.text_input("Username", key="login_user_input")
            login_password = st.text_input("Password", type="password", key="login_pass_input")

            if st.button("Log In", use_container_width=True, key="btn_login"):
                if not login_username.strip() or not login_password.strip():
                    st.warning("Please enter both username and password.")
                else:
                    name = authenticate_user(login_username.strip(), login_password.strip())
                    if name:
                        st.session_state.logged_in_username = login_username.strip()
                        st.session_state.logged_in_name = name
                        # Immediately generate a new unique session for a fresh start
                        st.session_state.active_session_id = create_chat_session(login_username.strip(), "New Chat")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        # ── Sign Up Tab ──
        with tab_signup:
            st.markdown("#### Create Account")
            signup_name = st.text_input("Full Name", key="signup_name_input")
            signup_username = st.text_input("Username", key="signup_user_input")
            signup_password = st.text_input("Password", type="password", key="signup_pass_input")

            if st.button("Create Account", use_container_width=True, key="btn_signup"):
                if not signup_name.strip() or not signup_username.strip() or not signup_password.strip():
                    st.warning("Please fill in all fields.")
                else:
                    success = create_user(
                        username=signup_username.strip(),
                        password=signup_password.strip(),
                        name=signup_name.strip(),
                    )
                    if success:
                        st.success("✅ Account created successfully! Please switch to the **Login** tab to sign in.")
                    else:
                        st.error("❌ Username already exists. Please choose a different username.")

    st.stop()


# ============================================================================
# 4. MAIN APP: LOGGED IN SESSION
# ============================================================================

current_username = st.session_state.logged_in_username
current_display_name = st.session_state.logged_in_name

# Ensure active_session_id is initialized
if not st.session_state.active_session_id:
    st.session_state.active_session_id = create_chat_session(current_username, "New Chat")

current_session_id = st.session_state.active_session_id


# ============================================================================
# 5. SIDEBAR LAYOUT
# ============================================================================

with st.sidebar:
    # 1. New Chat Button: Immediately creates a brand new UUID session
    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.active_session_id = create_chat_session(current_username, "New Chat")
        st.rerun()

    st.markdown("")

    # 2. Document Upload Section: Strictly bound to current_session_id
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "docx", "txt"],
        help="Upload a PDF, Word, or text file to index strictly into this specific chat session.",
    )

    if uploaded_file is not None:
        if st.button("🚀 Index Document", use_container_width=True):
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            saved_file_path = data_dir / uploaded_file.name

            with open(saved_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            progress_bar = st.progress(0, text="Preparing document...")
            with st.spinner(f"Indexing {uploaded_file.name} into current session..."):
                try:
                    progress_bar.progress(25, text="Loading document...")
                    progress_bar.progress(50, text="Splitting into chunks...")
                    progress_bar.progress(75, text="Generating embeddings with session tag...")
                    result = ingest_document(
                        file_path=str(saved_file_path),
                        username=current_username,
                        session_id=current_session_id,
                    )
                    progress_bar.progress(100, text="Completed!")

                    st.success(
                        f"Indexed **{result['chunks_created']}** chunks from `{uploaded_file.name}`."
                    )
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    # 3. Recent Chats Section (Scoped to logged-in user with UUID lookup)
    st.subheader("Recent Chats")
    recent_chats = get_recent_chats(current_username)

    if not recent_chats:
        st.caption("No recent chats yet.")
    else:
        # Display user's past chats in reverse order (newest first)
        for chat_item in reversed(recent_chats):
            chat_sid = chat_item["session_id"]
            chat_title = chat_item["title"]
            is_active = (chat_sid == current_session_id)
            label = f"💬  {chat_title}" + (" (active)" if is_active else "")

            if st.button(label, key=f"btn_chat_{chat_sid}", use_container_width=True):
                st.session_state.active_session_id = chat_sid
                st.rerun()

    st.divider()

    # 4. Settings & User Controls
    top_k = st.slider("Retrieval context chunks (k)", min_value=1, max_value=5, value=3)

    if st.button("🗑️ Clear All Chats", use_container_width=True):
        delete_all_chats(current_username)
        st.session_state.active_session_id = create_chat_session(current_username, "New Chat")
        st.rerun()

    st.markdown("")

    # 5. Logout Button
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in_username = None
        st.session_state.logged_in_name = None
        st.session_state.active_session_id = None
        st.rerun()


# ============================================================================
# 6. MAIN CHAT AREA & STRICT SESSION GUARDRAIL
# ============================================================================

current_title = get_chat_title(current_username, current_session_id)
history_messages = get_chat_messages(current_username, current_session_id)

# Render Header / Greeting
if current_title == "New Chat" and len(history_messages) == 0:
    st.title(f"Hi {current_display_name}!")
    st.markdown("Welcome to **Knowledge Assistant**. Upload a document in the sidebar to start asking questions in this chat.")
else:
    st.title(current_title)

# Render Chat History for Active Session
for message in history_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Logic & Guardrail
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Dynamic Chat Naming if title is still default "New Chat"
    if current_title in ("New Chat", "", None):
        words = prompt.strip().split()
        dynamic_title = " ".join(words[:5]) + ("..." if len(words) > 5 else "")
        update_chat_title(current_username, current_session_id, dynamic_title)
    else:
        dynamic_title = None

    # 2. Instantly render and save the user message
    save_chat_message(current_username, current_session_id, "user", prompt, title=dynamic_title)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Instantly open assistant container & show spinner BEFORE any backend/DB check
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            # --- ALL BACKEND & DATABASE CHECKS RUN INSIDE THIS SPINNER ---
            has_docs = has_session_documents(current_username, current_session_id)

            # Condition A: 0 chunks in this session (Greeting vs Guardrail Warning)
            if not has_docs:
                clean_input = prompt.lower().strip().rstrip("!?.")
                common_greetings = {
                    "hi", "hello", "hey", "hii", "hiii", "heyy",
                    "greetings", "good morning", "good afternoon", "good evening"
                }

                if clean_input in common_greetings:
                    answer_text = (
                        f"Hi {st.session_state.logged_in_name}! I would love to help you. "
                        f"Please upload a document in the sidebar first so we can get started."
                    )
                else:
                    answer_text = "Please upload a document in this specific chat first before asking questions."

                sources = []
                error_msg = None

            # Condition B: Chunks exist -> Run RAG Pipeline (similarity search + LLM)
            else:
                try:
                    response = ask_question(
                        question=prompt,
                        k=top_k,
                        username=current_username,
                        session_id=current_session_id,
                    )
                    answer_text = response["answer"]
                    sources = response.get("source_documents", [])
                    error_msg = None
                except Exception as e:
                    answer_text = None
                    sources = []
                    error_msg = f"⚠️ An error occurred: `{e}`"

        # 4. Display response immediately after exiting spinner
        if answer_text:
            st.markdown(answer_text)

            if sources:
                with st.expander(f"📚 View Sources & Citations ({len(sources)})"):
                    for idx, doc in enumerate(sources, start=1):
                        source_file = doc.metadata.get("source", "Unknown file")
                        page = doc.metadata.get("page")
                        page_label = f" · Page {page + 1}" if page is not None else ""
                        st.markdown(
                            f"**Citation {idx}** — `{source_file}`{page_label}\n"
                            f"```\n{doc.page_content.strip()}\n```"
                        )

            save_chat_message(current_username, current_session_id, "assistant", answer_text)

        elif error_msg:
            st.error(error_msg)
            save_chat_message(current_username, current_session_id, "assistant", error_msg)

    # 5. Explicitly rerun so the chat title and sidebar update immediately
    st.rerun()
