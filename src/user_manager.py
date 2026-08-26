"""
src/user_manager.py — Local Multi-User Authentication & UUID-Based Chat Persistence

Responsibility:
    - Multi-user authentication (create user, login/authenticate)
    - UUID-based chat session tracking for strict session-level isolation
    - All chat actions are scoped to the authenticated username and unique session_id (UUID)
    - Immediate disk persistence and robust error recovery

Data Schema (user_data.json):
    {
        "users": {
            "username123": {
                "password": "password123",
                "name": "Aryan",
                "chats": {
                    "e4b1c2a0-1234-4567-89ab-cdef01234567": {
                        "title": "What is PCI-DSS?",
                        "messages": [
                            {"role": "user", "content": "What is PCI-DSS?"},
                            {"role": "assistant", "content": "PCI-DSS stands for..."}
                        ]
                    }
                }
            }
        }
    }

Imported by:
    - app.py (Streamlit UI)
"""

import json
import os
import uuid
from typing import Optional, List, Dict, Any

# Path to the local JSON persistence file (project root)
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_data.json")


# ============================================================================
# INTERNAL FILE HELPERS
# ============================================================================

def _init_file() -> Dict[str, Any]:
    """
    Ensure user_data.json exists with the multi-user structure {"users": {}}.
    Automatically handles migrations or missing files.
    Returns the loaded data dictionary.
    """
    default_structure = {"users": {}}

    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(default_structure, f, indent=2, ensure_ascii=False)
        return default_structure

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            data = default_structure

        if "users" not in data or not isinstance(data["users"], dict):
            data = default_structure
            _save(data)

        return data

    except (json.JSONDecodeError, IOError, ValueError):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(default_structure, f, indent=2, ensure_ascii=False)
        return default_structure


def _save(data: Dict[str, Any]) -> None:
    """
    Write the full data dictionary back to user_data.json immediately.
    """
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# MULTI-USER AUTHENTICATION
# ============================================================================

def create_user(username: str, password: str, name: str) -> bool:
    """
    Register a new user account.

    Args:
        username (str): Unique login identifier.
        password (str): Account password.
        name (str):     Display name (e.g. "Aryan").

    Returns:
        bool: True if created successfully, False if the username already exists or inputs are empty.
    """
    username = username.strip()
    password = password.strip()
    name = name.strip()

    if not username or not password or not name:
        return False

    data = _init_file()

    if username in data["users"]:
        return False

    data["users"][username] = {
        "password": password,
        "name": name,
        "chats": {},
    }

    _save(data)
    return True


def authenticate_user(username: str, password: str) -> Optional[str]:
    """
    Authenticate user credentials.

    Args:
        username (str): Login identifier.
        password (str): Account password.

    Returns:
        Optional[str]: The user's display name if authentication succeeds, else None.
    """
    username = username.strip()
    password = password.strip()

    if not username or not password:
        return None

    data = _init_file()

    user_record = data["users"].get(username)
    if not user_record:
        return None

    if user_record.get("password") == password:
        return user_record.get("name", username)

    return None


def user_exists(username: str) -> bool:
    """
    Check if a specific username exists.

    Args:
        username (str): Username to check.

    Returns:
        bool: True if user exists, False otherwise.
    """
    data = _init_file()
    return username.strip() in data["users"]


# ============================================================================
# UUID-BASED CHAT HISTORY MANAGEMENT
# ============================================================================

def create_chat_session(username: str, title: str = "New Chat") -> str:
    """
    Create a new chat session for a user with a unique UUID.

    Args:
        username (str): The logged-in user's username.
        title (str):    Initial session display title.

    Returns:
        str: Newly generated session_id (UUID).
    """
    username = username.strip()
    data = _init_file()

    user_record = data["users"].get(username)
    if not user_record:
        raise ValueError(f"User '{username}' does not exist.")

    if "chats" not in user_record or not isinstance(user_record["chats"], dict):
        user_record["chats"] = {}

    session_id = str(uuid.uuid4())
    user_record["chats"][session_id] = {
        "title": title.strip() if title else "New Chat",
        "messages": [],
    }

    _save(data)
    return session_id


def get_recent_chats(username: str) -> List[Dict[str, str]]:
    """
    Retrieve all chat sessions for a specific user.

    Args:
        username (str): The logged-in user's username.

    Returns:
        List[Dict[str, str]]: List of dicts with 'session_id' and 'title',
                              e.g. [{"session_id": "uuid-1", "title": "What is RAG?"}]
    """
    data = _init_file()
    user_record = data["users"].get(username.strip())
    if not user_record or "chats" not in user_record:
        return []

    results = []
    for session_id, chat_data in user_record["chats"].items():
        title = chat_data.get("title", "New Chat") if isinstance(chat_data, dict) else "New Chat"
        results.append({
            "session_id": session_id,
            "title": title,
        })
    return results


def get_chat_messages(username: str, session_id: str) -> List[Dict[str, str]]:
    """
    Retrieve all messages for a specific UUID chat session of a user.

    Args:
        username (str):   The logged-in user's username.
        session_id (str): The UUID string identifying the chat session.

    Returns:
        List[Dict]: List of message dictionaries with 'role' and 'content'.
    """
    data = _init_file()
    user_record = data["users"].get(username.strip())
    if not user_record or "chats" not in user_record:
        return []

    chat_data = user_record["chats"].get(session_id)
    if not chat_data or not isinstance(chat_data, dict):
        return []

    return chat_data.get("messages", [])


def get_chat_title(username: str, session_id: str) -> str:
    """
    Retrieve the title of a specific UUID chat session.

    Args:
        username (str):   The logged-in user's username.
        session_id (str): The UUID string identifying the chat session.

    Returns:
        str: The title of the chat, or "New Chat" if not found.
    """
    data = _init_file()
    user_record = data["users"].get(username.strip())
    if not user_record or "chats" not in user_record:
        return "New Chat"

    chat_data = user_record["chats"].get(session_id)
    if not chat_data or not isinstance(chat_data, dict):
        return "New Chat"

    return chat_data.get("title", "New Chat")


def update_chat_title(username: str, session_id: str, new_title: str) -> None:
    """
    Update the title of an existing UUID chat session.

    Args:
        username (str):   The logged-in user's username.
        session_id (str): The UUID string identifying the chat session.
        new_title (str):  The new display title.
    """
    username = username.strip()
    data = _init_file()

    user_record = data["users"].get(username)
    if not user_record or "chats" not in user_record:
        return

    chat_data = user_record["chats"].get(session_id)
    if chat_data and isinstance(chat_data, dict):
        chat_data["title"] = new_title.strip()
        _save(data)


def save_chat_message(
    username: str,
    session_id: str,
    role: str,
    content: str,
    title: Optional[str] = None,
) -> None:
    """
    Append a message to a specific UUID chat session for a user.
    Creates the session if it does not already exist.
    Updates the title if provided and current title is default.
    Persists changes immediately to user_data.json.

    Args:
        username (str):            The logged-in user's username.
        session_id (str):          The UUID string identifying the chat session.
        role (str):                Message role ("user" or "assistant").
        content (str):             Message text content.
        title (str, optional):     Display title for the session (if setting dynamically).
    """
    username = username.strip()
    data = _init_file()

    user_record = data["users"].get(username)
    if not user_record:
        return

    if "chats" not in user_record or not isinstance(user_record["chats"], dict):
        user_record["chats"] = {}

    if session_id not in user_record["chats"]:
        user_record["chats"][session_id] = {
            "title": title.strip() if title else "New Chat",
            "messages": [],
        }
    elif title and user_record["chats"][session_id].get("title") in ("New Chat", "", None):
        user_record["chats"][session_id]["title"] = title.strip()

    user_record["chats"][session_id]["messages"].append({
        "role": role,
        "content": content,
    })

    _save(data)


def delete_chat_session(username: str, session_id: str) -> None:
    """
    Delete a single UUID chat session for a user.

    Args:
        username (str):   The logged-in user's username.
        session_id (str): The UUID string identifying the chat session to delete.
    """
    username = username.strip()
    data = _init_file()

    user_record = data["users"].get(username)
    if not user_record or "chats" not in user_record:
        return

    user_record["chats"].pop(session_id, None)
    _save(data)


def delete_all_chats(username: str) -> None:
    """
    Clear all chat sessions for a specific user.

    Args:
        username (str): The logged-in user's username.
    """
    username = username.strip()
    data = _init_file()

    user_record = data["users"].get(username)
    if not user_record:
        return

    user_record["chats"] = {}
    _save(data)
