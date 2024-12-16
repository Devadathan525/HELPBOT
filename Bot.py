import os
import shutil
import streamlit as st
import streamlit.components.v1 as components
from dataclasses import dataclass
from typing import Literal
from langchain_openai import OpenAI
from langchain.chains.conversation.memory import ConversationSummaryMemory
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from langchain_community.chat_message_histories import ChatMessageHistory
import logging
import traceback
import zipfile
from datetime import datetime
import math
import sys

# Determine base path
if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

# Define directories
PERSIST_DIR = os.path.join(base_path, "netscout", "storage")
DOC_DIR = os.path.join(base_path, "data")
ARCHIVE_DIR = os.path.join(base_path, "archives")
MAX_LOG_SIZE_KB = 100

# Setup a custom logger
logger = logging.getLogger("custom_logger")
logger.setLevel(logging.INFO)

# Check if the logger already has handlers to avoid adding multiple handlers
if not logger.hasHandlers():
    # Create a file handler for logger to a file
    file_handler = logging.FileHandler("app4.log")
    file_handler.setLevel(logging.INFO)

    # Create a logger format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)


def archive_log_file(log_file_path, archive_dir, max_size_kb):
    # Check if the log file exists and its size
    if os.path.exists(log_file_path):
        file_size_kb = math.ceil(os.path.getsize(log_file_path) / 1024)  # Round to KB
        if file_size_kb > max_size_kb:
            # Ensure the archive directory exists (create if needed)
            os.makedirs(archive_dir, exist_ok=True)

            # Create a timestamp for the archive file name
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            archive_path = os.path.join(archive_dir, f"app_{timestamp}.zip")

            # Zip the log file
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.write(log_file_path, os.path.basename(log_file_path))

            # Clear the original log file contents
            with open(log_file_path, 'w'):
                pass  # This will truncate the file

            logger.info(f"Archived log file to {archive_path}")


def create_context():
    """Create or recreate the context from documents."""
    if os.path.exists(PERSIST_DIR):
        logger.info("Creating context...")
        shutil.rmtree(PERSIST_DIR)
    
    try:
        # Attempt to load data from documents
        documents = SimpleDirectoryReader(DOC_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)

        st.sidebar.write("Context Created:")
        logger.info("Context Created")
        
    except ValueError as e:
        # Handle the case where no files are found
        logger.error(f"ValueError: {e}\nStack trace:\n{traceback.format_exc()}")
        st.sidebar.write("Error: No files found in data directory. Check the log for details.")
    except Exception as e:
        # Handle any other exceptions
        logger.error(f"Unexpected error occurred: {e}\nStack trace:\n{traceback.format_exc()}")
        st.sidebar.write("An unexpected error occurred. Check the log for details.")

if st.sidebar.button('Load Context'):
    create_context()


@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

def initialize_session_state():
    """Initialize Streamlit session state for chat history and token count."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        llm = OpenAI(
            temperature=0,
            openai_api_key="")
        st.session_state.conversation = ChatMessageHistory(
            llm=llm,
            memory=ConversationSummaryMemory(llm=llm),
        )
    if "human_prompt" not in st.session_state:
        st.session_state.human_prompt = ""

def on_click_callback():
    """Handle user input and generate AI response."""
    if st.session_state.human_prompt == st.session_state.history[-1].message if st.session_state.history else "":
        return
    
    human_prompt = st.session_state.human_prompt.strip()  # Clean up the input
    logger.debug(f"User input received: '{human_prompt}'")

    if not human_prompt:
        logger.warning("Empty input received.")
        st.session_state.history.append(Message("human", "[Empty Input]"))
        st.session_state.history.append(Message("ai", "You didn't enter any query. Please provide a valid input."))
        st.session_state.human_prompt = ""
        return

    # Define specific responses for special cases
    special_cases = {
        "0": "This is an invalid query.",
        "betscout": "Betscout is not a valid query. Did you mean Netscout?",
        "Give me account details": "Sorry we are not allowed to disclose private information.",
        "hi": "Hello! How can I assist you today?",
    }

    # Handle special cases
    if human_prompt in special_cases:
        logger.warning(f"Handling special case for input: '{human_prompt}'")
        response = special_cases[human_prompt]

        # Clear input and return early
        st.session_state.history.append(Message("human", human_prompt))
        st.session_state.history.append(Message("ai", response))
        st.session_state.human_prompt = ""
        return

    # Process the input query
    logger.info(f"Processing user input: {human_prompt}")
    try:
        # Load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

        # Query the index
        query_engine = index.as_query_engine()
        response = query_engine.query(human_prompt)
        logger.info(f"AI response: {response}")

    except FileNotFoundError as e:
        stack_trace = traceback.format_exc()
        logger.error(f"File not found error: {e}\nStack trace:\n{stack_trace}")
        response = "The index file was not found. Please check the system configuration."

    except ValueError as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Value error: {e}\nStack trace:\n{stack_trace}")
        response = "There was an issue with the value provided. Please try again."

    except RuntimeError as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Runtime error: {e}\nStack trace:\n{stack_trace}")
        response = "A runtime error occurred. Please try again later."

    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error occurred: {e}\nStack trace:\n{stack_trace}")
        response = "An unexpected error occurred. Please try again."

    # Append messages to chat history
    st.session_state.history.append(Message("human", human_prompt))
    st.session_state.history.append(Message("ai", response))

    # Clear the input field after processing
    st.session_state.human_prompt = ""
    logger.debug("Input field cleared.")

st.markdown("""
    <style>
    body {
        background-color: #f0f0f0;  /* Light grey background for the app */
    }
    .chat-bubble {
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        max-width: 70%;
        font-size: 16px;
        line-height: 1.4;
    }
    .human-bubble {
        background-color: #808080;  /* Grey background for human messages */
        color: #97d700;  /* Green text for human messages */
        align-self: flex-start;
    }
    .ai-bubble {
        background-color: #97d700;  /* Green background for AI messages */
        color: #000000;  /* Black text for AI messages */
        align-self: flex-end;
    }
    .row {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        margin-bottom: 10px;
    }
    .row-reverse {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin-bottom: 10px;
    }
    .stButton button {
        background-color: #97d700;  /* Green background for submit button */
        color: #000000;
        border: none;
        padding: 8px 16px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    .stButton button:hover {
        background-color: #45a049;  /* Darker green on hover */
    }
    </style>
""", unsafe_allow_html=True)

initialize_session_state()

##st.image("static/netscout.png", width=600)
st.title("HelpBot")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

with chat_placeholder:
    for chat in st.session_state.history:
        is_ai_message = chat.origin == 'ai'
        chat_bubble_class = 'ai-bubble' if is_ai_message else 'human-bubble'
        chat_row_class = 'row' if is_ai_message else 'row-reverse'

        div = f"""
        <div class="{chat_row_class}">
            <div class="chat-bubble {chat_bubble_class}">
                {chat.message}
            </div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)
    
    # Add extra space at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)

with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        label_visibility="collapsed",
        placeholder="Ask Anything..",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,
    )


components.html("""
<script>
const streamlitDoc = window.parent.document;
const buttons = Array.from(streamlitDoc.querySelectorAll('.stButton > button'));
const submitButton = buttons.find(el => el.innerText === 'Submit');
streamlitDoc.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        e.preventDefault(); // Prevent default form submission
        submitButton.click();
    }
});
</script>
""", height=0, width=0)

# Archive log file if needed
archive_log_file("app4.log", ARCHIVE_DIR, MAX_LOG_SIZE_KB)
