# streamlit_app.py
import streamlit as st
import requests
import json
import os
import base64
from openai import OpenAI

API_URL = os.environ.get("API_URL", "http://172.20.0.2:5001")

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

st.set_page_config(page_title="RAG Query System", page_icon="🤖", layout="wide")

# Add this for debugging
st.write(f"API_URL: {API_URL}")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to generate audio from text
def text_to_speech(text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )

    # Save the audio to a file
    audio_file_path = "response_audio.mp3"
    with open(audio_file_path, "wb") as audio_file:
        audio_file.write(response.content)

    return audio_file_path

# Function to create an HTML audio player
def get_audio_player(audio_file_path):
    audio_file = open(audio_file_path, "rb")
    audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    return f'<audio autoplay controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Query", "Ingest"])

if page == "Query":
    st.title("RAG Query System")

    # Input for the question
    question = st.text_input("Enter your question:")

    # Input for the context
    context = st.text_input("Enter the context:")

    if st.button("Submit"):
        if question and context:
            # Prepare the payload
            payload = {
                "question": question,
                "context": context,
                "stream": True
            }

            # Send POST request to the API
            with st.spinner("Generating response..."):
                response = requests.post(f"{API_URL}/query", json=payload, stream=True)

                # Check if the request was successful
                if response.status_code == 200:
                    # Create placeholder for streaming response
                    response_placeholder = st.empty()
                    full_response = ""
                    sources = []

                    # Process the streaming response
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if "token" in data:
                                full_response += data["token"]
                                response_placeholder.markdown(full_response)
                            elif "sources" in data:
                                sources = data["sources"]

                    # Generate audio for the full response
                    audio_file_path = text_to_speech(full_response)

                    # Display audio player
                    st.markdown("## Audio Response")
                    st.markdown(get_audio_player(audio_file_path), unsafe_allow_html=True)

                    # Display the full response and sources
                    # st.markdown("## Full Response")
                    # st.write(full_response)
                    #
                    # st.markdown("## Sources")
                    # for i, source in enumerate(sources, 1):
                    #     st.write(f"Source {i}:")
                    #     st.text(source)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
        else:
            st.warning("Please enter both a question and a context.")

elif page == "Ingest":
    st.title("Ingest New Document")

    # File uploader
    uploaded_file = st.file_uploader("Choose a text file", type="txt")

    # Input for the context of the uploaded file
    file_context = st.text_input("Enter the context for the uploaded file:")

    if st.button("Ingest File"):
        if uploaded_file is not None and file_context:
            # Read the contents of the file
            file_contents = uploaded_file.getvalue().decode("utf-8")

            # Prepare the payload
            payload = {
                "file_content": file_contents,
                "context": file_context
            }

            # Send POST request to the API
            with st.spinner("Ingesting file..."):
                response = requests.post(f"{API_URL}/ingest", json=payload)

                # Check if the request was successful
                if response.status_code == 200:
                    st.success("File ingested successfully!")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
        else:
            st.warning("Please upload a file and enter a context.")

    def fetch_ingested_files():
        try:
            response = requests.get(f"{API_URL}/ingested_files")
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error fetching ingested files. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching ingested files: {e}")
        return []

    # Display ingested files (you'll need to implement this on the backend)
    st.markdown("## Ingested Files")
    ingested_files = fetch_ingested_files()
    for file in ingested_files:
        st.write(f"File: {file['name']}, Context: {file['context']}")