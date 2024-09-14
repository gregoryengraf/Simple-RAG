# streamlit_app.py
import streamlit as st
import requests
import json
import os

API_URL = os.environ.get("API_URL", "http://172.20.0.2:5001")

st.set_page_config(page_title="RAG Query System", page_icon="🤖", layout="wide")

# Add this for debugging
st.write(f"API_URL: {API_URL}")

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

                    # Display the full response and sources
                    st.markdown("## Full Response")
                    st.write(full_response)

                    st.markdown("## Sources")
                    for i, source in enumerate(sources, 1):
                        st.write(f"Source {i}:")
                        st.text(source)
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