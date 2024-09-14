import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response, stream_with_context
from langchain.text_splitter import CharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json
from sqlalchemy import create_engine, Column, Integer, String, text, select
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Get PostgreSQL connection string from environment variable
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")
if not PG_CONNECTION_STRING:
    raise ValueError("PG_CONNECTION_STRING not found in environment variables")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Initialize PGVector
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection=PG_CONNECTION_STRING,
    use_jsonb=True,
)

# Create engine and session
engine = create_engine(PG_CONNECTION_STRING)
Session = sessionmaker(bind=engine)

# Create the vector extension
with engine.connect() as connection:
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    connection.commit()

# Define the SQLAlchemy model
Base = declarative_base()

class IngestedFile(Base):
    __tablename__ = 'ingested_files'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    context = Column(String)

# Create the table if it doesn't exist
Base.metadata.create_all(engine)

@app.route('/ingest', methods=['POST'])
def ingest():
    data = request.json
    file_content = data.get('file_content')
    context = data.get('context')
    file_name = data.get('file_name', 'Unnamed File')

    if not file_content or not context:
        return jsonify({"error": "Missing file_content or context"}), 400

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(file_content)

    # Create embeddings and store in the database
    texts_with_context = [{"text": chunk, "context": context} for chunk in chunks]
    vectorstore.add_texts([t["text"] for t in texts_with_context], metadatas=[{"context": t["context"]} for t in texts_with_context])

    # Add the file to the ingested_files table
    with Session() as session:
        new_file = IngestedFile(name=file_name, context=context)
        session.add(new_file)
        session.commit()

    return jsonify({"message": "File ingested successfully"}), 200

@app.route('/ingested_files', methods=['GET'])
def get_ingested_files():
    with Session() as session:
        files = session.query(IngestedFile).all()
    return jsonify([{"id": f.id, "name": f.name, "context": f.context} for f in files])

def create_qa_chain(stream=False):
    # Create a prompt template
    template = """You are an AI assistant. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Human: {question}

    AI Assistant: Let me analyze the question and provide a detailed answer based on the given context.

    Question type: [Determine the type of question, e.g., factual, opinion, clarification]
    Intent: [Identify the main intent or goal of the question]

    Analysis:
    1. [Provide a brief analysis of how the question relates to the context]
    2. [Mention any key points from the context that are relevant to answering the question]

    Answer:
    [Provide a comprehensive answer to the question based on the context]

    Is there anything else you would like to know about this topic?
    
    Please, provide the answer in the same Human language.
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Create a retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Create a RetrievalQA chain
    if stream:
        callbacks = [StreamingStdOutCallbackHandler()]
    else:
        callbacks = []

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(streaming=stream, callbacks=callbacks),
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    context = data.get('context')
    stream = data.get('stream', False)

    if not question or not context:
        return jsonify({"error": "Missing question or context"}), 400

    qa_chain = create_qa_chain(stream=stream)

    if stream:
        return Response(stream_with_context(streaming_query(qa_chain, question)), content_type='application/json')
    else:
        # Get the answer
        result = qa_chain.invoke({"query": question})
        return jsonify({
            "question": question,
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }), 200

def streaming_query(qa_chain, question):
    class StreamHandler(StreamingStdOutCallbackHandler):
        def __init__(self):
            self.tokens = []

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.tokens.append(token)
            if token.endswith(("\n", ".", "!", "?")):
                yield json.dumps({"token": "".join(self.tokens)}) + "\n"
                self.tokens = []

    stream_handler = StreamHandler()

    # Use combine_documents_chain instead of llm
    qa_chain.combine_documents_chain.llm_chain.llm.callbacks = [stream_handler]

    result = qa_chain.invoke({"query": question})

    # Use a generator function to yield tokens
    def token_generator():
        yield from stream_handler.on_llm_new_token("", **{})
        # Yield any remaining tokens
        if stream_handler.tokens:
            yield json.dumps({"token": "".join(stream_handler.tokens)}) + "\n"
        # Yield the sources
        yield json.dumps({
            "sources": [doc.page_content for doc in result["source_documents"]]
        }) + "\n"

    return token_generator()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)