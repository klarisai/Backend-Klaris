import os
import logging
import asyncio
import tempfile
from typing import Dict, Any
from datetime import datetime
import pytz

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import qdrant_client
import edge_tts

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='.')
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TTS_VOICE = os.getenv("TTS_VOICE", "id-ID-ArdiNeural")  # Default to Indonesian voice

# Initialize conversation memory with larger window size
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    k=10  # Increased to keep last 10 conversations for better context
)

def get_vector_store() -> Qdrant:
    """Initialize and return the Qdrant vector store."""
    try:
        client = qdrant_client.QdrantClient(
            url=QDRANT_HOST,
            api_key=QDRANT_API_KEY
        )
        logger.info("Qdrant client connected successfully.")
        
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        logger.info("OpenAI embeddings created successfully.")

        vector_store = Qdrant(
            client=client, 
            collection_name=QDRANT_COLLECTION_NAME, 
            embeddings=embeddings,
        )
        return vector_store
    except Exception as e:
        logger.error(f"Error in get_vector_store: {str(e)}")
        raise

def create_qa_chain(vector_store: Qdrant) -> ConversationalRetrievalChain:
    """Create and return the conversational question-answering chain."""
    # Get current date and time in Manado timezone
    tz = pytz.timezone('Asia/Makassar')  # Manado uses WITA (Waktu Indonesia Tengah)
    current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    
    prompt_template = f"""
    You are Klaris, a friendly and intelligent virtual assistant specifically designed for Universitas Klabat (UNKLAB). 
    Current date and time in universitas klabat: {current_time}

    Follow these guidelines:

    1. Response Guidelines:
       - Provide concise yet comprehensive answers
       - Focus only on relevant UNKLAB information
       - Use clear and straightforward language
       - Include key details while avoiding unnecessary information
       
    2. Answer Format:
       - Start with the most important information
       - Use bullet points for multiple items
       - Keep explanations brief but complete
       - Include specific details when necessary
       
    3. Information Accuracy:
       - Only use verified UNKLAB data
       - For personnel:
         * State name, title, and current role
         * List only relevant responsibilities
       - For programs:
         * Provide official names and key details
         * Include essential program information
         * State current accreditation status
       
    4. Communication Style:
       - Use formal Indonesian
       - Be direct and informative
       - Ask for clarification if needed
       - Acknowledge information limitations

    Previous Chat History: {{chat_history}}
    Additional Context: {{context}}
    Current Question: {{question}}

    If the question is not about Universitas klabat (UNKLAB) or yourself, respond with: "Maaf, saya hanya dapat menjawab pertanyaan seputar Universitas Klabat (UNKLAB), silahkan bertanya tentang unklab."

    Otherwise, provide a professional and informative answer in Indonesian:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question", "chat_history"]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 10})  # Increased for more context

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model='gpt-4o-mini', temperature=0.7, openai_api_key=OPENAI_API_KEY),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': PROMPT},
        verbose=True
    )

    return qa

async def text_to_speech(text: str) -> str:
    """Convert text to speech using edge-tts."""
    try:
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(output_file.name)
        logger.info(f"Audio file generated: {output_file.name}")
        return output_file.name
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        raise

@app.route('/')
def home():
    return jsonify({"message": "Selamat datang di API QA Universitas Klabat. Gunakan endpoint /api/query untuk mengajukan pertanyaan."})

@app.route('/api/query', methods=['POST'])
async def query_qa():
    data = request.json
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "Query diperlukan"}), 400

    try:
        vector_store = get_vector_store()
        qa_chain = create_qa_chain(vector_store)

        logger.info(f"Processing query: {user_query}")
        result = await qa_chain.ainvoke({"question": user_query})

        answer = result['answer']
        source_docs = result.get('source_documents', [])

        # logger.info(f"Answer generated: {answer[:50]}...")  # Log first 50 characters

        # Generate TTS
        audio_file = await text_to_speech(answer)

        return jsonify({
            "answer": answer,
            "audio_file": os.path.basename(audio_file),
            "source_documents": [{"text": doc.page_content, "metadata": doc.metadata} for doc in source_docs]
        })
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": "Terjadi kesalahan: " + str(e)}), 500

@app.route('/api/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    try:
        directory = tempfile.gettempdir()
        return send_file(os.path.join(directory, filename), mimetype="audio/mpeg")
    except Exception as e:
        logger.error(f"Error serving audio file: {str(e)}")
        return jsonify({"error": "Terjadi kesalahan saat menyajikan file audio"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint tidak ditemukan"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Terjadi kesalahan internal server"}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')