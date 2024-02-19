# app.py
from flask import Flask, render_template, request, jsonify
from llama_index import SimpleDirectoryReader, Document, ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from werkzeug.utils import secure_filename
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index import load_index_from_storage
from llama_index.llms import OpenAI
import os
from flask_cors import CORS





app = Flask(__name__)
CORS(app)





# Create a variable to store the uploaded file path
uploaded_file_path = ""
documents = None
document = None
index = None

def build_sentence_window_index():
    global documents, document, index, index2

    # Check if a document has been uploaded
    if not uploaded_file_path or not os.path.isfile(uploaded_file_path):
        return jsonify({'error': 'Please upload a document first'})

    # Load the document
    documents = SimpleDirectoryReader(input_files=[uploaded_file_path]).load_data()
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    # Build the sentence window index
    index = build_sentence_window_index_helper(3, save_dir="sentence_index_3")
    index2 = build_sentence_window_index_helper(5, save_dir="sentence_index_5")

    return jsonify({'success': True})

def build_sentence_window_index_helper(sentence_window_size,save_dir):
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
        embed_model="local:BAAI/bge-small-en-v1.5",
        node_parser=node_parser
    )

    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents([document], service_context=sentence_context)
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context
        )

    return sentence_index

def get_sentence_window_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model="BAAI/bge-reranker-base")

    sentence_window_engine = sentence_index.as_query_engine(similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank])
    return sentence_window_engine


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_document', methods=['POST'])
def upload_document():
    global uploaded_file_path, documents, document, index, index2

    # Check if 'file' is in request.files
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the file to the current working directory
    upload_folder = os.getcwd()  # Get the current working directory
    uploaded_file_path = os.path.join(upload_folder, secure_filename(file.filename))
    file.save(uploaded_file_path)

    # Build the sentence window index
    build_sentence_window_index()

    return jsonify({'success': True})




@app.route('/generate_response', methods=['POST'])
def generate_response():
    global uploaded_file_path, documents, document, index, index2

    # Check if a document has been uploaded
    if not uploaded_file_path or not os.path.isfile(uploaded_file_path):
        return jsonify({'error': 'Please upload a document first'})

    question = request.form.get('question')

    # Get the query engine
    query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)
    query_engine2=get_sentence_window_query_engine(index2, similarity_top_k=6)


    # Query for the response

    response = query_engine.query(question)

    response2 = query_engine2.query(question)
         
    


    return render_template('results.html',question=question, response=response, response2=response2)




if __name__ == '__main__':
    app.run(debug=True)

