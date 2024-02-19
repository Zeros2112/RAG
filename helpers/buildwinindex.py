# Import necessary modules and classes from the 'utils' module
from utils import *

def build_sentence_window_index(
    document, llm, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="sentence_index"
):
    # Create the SentenceWindowNodeParser with default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    
    # Create a service context for sentence indexing with specified language model and embedding model
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    
    # Check if the save directory exists
    if not os.path.exists(save_dir):
        # If the directory doesn't exist, create a new VectorStoreIndex from the document
        sentence_index = VectorStoreIndex.from_documents(
            [document], service_context=sentence_context
        )
        # Persist the index to the specified directory
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        # If the directory exists, load the index from the storage context
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    # Return the built sentence window index
    return sentence_index
