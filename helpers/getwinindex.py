# Import necessary modules and classes from the 'utils' module
from utils import *

def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    # Define a postprocessor for metadata replacement
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    
    # Define a SentenceTransformerRerank with specified rerank top-n and model
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    # Create a sentence window query engine from the sentence index
    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    
    # Return the built sentence window query engine
    return sentence_window_engine
