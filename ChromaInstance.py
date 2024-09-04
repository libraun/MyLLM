import chromadb

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from nltk.tokenize import sent_tokenize


class ChromaInstance:

    def __init__(self, db_path: str, db_name: str):

        self.embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2")

        # Get the persistent web-retrieved document collection
        client = chromadb.PersistentClient(path=db_path)

        # Then create a LangChain Chroma instance (typically faster than normal chroma)
        # from persistent collection.
        self.query_config = Chroma(
            client=client, 
            collection_name=db_name,
            embedding_function=self.embedding_function,
            persist_directory=db_name
        )
    
    def query_similar_document(self, text: str, max_similarity_score: float=1.2):
    
        # Filter out potential "wikipedia lists"
        init_query = self.query_config.similarity_search_with_score(text, k=3)
        if not init_query:
            return ""
        
        md_doc = [d.page_content for d, score in init_query if "list" not in d.page_content.split() \
                  and score < max_similarity_score][0]

        # If score is below the max, then use this document
        if md_doc is None:
            
            return ""

        # Split document by sentences, embed, then reorder top k 
        # similar sentences as the final input document 
        md_docs = sent_tokenize(md_doc)                    
        sent_collection = Chroma.from_texts(md_docs, self.embedding_function)
        
        # Get either 2 sentences or the only sentence (whichever is lower)
        num_items = min(2, len(md_docs))
        docs = sent_collection.similarity_search(text, k=num_items)

        # Remove non-unique entries
        docs = list(set([doc.page_content for doc in docs]))
        docs = " ".join(docs)

        return text.preprocess(docs)