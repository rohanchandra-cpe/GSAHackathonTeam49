import cohere
import uuid
import hnswlib
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

# Cohere API key
api_key = 'yCtarI9Timle3sifbmKfgj2m3tgD9ntkJrmsf9H2'

# Set up Cohere client
co = cohere.Client(api_key)

raw_documents = [
    # {
    #     "title": "Food Supply Chains",
    #     "url": "https://www.usda.gov/coronavirus/food-supply-chain"},
    # {
    #     "title": "USDA Programs",
    #     "url": "https://www.rd.usda.gov/programs-services/single-family-housing-programs/single-family-housing-direct-home-loans"},
    # {
    #     "title": "Funding Opportunities",
    #     "url": "https://www.usda.gov/media/press-releases/2022/08/24/usda-announces-550-million-american-rescue-plan-funding-projects"},
    # {
    #     "title": "Single Family Loan Program",
    #     "url": "https://www.rd.usda.gov/programs-services/single-family-housing-programs/single-family-housing-guaranteed-loan-program"},
    # {
    #     "title": "Farm Loan Program",
    #     "url": "https://www.fsa.usda.gov/programs-and-services/farm-loan-programs/"},
    # {
    #     "title": "Transformer Models",
    #     "url": "https://docs.cohere.com/docs/transformer-models"}
    {
        "title": "Federal Government Chrome",
        "url": "https://www.cnn.com/2024/11/20/business/google-sell-chrome-justice-department/index.html"},
    {
        "title": "Senate Absences",
        "url": "https://www.cnn.com/2024/11/20/politics/judges-trump-biden-missing-senators/index.html"
    }

]

class Vectorstore:
    def __init__(self, raw_documents: List[Dict[str, str]]):
        self.raw_documents = raw_documents
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.load_and_chunk()
        self.embed()
        self.index()
        
    def load_and_chunk(self) -> None:
        """
        Loads the text from the sources and chunks the HTML content.
        """
        print("Loading documents...")

        for raw_document in self.raw_documents:
            elements = partition_html(url=raw_document["url"])
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                self.docs.append(
                    {
                        "title": raw_document["title"],
                        "text": str(chunk),
                        "url": raw_document["url"],
                    }
                )

    def embed(self) -> None:
        """
        Embeds the document chunks using the Cohere API.
        """
        print("Embedding document chunks...")

        batch_size = 90
        self.docs_len = len(self.docs)
        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i : min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)


    def index(self) -> None:
        """
        Indexes the documents for efficient retrieval.
        """
        print("Indexing documents...")

        self.idx = hnswlib.Index(space="ip", dim=1024)
        self.idx.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.idx.get_current_count()} documents.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves document chunks based on the given query.
    
        Parameters:
        query (str): The query to retrieve document chunks for.
    
        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved document chunks, with 'title', 'text', and 'url' keys.
        """

        # Dense retrieval
        query_emb = co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        ).embeddings

        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]

        # Reranking
        rank_fields = ["title", "text"] # We'll use the title and text fields for reranking

        docs_to_rerank = [self.docs[doc_id] for doc_id in doc_ids]

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v3.0",
            rank_fields=rank_fields
        )
    
        doc_ids_reranked = [doc_ids[result.index] for result in rerank_results.results]
        
        docs_retrieved = []
        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )

        return docs_retrieved

def init_chatbot(vectorstore: Vectorstore):

    """
    Initializes an instance of the Chatbot class.

    Parameters:
    vectorstore (Vectorstore): An instance of the Vectorstore class.

    """
    vectorstore = vectorstore
    conversation_id = str(uuid.uuid4())
    return vectorstore, conversation_id

def run_chatbot(vectorstore, conversation_id, message):
    """
    :param self: 
    :return: 
            
    Runs the chatbot application
    """

    # Generate search queries, if any
    response = co.chat(message=message, search_queries_only=True)
                
    # if there are search queries, retrieve document chunks and respond
    if response.search_queries:
        print ("Retrieving information...", end="")
                    
        # Retrieve document chunks for each query
        documents = []
        for query in response.search_queries:
            documents.extend(vectorstore.retrieve(query.text))
                    
        # Use document chunks to respond
        response = co.chat_stream(
            message=message,
            model='command-r',
            documents=documents,
            conversation_id=conversation_id,
        )
    # If there is no search query, directly respond
    else:
        response = co.chat_stream(
            message=message,
            model="command-r",
            conversation_id=conversation_id,
        )
    
    return response

# process the documents
test_vectorstore = Vectorstore(raw_documents)