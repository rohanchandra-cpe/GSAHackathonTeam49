We were challenged to provide some ways so USDA.gov could be made AI agent-friendly and LLMs could learn from it easily,
Our analysis of the site revealed a wealth of resources, including articles, PDFs, HTML, and FAQs. While the site does have a search capability, the results often return entire documents, requiring USDA users to sift through large amounts of text. This can be time-consuming and discouraging for users.
The solution we are providing is a Retrieval Augmented Generation. Where the documents and all the information from the USDA site could be indexed into Vector space and an LLM could use this information to provide precise answers to a user.
We used Cohere embeddings from their embed-english-v3 model along with their endpoints embed endpoints.
We parsed, chunked, and scraped information using the Python unstructured library and then converted the USDA documents into vector embeddings.
The embeddings were then indexed using a vector indexing library Hnswlib (in production a vector DB would be used instead such as Weaviate or Pinecone, but for this hackathon, we used an in-memory indexing)
This index was then used in RAG to provide much better and precise answers to the users.
We created a simple app using Streamlit to try the USDA AI agent we created using a browser.
Usage:
Run from the home directory
Streamlit run src/usda_streamlit_app.py