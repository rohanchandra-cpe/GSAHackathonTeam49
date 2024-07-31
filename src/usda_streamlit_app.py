import streamlit as st
import ragpoweredchatbot as rg

st.title("RAG Powered Chatbot for USDA by Team 49 (AC and RC)")
st.write("This is a chatbot to aid whoever wants to visit the USDA website to help them \
         get better info on how to utilize the services the USDA has to offer!")

text_datastore = rg.test_vectorstore
text_datastore, conversation_id = rg.init_chatbot(text_datastore)

def ask_prompt(input_message):
    if input_message.lower() == "quit":
        st.write("Ending chat.")
    else:
        st.write(f"Farmers and Ranchers: {input_message}")
        
    response = rg.run_chatbot(text_datastore, conversation_id, input_message)

    # Print the chatbot response, citations, and documents
    st.write("\nUSDA AI Agent:")
    citations = []
    cited_documents = []

    # Display response
    output = ""
    for event in response:
        if event.event_type == "text-generation":
            output = output + event.text
        elif event.event_type == "citation-generation":
            citations.extend(event.citations)
        elif event.event_type == "search-results":
            cited_documents = event.documents
    st.write(output)

    # Display citations and source documents
    if citations:
        st.write("\n\nCITATIONS:")
        for citation in citations:
            st.write(citation)

        st.write("\nDOCUMENTS:")
        for document in cited_documents:
            st.write(document)
                
    st.write(f"\n{'-'*100}\n")

# Get the user message
input_message = st.text_input("Enter AI Prompt Here: ")
if input_message:
    ask_prompt(input_message)

