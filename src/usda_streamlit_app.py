import streamlit as st
import ragpoweredchatbot as rg
import uuid

st.title("RAG Powered Chatbot for USDA by Team 49 (AC and RC)")
st.write("This is a chatbot to aid whoever wants to visit the USDA website to help them \
         get better info on how to utilize the services the USDA has to offer!")

text_datastore = rg.test_vectorstore
text_datastore, conversation_id = rg.init_chatbot(text_datastore)

def ask_prompt(input_message):
    if input_message.lower() == "quit":
        # print ("Ending chat.")
        st.write("Ending chat.")
    else:
        # print (f"User: {user_input}")
        st.write(f"User: {input_message}")
        
    response = rg.run_chatbot(text_datastore, conversation_id, input_message)

    # Print the chatbot response, citations, and documents
    # print("\nChatbot:")
    st.write("\nChatbot:")
    citations = []
    cited_documents = []

    # Display response
    output = ""
    for event in response:
        if event.event_type == "text-generation":
            output = output + event.text
            # st.write(event.text.strip())
            # print(event.text, end="")
            # print(event.text)
        elif event.event_type == "citation-generation":
            citations.extend(event.citations)
        elif event.event_type == "search-results":
            cited_documents = event.documents
    st.write(output)

    # Display citations and source documents
    if citations:
        # print("\n\nCITATIONS:")
        st.write("\n\nCITATIONS:")
        for citation in citations:
            st.write(citation)
            # print(citation)

        # print("\nDOCUMENTS:")
        st.write("\nDOCUMENTS:")
        for document in cited_documents:
            # print(document)
            st.write(document)
                
    st.write(f"\n{'-'*100}\n")
    # print(f"\n{'-'*100}\n")

# Get the user message
input_message = st.text_input("Enter AI Prompt Here: ")
if input_message:
    print(input_message)
    ask_prompt(input_message)

# button = st.button("Ask USDA AI Agent")
# if button:
#     ask_prompt(input_message)

# while(True):
#     # Get the user message
#     user_input = st.text_input("Enter AI Prompt Here: ", key=str(uuid.uuid4()))
                
#     # Typing "quit" ends the conversation
#     if user_input == "":
#         continue
#     else:
#         if user_input.lower() == "quit":
#             # print ("Ending chat.")
#             st.write("Ending chat.")
#             break
#         else:
#             # print (f"User: {user_input}")
#             st.write(f"User: {user_input}")
        
#         response = rg.run_chatbot(text_datastore, conversation_id, user_input)

#         # Print the chatbot response, citations, and documents
#         # print("\nChatbot:")
#         st.write("\nChatbot:")
#         citations = []
#         cited_documents = []

#         # Display response
#         for event in response:
#             if event.event_type == "text-generation":
#                 st.write(event.text)
#                 # print(event.text, end="")
#             elif event.event_type == "citation-generation":
#                 citations.extend(event.citations)
#             elif event.event_type == "search-results":
#                 cited_documents = event.documents

#         # Display citations and source documents
#         if citations:
#             # print("\n\nCITATIONS:")
#             st.write("\n\nCITATIONS:")
#             for citation in citations:
#                 st.write(citation)
#                 # print(citation)

#             # print("\nDOCUMENTS:")
#             st.write("\nDOCUMENTS:")
#             for document in cited_documents:
#                 # print(document)
#                 st.write(document)
                
#         st.write(f"\n{'-'*100}\n")
#         # print(f"\n{'-'*100}\n")
