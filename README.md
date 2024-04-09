# RAG_Chat
Chat application based on Retrieval Augmented Generation RAG technique, using LangChain, chromadb, BeatifulSoup and Streamlit libraries on Mistral7B llm. 

## High Level approach
The application pipeline broadly follows the typical ML pipeline in terms of data retrieval and wrangling in the pre-processing stage. This curated data is fed into the model to generate the response during the Model interaction stage. Finally, the output from the model is formated for any desired output specific requirements.

### Pre-processing
This application during the Pre-processing stage retrieves web pages from the pre-defined list of input urls. Beatiful soup library is used to scrap the web pages and additional processing is carried out to extract the content from the scrapped web pages. This extracted information is further processed using langchain RecursiveCharacterTextSplitter to chuck the information with a chunk_size=300 and chunk_overlap=15.

This extracted information chucks called as KnowledgeBase, is further processed in the pipeline to convert them into embeddings using HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") and finally store in Chroma vector database 

### Model Interaction

A simple chat application built on Streamlit library Chat component for interacting with the model. It has an Chat_Input component to receive the user input/query and a output container component. 

The model "mistralai/Mistral-7B-Instruct-v0.2" hosted on Huggingface inference API is being used by this RAG pipeline for the interaction. The chat_input is sent to the model to receive the output via the API endpoint. The conversation memory is retained such that the model is interaction has continuity making the model remember the previous answers.

Please Note : Create a Access Token in the Huggingface website for accessing the hosted model for intference API.

### Post-Processing

In the post-processing stage the output from the llm is used to enhance the user experience by adding response to the container component of the streamlit library with User and AI avatars to distinguish between them.

[![](https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/createdevenv2.svg)](https://console.brev.dev/environment/new?os=k9x5vzgw2&us=9oaswwpei&instance=t3.large-spot&diskStorage=120Gi&region=us-east-1&image=ami-002e20fe52878f8bf)
