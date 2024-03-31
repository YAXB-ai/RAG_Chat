#````````````````````````````````````````````#
#````````````IMMPORTING PAKAGES```````````````#
#````````````````````````````````````````````#

import requests
import chromadb
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from  langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

#````````````````````````````````````````````#
#````````````IMPORTING PAKAGES```````````````#
#````````````````````````````````````````````#

# #````````````````````````````````````````````#
# #````````````URL lIST```````````````#
# #````````````````````````````````````````````#

#urls=["https://www.cookmedical.com/","https://www.cookmedical.com/products/ ","https://www.cookmedical.com/patient-resources/","https://www.cookmedical.com/businesscare-integration/","https://www.cookmedical.com/support/capital-equipment-service/","https://www.cookmedical.com/support/general-product-information/","https://www.cookmedical.com/open-payments/","https://www.cookmedical.com/support/ordering-returns/","https://www.cookmedical.com/support/product-performance-reporting/","https://www.cookmedical.com/support/reimbursement/","https://www.cookmedical.com/support/supplier-information/","https://www.cookmedical.com/support/","https://www.cookmedical.com/about/","https://csr.cookmedical.com/","https://www.cookmedical.com/diversity-equity-inclusion/","https://www.cookmedical.com/about/ethics-compliance/","https://www.cookmedical.com/about/history/","https://www.cookmedical.com/about/mission-and-values/","https://www.cookmedical.com/newsroom/","https://www.cookmedical.com/about/sustainability-environmental-practices/","https://www.cookmedical.com/divisions/vascular-division/","https://www.cookmedical.com/aortic-intervention/","https://www.cookmedical.com/interventional-radiology/","https://www.cookmedical.com/lead-management/","https://www.cookmedical.com/peripheral-intervention/","https://www.cookmedical.com/divisions/medsurg-division/","https://www.cookmedical.com/critical-care/","https://www.cookmedical.com/endoscopy/","https://www.cookmedical.com/otolaryngology/","https://www.cookmedical.com/reproductive-health/","https://www.cookmedical.com/surgery/","https://www.cookmedical.com/urology/","https://www.cookmedical.com/careers/","https://www.cookmedical.com/contact/","https://vista.cookmedical.com/","https://cookcsd.b2clogin.com/cookcsd.onmicrosoft.com/oauth2/v2.0/authorize?p=b2c_1_signuporsigninpolicyprod&client_id=a401fa28-da2c-4f55-ad4e-77de950e66f4&redirect_uri=https%3A%2F%2Fmycook.cookmedical.com%2F&scope=openid%20offline_access&response_type=code&response_mode=form_post"]
urls=["https://www.cookmedical.com/","https://www.cookmedical.com/products/ ","https://www.cookmedical.com/patient-resources/","https://www.cookmedical.com/businesscare-integration/","https://www.cookmedical.com/support/capital-equipment-service/","https://www.cookmedical.com/support/general-product-information/","https://www.cookmedical.com/open-payments/","https://www.cookmedical.com/support/ordering-returns/","https://www.cookmedical.com/support/product-performance-reporting/","https://www.cookmedical.com/support/reimbursement/","https://www.cookmedical.com/support/supplier-information/","https://www.cookmedical.com/support/","https://www.cookmedical.com/about/","https://csr.cookmedical.com/","https://www.cookmedical.com/diversity-equity-inclusion/","https://www.cookmedical.com/about/ethics-compliance/","https://www.cookmedical.com/about/history/","https://www.cookmedical.com/about/mission-and-values/","https://www.cookmedical.com/newsroom/","https://www.cookmedical.com/about/sustainability-environmental-practices/","https://www.cookmedical.com/divisions/vascular-division/","https://www.cookmedical.com/aortic-intervention/","https://www.cookmedical.com/interventional-radiology/","https://www.cookmedical.com/lead-management/","https://www.cookmedical.com/peripheral-intervention/","https://www.cookmedical.com/divisions/medsurg-division/","https://www.cookmedical.com/critical-care/","https://www.cookmedical.com/endoscopy/","https://www.cookmedical.com/otolaryngology/","https://www.cookmedical.com/reproductive-health/","https://www.cookmedical.com/surgery/","https://www.cookmedical.com/urology/","https://www.cookmedical.com/careers/","https://www.cookmedical.com/contact/","https://vista.cookmedical.com/"]
# #"https://www.cookmedical.com/"

# #````````````````````````````````````````````#
# #````````````URL lIST```````````````#
# #````````````````````````````````````````````#

# #````````````````````````````````````````````#
# #````````````EMBEDDING MODEL```````````````#
# #````````````````````````````````````````````#

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# #````````````````````````````````````````````#
# #````````````EMBEDDING MODEL```````````````#
# #````````````````````````````````````````````#

# #````````````````````````````````````````````#
# #````````````ACCESS TEXT FILE```````````````#
# #````````````````````````````````````````````#

# fileobj=open("./sample.txt","a")

# #````````````````````````````````````````````#
# #````````````ACCESS TEXT FILE```````````````#
# #````````````````````````````````````````````#

# #````````````````````````````````````````````#
# #````````````LOADING DOCUMENTS```````````````#
# #````````````````````````````````````````````#

#for url in ["https://cookcsd.b2clogin.com/cookcsd.onmicrosoft.com/oauth2/v2.0/authorize?p=b2c_1_signuporsigninpolicyprod&client_id=a401fa28-da2c-4f55-ad4e-77de950e66f4&redirect_uri=https%3A%2F%2Fmycook.cookmedical.com%2F&scope=openid%20offline_access&response_type=code&response_mode=form_post"]:
for url in urls:
    ##parse and load html document from url
    
    loader=WebBaseLoader(url)
    document=loader.load()

    ## writting to text file sample.txt
    
    # fileobj.write(f"<={urls.index(url)}=>######################################################################")
    # fileobj.write(document[0].page_content)
    # fileobj.write("\n")
    
    # fileobj.write(f"<={urls.index(url)}=>#######################################################################")
    ##split documents into chunks
    
    splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=15)
    docs=splitter.split_documents(document)
    
    ##load documents to chromadb with embedding function
    
    client=Chroma.from_documents(docs,embeddings,persist_directory="./knowledge_base")




# #````````````````````````````````````````````#
# #````````````LOADING DOCUMENTS```````````````#
# #````````````````````````````````````````````#


client=chromadb.PersistentClient("./knowledge_base")
val=client.list_collections()
print(val)
print("\n")
coll=client.get_collection(name='langchain')

print(coll.count())




# import requests
# import os

# from  langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from sentence_transformers import SentenceTransformer
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import HuggingFaceEndpoint

# from bs4 import BeautifulSoup

# from langchain_community.llms import HuggingFaceEndpoint
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"







# text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=15)

# embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# loader=WebBaseLoader("https://www.ndtv.com/india-news/systematic-effort-by-pm-to-cripple-congress-financially-sonia-gandhi-5281139")
# data=loader.load()
# text_load=text_splitter.split_documents(data)
# db=Chroma(persist_directory="./persist_data",embedding_function=embeddings)
# retriver=db.as_retriever()
#answer=retriver.get_relevant_documents("ramachandran") 



# def get_answer(query):

#     question = query

#     template = """Question: {question}

#     Answer: Let's think step by step."""

#     prompt = PromptTemplate.from_template(template)




#     repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

#     llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.5)
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
#     answer=llm_chain.run(question)
#     return answer


#####                 #######
###### STREAMLIT CODE ########
# import streamlit as stream
# import random
# import time

      
# import streamlit as st

# st.title('This is a title')
# st.title('Meet The Best :blue[Bot] Ever___..')

# container=stream.container(height=400,border=True)
# message=stream.chat_input("say somthing")
# if 'converse' not in stream.session_state:
#     stream.session_state.converse=[]

# with container:
#     demo=container.chat_message(name="demo",avatar="🤖")
#     with demo:
#         demo.write("how can i help u")

#     if(message):
        
            
#             stream.session_state.converse.append({"user":message,"assistant":get_answer(message)})
#             for i in stream.session_state.converse:
#                 user=stream.chat_message(name="user",avatar="🦂")
#                 user.write(i['user'])
                
#                 ass=stream.chat_message(name="assistant",avatar="🤖")
#                 ass.write(i['assistant'])

# sidebar=stream.sidebar
# with sidebar:
#     sidebar.text_input("enter a url")
#     sidebar.write(stream.session_state.converse)
    
#####                 #######
###### STREAMLIT CODE ########
    
            #######################################################################################

#####                 #######
###### WEBSCRAPING ######## 





# print(text.content)
# soup=BeautifulSoup(text.content,"html.parser")
# p=soup.find("body").find_all("div")[0].find_all("table")[3].find_all("p",limit=6)
# p.pop(0)
# datas=[]
# data_load=[]

# for data in p:
#     datas.append(data.get_text())


#text_load=text_splitter.split_documents(Document(page_content="hi guru",metadata={"source":"user"}))     
   

