from flask import Flask, request,make_response
import uuid 
import json
from langchain.vectorstores import Chroma
import os
from langchain.docstore.document import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
import warnings
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
import chroma

warnings.filterwarnings("ignore")
persist_directory='database'
app = Flask(__name__)
from openAi_config import main
embeddingFunction = main()
# print(embeddingFunction)
def preprocess_resume_text(resume_text):
  """Preprocesses the resume text."""
  # Clean the text
  resume_text = resume_text.lower()
  resume_text = resume_text.replace(",", "")
  resume_text = resume_text.replace(".", "")
  resume_text = resume_text.replace(":", "")
  resume_text = resume_text.replace(";", "")
  # Remove stop words
  stop_words = ["the", "is", "are", "of", "to", "and", "a", "in", "that", "have"]
  resume_text = " ".join([word for word in resume_text.split() if word not in stop_words])
  # Stem or lemmatize the words
  # ...
  return resume_text

def create_resume_index(resume_text):
  """Creates an index of the resume text using Chroma DB."""
  chroma_client = chroma.Client(persist_directory="IndexDb")
  chroma_index = chroma.Index()

  # Add the resume text to the index
  chroma_index.add_document(resume_text)

  # Save the index
  chroma_client.save_index(chroma_index, "resume_index")
@app.route('/api/embeddings/CreateEmbeddings', methods=['POST'])
def hello_world():
    data = request.get_json()
    docs=[]
    category=["FINANCE","BANKING","NON-IT","IT"]
    docs.append(Document(page_content=data['resumeText'], metadata={"fileName": data['fileName'], "id": str(data['id'])}))
    # for mydata in theData:
    #     metadata={"fileName":mydata['filename'],"id":str(mydata['_id']['$oid'])}
    #     text = mydata['resumetext']
    #     docs.append(Document(page_content=text, metadata=metadata))
    
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddingFunction, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None
    return make_response({'message':"Embedding successfull!","status":True})

@app.route('/api/embeddings/searchResumes',  methods=['POST'])
def get_data():
     data = request.get_json()
     vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddingFunction) 
    #  search_query="java and springboot developer with 2+ years of experience"
     results = vectordb.similarity_search_with_score(data["search_query"],k=3)
     print(results[0][0].metadata["id"])
     myResult=[result[0].metadata["id"] for result in results]
     print(myResult)
     return make_response({'data':myResult,"status":True}) 
@app.route('/api/embeddings/exampleEmbedding',  methods=['POST'])
def get_data_ex():
    # data = request.get_json()
    docs=[]
    with open('profiles.json', 'r',encoding='utf-8') as file:
        data = json.load(file)
    # print(data[0])
    newData=data[:15]
    results=[]
    # for resume in newData:
    #     results.append(checkRelavency(resume['resumetext']))
    #     docs.append(Document(page_content=resume['resumetext'], metadata={"fileName": resume['filename'], "id": str(resume['_id']['$oid'])}))
    for mydata in data:
        # metadata={"fileName":mydata['filename'],"id":str(mydata['_id']['$oid'])}
        text = mydata['resumetext']
        new_text=preprocess_resume_text(text)
        create_resume_index(new_text)
        # docs.append(Document(page_content=text, metadata=metadata))
    # print(len(docs))
    # vectordb = Chroma.from_documents(documents=docs, embedding=embeddingFunction, persist_directory="IndexDb")
    # vectordb.persist()
    # vectordb = None
    return make_response({'message':"Embedding successfull!","status":True})

@app.route('/api/embeddings/examplesearchResumes',  methods=['POST'])
def get_data_ex1():
     data = request.get_json()
     vectordb = Chroma(persist_directory="exampleDb", embedding_function=embeddingFunction) 
    #  search_query="java and springboot developer with 2+ years of experience"
     results = vectordb.similarity_search_with_score(data["search_query"],k=5) 
    #  print(results[0][1])
     myResult=[{"id":result[0].metadata["id"],"score":result[1],"filename":result[0].metadata["fileName"]}for result in results]
    #  print(myResult)
     return make_response({'data':myResult,"status":True}) 
def checkRelavency(text):
    vectordb = Chroma(persist_directory="exampleDb", embedding_function=embeddingFunction) 
    #  search_query="java and springboot developer with 2+ years of experience"
    results = vectordb.similarity_search_with_score(text,k=1) 
    #  print(results[0][1])
    myResult=[{"id":result[0].metadata["id"],"score":result[1],"filename":result[0].metadata["fileName"]}for result in results]
    return myResult
if __name__ == '__main__':
    app.run(debug=True)
    
