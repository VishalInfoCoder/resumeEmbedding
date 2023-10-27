from flask import Flask, request,make_response
import uuid 
import json
from langchain.vectorstores import Chroma
import os
from langchain.docstore.document import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory='database'
app = Flask(__name__)
from openAi_config import main
embedding_function = main()
# print(embeddingFunction)
@app.route('/api/embeddings/CreateEmbeddings', methods=['POST'])
def hello_world():
    data = request.get_json()
    docs = Document(page_content=data['resumetext'], metadata={"fileName": data['filename'], "id": str(data['id'])})
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
     search_query="java and springboot developer with 2+ years of experience"
     results = vectordb.similarity_search_with_score(search_query,k=3)
     return make_response({'data':results,"status":True}) 

if __name__ == '__main__':
    app.run(debug=True)
