from flask import Flask, request, jsonify
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Replace this with your actual OpenAI API key
openai_api_key = "sk-yZaLG2XRTFqilN1owXZhyceUknDz5-Swx9afmpDU7fT3BlbkFJc6pfOZTBjtjG1Y0m1lOhVdkqu0b3j1tl2SRDygbpoA"

# Step 1: Load the data from the URL
url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
data = loader.load()

# Step 2: Split the content into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(data)

# Step 3: Create embeddings for the documents
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
document_embeddings = embeddings.embed_documents([doc.page_content for doc in documents])

# Step 4: Store the embeddings in a vector store
faiss_index = FAISS.from_vectors(document_embeddings, documents)

# Step 5: Set up the Retrieval-based Question-Answering chain
llm = OpenAI(openai_api_key=openai_api_key)
qa_chain = RetrievalQA(llm=llm, retriever=faiss_index.as_retriever())

# Step 6: Create a Flask API
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("question")
    response = qa_chain.run(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5000)
