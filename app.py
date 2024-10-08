from flask import Flask, render_template, jsonify,request
from src.helper import download_HuggingFace_Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["GROQ_API_KEY"]="YOUR API KEY"
os.environ["PINECONE_API_KEY"]="YOUR API KEY"

embeddings = download_HuggingFace_Embeddings()


index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriver = docsearch.as_retriever(search_type="similarity", search_kwargs= {"k":3})


llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",system_prompt,
        ),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriver,question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get",methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input= msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)