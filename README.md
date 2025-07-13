# Chat With PDFs (Resuable for any domain custom knowledge) - RAG Application
## Built Using Amazon Bedrock, Langchain, Python, Docker, Amazon S3
## Models used:
    Amazon Titan Embedding G1 - Text
    Anthropic Claude 2.1

## Introduction
A Chatbot like RAG application with AWS Amazon Bedrock, docker, python, Langchain, and Streamlit. Used Retrieval-Augmented generation concept to provide context to the Large Language model along with user query to generate response from any customizable Knowledgebase.

following are discussed below:
- Architecture of the applications
- 2 applications for (ADMIN and USER) and create DOCKER images


## Architecture
![image info](./Bedrock-ChatWithPdf.png)

## ADMIN Application:
    - Build Admin Web application where AdminUser can upload the pdfs.
    - Append to exsisting data or Replace the current data.
    - The PDFs text is split into chunks
    - Using the Amazon Titan Embedding Model, create the vector representation of the chunks
    - Using FAISS, save the vector index locally
    - Upload the index to Amazon S3 bucket (can use other vector stores like OpenSearch, Pinecone, PgVector etc., I chose cost effective S3)

### Docker Commands:

  Build Docker Image:
  `docker build -t pdf-reader-admin .`

  Run ADMIN application:
  `docker run -e BUCKET_NAME=<YOUR S3 BUCKET NAME> -v ~/.aws:/root/.aws -p 8083:8083 -it pdf-reader-admin`



## USER Application:
    - Build User Web application where users can query / chat with the pdf.
    - At the application start, download the index files from S3 to build local FAISS index (vector store)
    - Langchain's RetrievalQA, does the following:
      - Convert the User's query to vector embedding using Amazon Titan Embedding Model (Make sure to use the same model that was used for creating the chunk's embedding on the Admin side)
      - Do similarity search to the FAISS index and retrieve 5 relevant documents pertaining to the user query to build the context
      - Using Prompt template, provide the question and context to the Large Language Model. We are using Claude model from Anthropic.
      - Display the LLM's response to the user.

### Docker Commands:

  Build Docker Image:
  `docker build -t pdf-reader-client .`

  Run ADMIN application:
  `docker run -e BUCKET_NAME=<YOUR S3 BUCKET NAME> -v ~/.aws:/root/.aws -p 8084:8084 -it pdf-reader-client`


#### Note: The docker volume mount is only needed in local. If you are running the container in ECS, or EKS, the iam role is used.
