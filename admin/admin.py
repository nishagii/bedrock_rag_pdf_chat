import os
import uuid
import boto3
import streamlit as st

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.embeddings import BedrockEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
except ImportError:
    # Fallback imports for older langchain versions
    try:
        from langchain.document_loaders import PyPDFLoader
        from langchain.embeddings import BedrockEmbeddings
    except ImportError:
        st.error(
            "Required packages not found. Please install langchain and its dependencies."
        )

# S3 client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Bedrock Embeddings 
bedrock_client = boto3.client(service_name="bedrock-runtime")

# Option 1: Amazon Titan (most common)
try:
    bedrock_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1", client=bedrock_client
    )
except Exception:
    try:
        # Option 2: Try Cohere embeddings
        bedrock_embeddings = BedrockEmbeddings(
            model_id="cohere.embed-english-v3", client=bedrock_client
        )
    except Exception:
        # Option 3: Fallback to a basic model or show error
        st.error(
            "No Bedrock embedding models are available. Please enable model access in AWS Bedrock console."
        )
        bedrock_embeddings = None


def get_unique_uuid():
    """
    generate unique uuids
    """
    return str(uuid.uuid4())


def split_text(pages, chunk_size, overlap_size):
    """
    Splits the text / pages into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap_size
    )
    docs = text_splitter.split_documents(pages)
    return docs


def create_vector_store(request_id, documents):
    """
    Create a vector store
    """
    if bedrock_embeddings is None:
        st.error(
            "Bedrock embeddings not available. Please enable model access in AWS Bedrock console."
        )
        return False

    try:
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        file_name = f"{request_id}.bin"
        folder_path = "/tmp/"
        vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

        ##upload to s3
        s3_client.upload_file(
            Filename=folder_path + file_name + ".faiss",
            Bucket=BUCKET_NAME,
            Key="my_faiss.faiss",
        )
        s3_client.upload_file(
            Filename=folder_path + file_name + ".pkl",
            Bucket=BUCKET_NAME,
            Key="my_faiss.pkl",
        )

        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False


def main():
    """Main function for the Streamlit PDF processing application."""
    st.title("PDF Document Processing Admin")
    st.write("Upload a PDF document for processing")

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        try:
            request_id = get_unique_uuid()
            st.write(f"Request ID: {request_id}")

            saved_file_name = f"{request_id}.pdf"
            with open(saved_file_name, mode="wb") as w:
                w.write(uploaded_file.getvalue())

            loader = PyPDFLoader(saved_file_name)
            pages = loader.load_and_split()

            st.success(f"Successfully processed PDF with {len(pages)} pages")
            st.write(f"Total Pages: {len(pages)}")

            ## split text(chunks=1000,overlap=200)
            splitted_docs = split_text(pages, 1000, 200)
            st.write(f"Splitted Docs len: {len(splitted_docs)}")
            st.write("===============")
            st.write(splitted_docs[0])
            st.write("===============")
            st.write(splitted_docs[1])

            st.write("creating the vector store")
            result = create_vector_store(request_id, splitted_docs)

            if result:
                st.write("yey!! PDF processed successfully")
            else:
                st.write("Error!! Please check again")

            # Optional: Upload to S3 if BUCKET_NAME is configured
            # if BUCKET_NAME:
            #     try:
            #         s3_client.upload_file(
            #             saved_file_name, BUCKET_NAME, f"pdfs/{saved_file_name}"
            #         )
            #         st.success(f"File uploaded to S3 bucket {BUCKET_NAME}")
            #     except Exception as e:
            #         st.warning(f"Could not upload to S3: {str(e)}")

            # # Clean up the file after processing
            # if os.path.exists(saved_file_name):
            #     os.remove(saved_file_name)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    main()
