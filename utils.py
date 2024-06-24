from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

def process_company(company_number, user_prompt, extractor):
    """
    Process a company's document by loading, splitting, embedding, and extracting information.

    Args:
        company_number (str): The unique identifier for the company.
        user_prompt (str): The prompt provided by the user for information extraction.
        extractor (dict): The extractor configuration for retrieving information.

    Returns:
        dict: A dictionary containing the results of the processing for the given company.
    """
    text_path = f'{company_number}.txt'

    results_dict = {}  # Initialize the results dictionary

    try:
        # Load the text document
        loader = TextLoader(text_path)
        document = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(document)
        
        # Use SentenceTransformer for embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create a FAISS vector store from the document chunks
        db = FAISS.from_documents(splits, embeddings)

        # print('after embedding')

        retriever = db.as_retriever(
            # search_type="mmr",
            # search_kwargs={'k': 20, 'lambda_mult': 0.25}
        )  # Only extract from first document

        rag_extractor = {
            "text": retriever  # fetch content of all docs
        } | extractor

        results = rag_extractor.invoke(
            user_prompt
        )

        # Store the results in the dictionary
        results_dict[company_number] = results

        # Output the current status
        print(f"Processed company {company_number}.")

    except Exception as e:
        # Output any error encountered
        print(f"Error processing company {company_number}: {e}")

    return results_dict
