from typing import List
import os
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
import boto3
from time import sleep

debugMode = False
__bucket = None
__S3 = None
__bucket_name = None


def debug(*args, **kwargs) -> None:
    """
    Print debug information if debugMode is True.
    
    Args:
        *args: Variable number of positional arguments to be printed.
        **kwargs: Variable number of keyword arguments to be passed to the print function.
    
    Returns:
        None
    """
    if debugMode:
        print("DEBUG: " + " ".join(map(str, args)), **kwargs)


def initS3Storage(bucket_name: str) -> None:
    """
    Initializes the S3 storage.

    Parameters:
    bucket_name (str): The name of the S3 bucket.

    Returns:
    None
    """
    global __bucket, __S3, __bucket_name
    __S3 = boto3.client('s3')
    s3 = boto3.resource('s3')
    __bucket = s3.Bucket(bucket_name)
    __bucket_name = bucket_name


def turnOnDebug() -> None:
    """
    Turns on the debug mode.

    This function sets the global variable `debugMode` to True, enabling debug mode.

    Parameters:
        None

    Returns:
        None
    """
    global debugMode
    debugMode = True


def buildS3VectorStoreFor(id: str) -> bool:
    """
    Builds an S3 vector store for the given product ID.

    Args:
        id: The product ID for which the S3 vector store needs to be built.

    Returns:
        bool: True if the S3 vector store is successfully built and uploaded, False otherwise.
    """
    __buildLocalVectorStoreFor(id)
    if __uploadS3VectorstoreFor(id):
        deleteLocalVectorStore(id)
        return True
    deleteLocalVectorStore(id)
    return False


def getS3VectorStoreFor(id: str) -> Chroma:
    """
    Retrieves the S3 vector store for the given ID.

    Parameters:
    id (str): The ID of the vector store.

    Returns:
    Chroma: The S3 vector store if it exists, otherwise None.
    """
    if __downloadS3VectorstoreFor(id):
        return __getLocalVectorStoreFor(id)
    return None


def checkS3VectorStoreFor(id: str) -> bool:
    """
    Check if the S3 vector store contains the specified ID.

    Args:
        id (str): The ID to check for in the S3 vector store.

    Returns:
        bool: True if the ID is found in the S3 vector store, False otherwise.
    """
    __checkS3init()
    
    for retry in range(3):
        try:
            res = __S3.list_objects(Bucket=__bucket_name, Prefix=f"vectorstores/{id}" )
            if 'Contents' not in res:
                return False
            return True
        except Exception as e:
            debug("ERROR:", e)
            debug("Retrying...")
            sleep(3)
    return False


def deleteS3VectorstoreFor(id: str) -> bool:
    """
    Deletes the S3 vector store for the given ID.

    Args:
        id (str): The ID of the vector store to delete.

    Returns:
        bool: True if the vector store was successfully deleted, False otherwise.
    """
    __checkS3init()
    
    for retry in range(3):
        try:
            res = __S3.list_objects(Bucket=__bucket_name, Prefix=f"vectorstores/{id}" )
            if 'Contents' not in res:
                debug("No vectorstore found, nothing to delete")
                return True
            keys = []
            for content in res["Contents"]:
                keys.append({"Key": content["Key"]})
                debug(f"Deleting {content['Key']}")
                __bucket.delete_objects(Delete={"Objects": keys})
            return True

        except Exception as e:
            debug("ERROR:", e)
            debug("Retrying...")
            sleep(3)
    return False


def __checkS3init() -> None:
    if __bucket is None or __S3 is None:
        raise Exception("S3 storage not initialized, for initialization call initS3Storage(bucket_name) first")


def __buildLocalVectorStoreFor(id: str) -> Chroma:
    """
    Builds a local vector store for the given ID.

    Args:
        id (str): The ID for which the vector store is built.

    Returns:
        Chroma: The built vector store.

    """
    documentList = __load_document(id)
    if len(documentList) == 0:
        debug("No documents found!")
    vectorstore = __buildVectorStoreFromDocuments(documentList, id)
    return vectorstore


def __getLocalVectorStoreFor(id: str) -> Chroma:
    """
    Retrieves the local vector store for the given ID.

    Args:
        id (str): The ID of the vector store.

    Returns:
        Chroma: The local vector store for the given ID, or None if it doesn't exist or couldn't be loaded.
    """
    path = os.path.abspath(f"./vectorStore/{id}")
    if not os.path.exists(path):
        return None
    try:
        return Chroma(persist_directory=path, embedding_function=OpenAIEmbeddings())
    except:
        debug("Vector store could not be loaded")
        return None


def deleteLocalVectorStore(id: str) -> bool:
    """
    Deletes the local vector store with the given ID.

    Args:
        id (str): The ID of the vector store to delete.

    Returns:
        bool: True if the vector store was successfully deleted, False otherwise.
    """
    vectorStore = __getLocalVectorStoreFor(id)
    if vectorStore is not None:
        vectorStore._client.clear_system_cache()
        Chroma.delete_collection(vectorStore)
    try:
        os.system(f"rm -rf ./vectorStore/{id}")
    except Exception as e:
        debug("ERROR:", e)
        return False
    return True


def __downloadS3VectorstoreFor(id: str) -> bool:
    """
    Downloads the vector store for the given ID from Amazon S3.

    Args:
        id (str): The ID of the vector store to download.

    Returns:
        bool: True if the vector store was successfully downloaded, False otherwise.
    """
    __checkS3init()
    
    deleteLocalVectorStore(id)
    
    for retry in range(3):
        try:
            res = __S3.list_objects(Bucket=__bucket_name, Prefix=f"vectorstores/{id}" )
            if 'Contents' not in res:
                debug("No vectorstore found, cannot download")
                return False
            for content in res["Contents"]:
                key = content["Key"]
                file_name = key.split("/")[-1]
                father = "/".join(key.split("/")[2:-1])
                if father:
                    os.system(f"mkdir -p ./vectorStore/{id}/{father}")
                else:
                    os.system(f"mkdir -p ./vectorStore/{id}")
                debug(f"Downloading {file_name} from {key}")
                __bucket.download_file(key, f"./vectorStore/{id}/{father}/{file_name}")
            return True
        
        except Exception as e:
            debug("ERROR:", e)
            debug("Retrying...")
            sleep(3)
    debug("Failed to download vectorstore, deleting local copy")
    deleteLocalVectorStore(id)
    return False


def __upload_file(id, file_name, father=None) -> bool:
    """
    Uploads a file to the specified location in the vector store. Not to be called directly.

    Args:
        id (str): The ID of the vector store.
        file_name (str): The name of the file to upload.
        father (str, optional): The parent directory in the vector store. Defaults to None.

    Returns:
        bool: True if the file was successfully uploaded, False otherwise.
    """
    __checkS3init()
    
    if father:
        path = f"vectorstores/{id}/{father}/{file_name}"
    else:
        path = f"vectorstores/{id}/{file_name}"
    file_name = f"./vectorStore/{id}/{father}/{file_name}" if father else f"./vectorStore/{id}/{file_name}"
    debug(f"Uploading {file_name} to {path}")
    
    for retry in range(3):
        try:
            __bucket.upload_file(file_name, path)
            return True
        except Exception as e:
            debug("ERROR:", e)
            debug("Retrying...")
            sleep(3)
    debug(f"Failed to upload {file_name} to {path}")
    return False


def __uploadS3VectorstoreFor(id: str, path=None, father=None) -> bool:
    """
    Uploads the vector store for the given ID to S3 storage.

    Args:
        id (str): The ID of the vector store.
        path (str, optional): The path to the vector store directory. Not to be used directly.
        father (str, optional): The parent directory in S3 storage. Not to be used directly.
    Returns:
        bool: True if the vector store was successfully uploaded, False otherwise.
    """
    if path is None:
        path = os.path.abspath(f"./vectorStore/{id}")
    deleteS3VectorstoreFor(id)
    list_of_files = os.listdir(path)
    for file_name in list_of_files:
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            fatherPath = father + "/" + file_name if father else file_name
            __uploadS3VectorstoreFor(id, file_path, fatherPath)
        else:
            if not __upload_file(id, file_name, father):
                debug(f"Failed to upload {file_name}, causing the whole vectorstore to fail")
                debug(f"Deleting vectorstore {id} for both local and S3 storage")
                deleteS3VectorstoreFor(id)
                deleteLocalVectorStore(id)
                return False
    return True


def __buildVectorStoreFromDocuments(listOfDocuments: List[Document], id) -> Chroma:
    """
    Builds a vector store from a list of documents.

    Args:
        listOfDocuments (List[Document]): A list of documents to build the vector store from.
        id: The ID of the vector store.

    Returns:
        Chroma: The built vector store.

    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents=listOfDocuments)

    filtered_splits = filter_complex_metadata(all_splits)
    debug(len(filtered_splits))
    path = os.path.abspath(f"./vectorStore/{id}")

    return Chroma.from_documents(documents=filtered_splits, embedding=OpenAIEmbeddings(), persist_directory=path)


def __load_document(id) -> List[Document]:
    """
    Load documents from the product knowledge pool for a given product ID.

    Args:
        id (str): The product ID.

    Returns:
        List[Document]: A list of loaded documents.

    Raises:
        Exception: If the product knowledge pool for the given product ID does not exist.

    """
    filesLocation = os.path.abspath(f"./productKnowledgePool/{id}")

    if not os.path.exists(filesLocation):
        raise Exception(f"Product knowledge pool for {id} does not exist, you must first setup the product knowledge pool for this product ID")

    listOfFiles = os.listdir(filesLocation)
    debug(listOfFiles)

    listOfDocuments = []
    for file in listOfFiles:
        file_type = os.path.splitext(file)[1]
        if file_type == ".docx" or file_type == ".doc":
            loader = Docx2txtLoader(file_path=os.path.join(filesLocation, file))
            doc = loader.load()
            listOfDocuments.extend(doc)
            debug(f"{file} loaded with docx parser, elements count: {len(doc)}")
        elif file_type == ".xlsx" or file_type == ".xls":
            loader = UnstructuredExcelLoader(file_path=os.path.join(filesLocation, file), mode="elements")
            doc = loader.load()
            listOfDocuments.extend(doc)
            debug(f"{file} loaded with excel parser, elements count: {len(doc)}")
        elif file_type == ".pdf":
            loader = PyPDFLoader(file_path=os.path.join(filesLocation, file))
            doc = loader.load()
            listOfDocuments.extend(doc)
            debug(f"{file} loaded with pdf parser, elements count: {len(doc)}")
        elif file_type == ".txt":
            with open(os.path.join(filesLocation, file), "r") as f:
                doc = f.read()
                listOfDocuments.append(Document(doc))
                debug(f"{file} loaded with txt parser")
            f.close()
        else:
            debug(f"{file} could not be loaded because of unknown file type (not .docx, .doc, .xlsx, .xls, .pdf, .txt)")
    debug(f"Total number of documents: { len(listOfDocuments) }")
    return listOfDocuments