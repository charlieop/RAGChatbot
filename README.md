# RAG Chatbot

## Introduction
This is a chatbot that uses Langchain to allow users to upload files as embedding context and chat with OpenAI GPT. 

## Setup
To use it, you would need a OpenAI API key and also have AWS Credential setup for S3 Bucket functionalities. 

To setup for AWS: check [tutorial for Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration) 

## To use
Upon importing chatbot, 2 files will be automatically created at your current working dir: `./productKnowledgePool` and ./vectorStore

To setup a vectorestore, first create a folder with the name of the product ID and put all files inside it. i.e. `./vectorStore/{product_ID}/{your files}`

Then to setup the vectorstore on S3
```
from RAGChatbot import VectorStore
VectorStore.initS3Storage("YOUR BUCKET NAME")
VectorStore.buildS3VectorStoreFor("PRODCT ID")
```

Now you can use the chatbot with:
```
from RAGChatbot import ChatBot
newChatbot = ChatBot("PRODUCT ID")
newChatbot.ask_and_print("Your question here")
```