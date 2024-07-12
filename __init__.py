import os

# Please set this before running the chatbot
apiKey = None

if apiKey is not None:
    os.environ["OPENAI_API_KEY"] = apiKey
elif os.environ.get("OPENAI_API_KEY") is None:
    raise Exception("API Key not set, please go to chatbot's __init__.py and set the apiKey variable to your OpenAI API key.")

# Create a folder to store the product knowledge pool
folderLocation = os.path.abspath(f"./productKnowledgePool")
if not os.path.exists(folderLocation):
    os.makedirs(folderLocation)

vectorLocation = os.path.abspath(f"./vectorStore")
if not os.path.exists(vectorLocation):
    os.makedirs(vectorLocation)