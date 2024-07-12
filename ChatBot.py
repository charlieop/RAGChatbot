from langchain_openai import ChatOpenAI
from chatbot import VectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory



llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

__contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

__system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

__contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", __contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

__qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", __system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


__question_answer_chain = create_stuff_documents_chain(llm, __qa_prompt)


class ChatBot:
    
    
    def __init__(self, id, from_s3=True):
        self.__id = id
        
        if from_s3:
            self.__vectorStore = VectorStore.getS3VectorStoreFor(self.__id)
        else:
            print("this is only for debug purposes, use from_s3=True for production")
            self.__vectorStore = VectorStore.__getLocalVectorStoreFor(self.__id)
        if self.__vectorStore is None:
            print("Vector store could not be loaded")
            raise Exception("Vector store could not be loaded")
        
        self.__retriever = self.__vectorStore.as_retriever(
            search_type="similarity", search_kwargs={"k": 15}
        )
        self.__history_aware_retriever = create_history_aware_retriever(
            llm, self.__retriever, __contextualize_q_prompt
        )
        self.history = None
        self.__rag_chain = create_retrieval_chain(
            self.__history_aware_retriever, __question_answer_chain
        )
        self.__conversational_rag_chain = RunnableWithMessageHistory(
            self.__rag_chain,
            self.__get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )


    def __del__(self):
        VectorStore.deleteVectorStore(self.__id)


    def __get_session_history(self, _):
        if self.history is None:
            self.history = ChatMessageHistory()
        return self.history


    def ask_question(self, question):
            """
            Asks a question to the chatbot and returns the answer.

            Args:
                question (str): The question to ask the chatbot.

            Returns:
                str: The answer provided by the chatbot.
            """
            return self.__conversational_rag_chain.stream(
                {
                "input": question,},
                config={
                    "configurable": {"session_id": "None"}
                },
            )["answer"]


    def ask_and_print(self, question):
        stream = self.ask_question(question)
        i = 0
        print("\nAnswer:")
        for chunk in stream:
            if i == 80:
                print()
                i = 0
            i += 1
            print(chunk, end="", flush=True)
