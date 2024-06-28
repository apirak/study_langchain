import os
import openai
import getpass
import langchain_core.messages as messages

from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage
# from langchain_core.messages import AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langsmith.wrappers import wrap_openai
from langsmith import traceable
from dotenv import load_dotenv
from config import get_settings


load_dotenv()

settings = get_settings()
open_ai_key = settings.OPEN_AI_KEY
open_ai_org = settings.OPEN_AI_ORG
lang_smith_key = settings.LANG_SMITH_KEY

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = lang_smith_key
os.environ['OPENAI_API_KEY'] = open_ai_key

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

# a = vectorstore.similarity_search("cats")
#print(a)

# embedding = OpenAIEmbeddings().embed_query("fish")
# b = vectorstore.similarity_search_by_vector(embedding)
# print(b)

# retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result
# c = retriever.batch(["cat", "shark"])
# print(c)

retriever = vectorstore.as_retriever(
  search_type="similarity",
  search_kwargs={"k": 1},
  # return_source_documents=True,
)
d = retriever.batch(["cat", "shark"])
print(d)

# Auto-trace LLM calls in-context
# model = wrap_openai(openai.Client())
model = ChatOpenAI(model="gpt-3.5-turbo")

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model

response = rag_chain.invoke("what cats do in their free time?")
print(response.content)

# store = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]


# @traceable # Auto-trace this function
# def pipeline(user_input: str):
#     result = client.chat.completions.create(
#         messages=[{"role": "user", "content": user_input}],
#         model="gpt-3.5-turbo"
#     )
#     return result.choices[0].message.content

# pipeline("Hi, OpenAI!")

# print(dir(messages))