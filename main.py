import os
from dotenv import load_dotenv

from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain_core.messages import AIMessage, HumanMessage

from DataManagement import DataManager
from HistoryAdding import HistoryAdder
from Retrievial import Retriever
from Generation import Generator

if __name__ == "__main__":
    load_dotenv()

    llm = ChatCohere()
    embedding = CohereEmbeddings(model="embed-english-v3.0")
    file_path = "data/romeo_and_juliet.txt"

    dm = DataManager(file_path, embedding)
    qdrant = dm.create_vector_store()

    r = Retriever(qdrant, "similarity", {"k": 3})
    g = Generator(llm)
    ha = HistoryAdder(llm)

    chat_history = []

    while True:
        query = str(input("You: "))

        if query.lower() == "exit":
            break
        
        query_aware_history = ha.get_hist_context(chat_history, query)

        rel_docs = r.get_context(query_aware_history)
        
        res = g.generate_answer(rel_docs, query_aware_history)
        print("Answer: ", res)

        chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=res),
            ]
        )
