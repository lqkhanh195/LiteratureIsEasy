import os
from dotenv import load_dotenv

from langchain_cohere import CohereEmbeddings
from DataManagement import DataManager
from Retrievial import Retriever
from Generation import Generator

if __name__ == "__main__":
    load_dotenv()

    embedding = CohereEmbeddings(model="embed-english-v3.0")
    file_path = "data/romeo_and_juliet.txt"

    dm = DataManager(file_path, embedding)
    qdrant = dm.create_vector_store()

    r = Retriever(qdrant, "similarity", {"k": 3})
    g = Generator()

    while True:
        query = str(input("You: "))
        rel_docs = r.get_context(query)
        
        print("Answer: ", g.generate_answer(rel_docs, query))
