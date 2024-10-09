from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant

class DataManager:
    def __init__(self, file_path, embedding):
        self.file_path = file_path
        self.embedding = embedding
    
    def load_file(self):
        loader = TextLoader(self.file_path)

        return loader.load()
    
    def split_text(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)

        doc_content = self.load_file()

        return text_splitter.split_documents(doc_content)
    
    def create_vector_store(self):
        embeddings = self.embedding

        doc_splitted = self.split_text()

        qdrant = Qdrant.from_documents(
            doc_splitted,
            embeddings,
            path="tmp/local_qdrant", # Location to store vector db index
            collection_name="my_cv",
            force_recreate=True
        )

        return qdrant