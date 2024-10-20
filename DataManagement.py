from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document

class DataManager:
    def __init__(self, content, embedding, file_name):
        self.file_name = file_name
        self.content = content
        self.embedding = embedding
    
    def load_content(self):
        metadata = {"source": self.file_name}
        doc = Document(page_content=self.content, metadata=metadata)

        return [doc]
    
    def split_text(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)

        doc_content = self.load_content()
        
        return text_splitter.split_documents(doc_content)
    
    def modify_vector_store(self, mode):
        embeddings = self.embedding

        doc_splitted = self.split_text()
        
        if mode == "create":
            qdrant = Qdrant.from_documents(
                doc_splitted,
                embeddings,
                location=":memory:",
                # path="tmp/local_qdrant",
                collection_name="my_cv",
                force_recreate=True
            )
        else:
            qdrant = Qdrant.add_documents(
                doc_splitted,
                embeddings,
                path="tmp/local_qdrant",
                collection_name="my_cv")

        return qdrant