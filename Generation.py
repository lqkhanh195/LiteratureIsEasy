from langchain import hub

class Generator:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = hub.pull("rlm/rag-prompt")

    def parse_output(self, output):
        return output.content

    def generate_answer(self, rel_docs, query):
        input = {"context": "\n\n".join(doc.page_content for doc in rel_docs),
                 "question": query}
        
        rag_chain = self.prompt | self.llm | self.parse_output
        return rag_chain.invoke(input) 