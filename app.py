import os
from io import StringIO
from dotenv import load_dotenv

import streamlit as st
from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain_core.messages import AIMessage, HumanMessage

from DataManagement import DataManager
from HistoryAdding import HistoryAdder
from Retrievial import Retriever
from Generation import Generator

load_dotenv()

llm = ChatCohere()
embedding = CohereEmbeddings(model="embed-english-v3.0")
g = Generator(llm)
ha = HistoryAdder(llm)

st.title("Let's learn literature together !!!")

# BUG:
# Token limit exceed
# Prompt in is None

if "hist" not in st.session_state:
    st.session_state.hist = []

if "disable_input" not in st.session_state:
    st.session_state.disable_input = False

if "processed_file" not in st.session_state:
    st.session_state.processed_file = []

if "db" not in st.session_state:
    st.session_state.db = None

if "hist_for_context" not in st.session_state:
    st.session_state.hist_for_context = []

def main():
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a file", type=["txt"])

        # if uploaded_file[i].name in st.session_state.processed_file:
        #     continue
        if uploaded_file:
            with st.spinner('Processing document...'):
                file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

                dm = DataManager(file_content, embedding, uploaded_file.name)
                st.session_state.db = dm.modify_vector_store("create")

                st.session_state.processed_file.append(uploaded_file.name)

    if st.session_state.db is None:
        st.info("Upload at least 1 file please")
    else:
        r = Retriever(st.session_state.db, "similarity", {"k": 2})

        for chat in st.session_state.hist:
                with st.chat_message(chat["role"]):
                    st.markdown(chat["content"])  

        prompt = st.chat_input("Nhập câu hỏi của bạn tại đây")

        if prompt is not None and prompt.strip != "":
            st.session_state.hist.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):    
                st.session_state.disable_input = True
                holder = st.empty()
                holder.markdown("Let me think...")
                
                query_aware_history = ha.get_hist_context(st.session_state.hist_for_context, prompt)
                rel_docs = r.get_context(query_aware_history)

                res = g.generate_answer(rel_docs, query_aware_history)

                holder.markdown(res)
                st.session_state.disable_input = False

            st.session_state.hist.append(
                {
                    "role": "assistant",
                    "content": res
                }
            )

            st.session_state.hist_for_context.extend(
            [
                HumanMessage(content=prompt),
                AIMessage(content=res),
            ]
            )


if __name__ == "__main__":
    main()

