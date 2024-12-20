import os
import streamlit as st

from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.summarize import load_summarize_chain


def create_embeddings(openai_api_key):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key
    )

    return embeddings


def search_similar_documents(query, no_of_documents, index_name, embeddings):
    validation = query is not None and \
        no_of_documents >= 1 and \
        index_name is not None and \
        embeddings is not None

    if not validation:
        print("Invalid Arguments Specified!")

        return None

    vector_store = PineconeVectorStore(
        index_name=index_name, embedding=embeddings)
    similar_documents = vector_store.similarity_search_with_score(
        query, k=no_of_documents)

    return similar_documents


def get_summary_from_llm(resume_document):
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(
        temperature=0.1,
        max_tokens=512,
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo"
    )

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([resume_document])

    return summary


def main():
    try:
        load_dotenv()

        openai_api_key = os.environ["OPENAI_API_KEY"]
        index_name = os.environ["PINECONE_INDEX_NAME"]
        embeddings = create_embeddings(openai_api_key)

        st.set_page_config(
            page_title="Resume Analysis and Summarization", page_icon="📄")

        st.title("Resume Analysis and Summarization")
        st.subheader(
            """
                This AI assistant would help you to screen validate resumes with Deep Analytics!
            """
        )

        job_description = st.text_area(
            "Please enter your JD here ...", height=200)
        document_count = st.text_input(
            "Please enter the number of documents to retrieve ...", 5)
        submit = st.button("Analyze")

        if submit:
            relevant_documents = search_similar_documents(
                job_description, int(document_count), index_name, embeddings)

            for document_index in range(len(relevant_documents)):
                document, score = relevant_documents[document_index]

                st.subheader(f":sparkles: Document {
                             document_index + 1}, Score : {score}")
                st.write("*** FILE *** " +
                         document.metadata["source"])

                with st.expander("Show Summary ... "):
                    summary = get_summary_from_llm(document)
                    st.write("*** SUMMARY *** " + summary)
    except Exception as error:
        print(f"An Error Occurred: {error}")


if __name__ == "__main__":
    main()
