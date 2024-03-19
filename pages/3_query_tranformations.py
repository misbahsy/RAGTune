from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatCohere
from langchain_community.embeddings import CohereEmbeddings
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_anthropic import ChatAnthropic
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_similarity,
    answer_correctness,
)
from langchain_community.document_loaders import DirectoryLoader
import pandas as pd
import random
import string
import streamlit as st
import pandas as pd
from utils import *
from st_pages import add_page_title

add_page_title()

st.sidebar.header("RAGTune")

if 'eval_questions' not in st.session_state or 'eval_answers' not in st.session_state or 'doc_path' not in st.session_state:
    st.warning("Please upload a document and save eval questions and answers")

# Generating random string to be used as collection name in chroma to avoid embedding dimensions error
def generate_random_string(length=10):
    """Generate a random string of alphanumeric characters"""
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

# First, we need to import Streamlit to access the session state.
eval_questions  =  st.session_state['eval_questions']
eval_answers  =  st.session_state['eval_answers']
doc_path  =  st.session_state['doc_path']

# Now, we use the 'doc_path' from the session state to load the document.
# We assume that 'doc_path' has been set in the session state in the Home.py file.
loader = DirectoryLoader(doc_path, show_progress=True, use_multithreading=True)


# Ask the user for input values for chunk_size, chunk_overlap, number_of_source_documents, search_type, temperature, and embeddings
st.subheader("Step 1: Generate embeddings")
chunk_size = st.number_input('Enter the chunk size for text splitting:', min_value=1, value=1000)
chunk_overlap = st.number_input('Enter the chunk overlap for text splitting:', min_value=0, value=200)
# number_of_source_documents = st.slider('Select the number of source documents for retrieval:', min_value=2, max_value=10, value=4)
# search_type = st.selectbox('Select the search type:', ('similarity', 'mmr'))
embeddings_option = st.selectbox('Select the embeddings to use:', ('CohereEmbeddings', 'OpenAIEmbeddings'))

# Based on the user's choice of embeddings, instantiate the appropriate embeddings class
if embeddings_option == 'CohereEmbeddings':
    embeddings = CohereEmbeddings()
elif embeddings_option == 'OpenAIEmbeddings':
    embeddings = OpenAIEmbeddings()

# Create a text splitter with the user-defined chunk_size and chunk_overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)



if st.button('Generate Embeddings'):
    docs = loader.load()
    splits = text_splitter.split_documents(docs)
    # Create a vectorstore with the user-defined embeddings
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db", collection_name=generate_random_string())

    # _retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": number_of_source_documents})
    _retriever = vectorstore.as_retriever()
    st.session_state['_retriever'] = _retriever

st.subheader("Step 2: Select LLM settings")
llm_options = {
    "Cohere - command-light": lambda: ChatCohere(temperature=temperature, max_tokens=max_tokens),
    "Cohere - command": lambda: ChatCohere(temperature=temperature, max_tokens=max_tokens),
    "Cohere - command-r": lambda: ChatCohere(temperature=temperature, max_tokens=max_tokens),
    "OpenAI - gpt-3.5-turbo": lambda: ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=max_tokens),
    "OpenAI - gpt-4-turbo-preview": lambda: ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=temperature, max_tokens=max_tokens),
    "OpenAI - gpt-4": lambda: ChatOpenAI(model_name="gpt-4", temperature=temperature, max_tokens=max_tokens),
    "Anthropic - claude-3-opus-20240229": lambda: ChatAnthropic(model_name="claude-3-opus-20240229", temperature=temperature, max_tokens=max_tokens),
    "Anthropic - claude-3-sonnet-20240229": lambda: ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=temperature, max_tokens=max_tokens),
    "Anthropic - claude-3-haiku-20240307": lambda: ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=temperature, max_tokens=max_tokens),
}
default_llm = st.selectbox('Select the default LLM model:', options=list(llm_options.keys()))
temperature = st.slider('Select the temperature for the language model:', min_value=0.0, max_value=2.0, value=0.7)
max_tokens = st.number_input('Enter the max tokens for the output:', min_value=0, max_value=4096, value=400)

llm = llm_options[default_llm]()

# st.subheader("Step 3: Select Query ")

# Define a dictionary mapping query transformation names to their respective functions
query_transformation_options = {
    'Multi Query': run_multi_query,
    'RAG Fusion': run_rag_fusion,
    'Decomposition Recursive': run_recursive_decomposition,
    'Decomposition Individual': run_individual_decomposition,
    'Step Back': run_step_back_rag,
    'HyDE': run_hyde,
}

st.subheader("Step 3: Select query transformations")
selected_transformations = st.multiselect('Select the query transformations to evaluate:', options=list(query_transformation_options.keys()))

# Add an "Evaluate" button
if st.button('Evaluate'):
    results = {}
    _retriever = st.session_state['_retriever']
    for transformation_name in selected_transformations:
        transformation_func = query_transformation_options[transformation_name]
        answers, contexts = [], []
        for question in eval_questions:
            # queries = transformation_func.invoke({"question": question})
            # retrieved_docs = _retriever.get_relevant_documents(queries)
            # answer = generate_answer(question, retrieved_docs)
            answer, retrieved_docs = transformation_func(llm, _retriever, question)
            # print("answer", answer)
            # print(transformation_name,retrieved_docs)
            answers.append(answer)
            contexts.append([doc.page_content for doc in retrieved_docs])

        from datasets import Dataset
        response_dataset = Dataset.from_dict({
            "question": eval_questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": eval_answers
        })

        from ragas import evaluate
        result = evaluate(
            response_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                answer_similarity,
                answer_correctness,
                context_recall,
                context_precision,
                context_relevancy,
            ],
        )

        df_results = result.to_pandas()
        results[transformation_name] = {
            "overview": result,
            "details": df_results
        }

    st.session_state['query_transformation_results'] = results

if 'query_transformation_results' in st.session_state and st.session_state['query_transformation_results']:
    with st.expander("See Evaluation Results"):
        for transformation_name, results in st.session_state['query_transformation_results'].items():
            st.write(f"Overview of Results for {transformation_name}:")
            st.write(results["overview"])
            st.write(f"Details of Results for {transformation_name}:")
            st.dataframe(results["details"], use_container_width=True)
else:
    st.warning("No results available. Please run the evaluation first.")

st.subheader("Step 4: Visualize Data")

if st.button('Prepare Charts'):
    if 'query_transformation_results' in st.session_state:
        results = st.session_state['query_transformation_results']

        data = []
        for transformation_name, transformation_result in results.items():
            result = transformation_result["overview"]
            for metric_name, metric_value in result.items():
                data.append({
                    "Transformation": transformation_name,
                    "Metric": metric_name,
                    "Value": metric_value
                })

        visual_df = pd.DataFrame(data)
        st.session_state["visual_df"] = visual_df
    else:
        st.warning("No results available. Please run the evaluation first.")

if 'visual_df' in st.session_state and not st.session_state['visual_df'].empty:
    with st.expander("See Visualization Results"):
        import plotly.express as px
        st.subheader("Side-by-Side Bar Charts")
        fig = px.bar(st.session_state["visual_df"], x="Metric", y="Value", color="Transformation", barmode='group', height=400)
        st.plotly_chart(fig)

        st.subheader("Overlaid Line Charts")
        line_chart = st.line_chart(st.session_state["visual_df"].pivot(index='Metric', columns='Transformation', values='Value'))

        st.write("Dataframe for download")
        st.session_state["visual_df"]
else:
    st.warning("No plots available. Please run the 'Prepare Charts' step first.")