import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatCohere
from langchain_community.embeddings import CohereEmbeddings
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_anthropic import ChatAnthropic
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    answer_similarity,
    answer_correctness,
    context_precision,
    context_recall,
    context_relevancy,
)
import pandas as pd
import random
import string
import plotly.express as px

from st_pages import add_page_title

add_page_title()

st.sidebar.header("RAGTune")

def generate_random_string(length=10):
    """Generate a random string of alphanumeric characters"""
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

@st.cache_data
def create_embedding_and_evaluate(parameter_name, chunk_size, chunk_overlap, number_of_source_documents, search_type, embeddings_option, splitter_type):
    if parameter_name == "Chunk Size":
        parameter_value = chunk_size
    elif parameter_name == "Chunk Overlap":
        parameter_value = chunk_overlap
    elif parameter_name == "Number of Source Documents":
        parameter_value = number_of_source_documents
    elif parameter_name == "Search Type":
        parameter_value = search_type
    elif parameter_name == "Embeddings Option":
        parameter_value = embeddings_option
    elif parameter_name == "Splitter":
        parameter_value = splitter_type

    if embeddings_option == 'CohereEmbeddings':
        embeddings = CohereEmbeddings()
    elif embeddings_option == 'OpenAIEmbeddings':
        embeddings = OpenAIEmbeddings()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    if splitter_type == 'RecursiveCharacterTextSplitter':
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == 'CharacterTextSplitter':
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == 'TokenTextSplitter':
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = loader.load()
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db", collection_name=generate_random_string())
    _retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": number_of_source_documents})

    _prompt = hub.pull("rlm/rag-prompt")

    llm = llm_options[default_llm]()
    rag_chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()}
        | _prompt
        | llm
        | StrOutputParser()
    )

    answers = []
    contexts = []
    for question in eval_questions:
        response = rag_chain.invoke(question)
        answers.append(response)
        retrieved_docs = _retriever.invoke(question)
        contexts.append([context.page_content for context in retrieved_docs])

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
            context_precision,
            context_recall,
            context_relevancy,
        ],
    )

    df_results = result.to_pandas()
    return {
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "overview": result,
        "details": df_results
    }

# @st.cache_data
def display_embedding_evaluation_results(current_parameter_name):
    if "embedding_results" in st.session_state and st.session_state["embedding_results"]:
        st.subheader("Step 3: View Embedding Evaluation Results")
        with st.expander("See Evaluation Results"):
            for result in st.session_state["embedding_results"]:
                # st.write(result)
                if result['parameter_name'] == current_parameter_name:
                    st.write(f"Evaluation for {result['parameter_name']}: {result['parameter_value']}")
                    st.write("Overview of Results:")
                    st.write(result["overview"])
                    st.write("Details of Results:")
                    st.dataframe(result["details"], use_container_width=True)
# @st.cache_data
def prepare_charts(current_parameter_name, is_barchart=False):
    st.subheader("Step 4: Visualize Data")
    data = []
    for result in st.session_state["embedding_results"]:
        if result['parameter_name'] == current_parameter_name:
            overview = result["overview"]
            parameter_name = result["parameter_name"]
            parameter_value = result["parameter_value"]
            for metric_name, metric_value in overview.items():
                data.append({
                    "Parameter": parameter_name,
                    "Parameter Value": parameter_value,
                    "Metric": metric_name,
                    "Value": metric_value
                })

    if data:
        visual_df = pd.DataFrame(data)
        st.session_state["visual_df"] = visual_df
    else:
        st.warning(f"No data available for parameter '{current_parameter_name}'. Please run the evaluation for this parameter.")

    if 'visual_df' in st.session_state and not st.session_state['visual_df'].empty:
        with st.expander("See Visualization Results"):
            if is_barchart:
                st.subheader("Grouped Bar Chart")
                fig = px.bar(visual_df, x="Metric", y="Value", color="Parameter Value", barmode='group', height=400)
                st.plotly_chart(fig)
            else:
                st.subheader("Scatter Charts")
                for metric in st.session_state["visual_df"]["Metric"].unique():
                    fig = px.scatter(st.session_state["visual_df"][st.session_state["visual_df"]["Metric"] == metric], 
                                     x="Parameter Value", y="Value", title=f"{current_parameter_name} vs {metric}",
                                     labels={"Parameter Value": current_parameter_name, "Value": metric})
                    st.plotly_chart(fig)

            st.write("Dataframe for download")
            st.dataframe(st.session_state["visual_df"], use_container_width=True)
    else:
        st.warning("No plots available. Please run the 'Prepare Charts' step first.")



# Load data from session state
if 'eval_questions' not in st.session_state or 'eval_answers' not in st.session_state or 'doc_path' not in st.session_state:
    st.warning("Please upload a document and save eval questions and answers")
else:
    eval_questions = st.session_state['eval_questions']
    eval_answers = st.session_state['eval_answers']
    doc_path = st.session_state['doc_path']

    loader = DirectoryLoader(doc_path, show_progress=True, use_multithreading=True)

    # Step 1: Select default LLM model
    st.subheader("Step 1: Select default LLM settings")
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

    # Step 2: Set ranges for embedding options
    st.subheader("Step 2: Select a tab below for evaluation")


    # Create tabs for each embedding option
    tab_chunk_size, tab_chunk_overlap, tab_number_of_source_documents, tab_search_type, tab_embeddings_option, tab_splitter = st.tabs(["Chunk Size", "Chunk Overlap", "Number of Source Documents", "Search Type", "Embeddings Option", "Splitter"])

    # Chunk Size tab
    with tab_chunk_size:
        current_parameter_name  = "Chunk Size"
        st.session_state["parameter_name"] = current_parameter_name
        
        st.write("Configure the varying parameter")
        chunk_size = st.select_slider(
            'Select the range for chunk size for text splitting:',
            options=list(range(50, 5001, 50)),
            value=(50, 5000),
            key = current_parameter_name
        )
        chunk_size_min, chunk_size_max = chunk_size
        chunk_size_data_points = st.slider('Number of data points to collect between chunk size range:', min_value=2, max_value=10, value=3, step=1)
        
        st.divider()
        st.write("Set the constant parameters")
        chunk_overlap = st.number_input('Enter the chunk overlap for text splitting:', min_value=0, value=50, key = current_parameter_name+'_overlap')
        number_of_source_documents = st.slider('Select the number of source documents for retrieval:', min_value=2, max_value=10, value=4, key = current_parameter_name+'_k')
        search_type = st.selectbox('Select the search type:', ('similarity', 'mmr'), key = current_parameter_name+'_search_type')
        embeddings_option = st.selectbox(
            'Select the embeddings to use:',
            ('CohereEmbeddings', 'OpenAIEmbeddings'),
            key=current_parameter_name+'_embedding'
        )
        splitter_type = st.selectbox(
            'Select the text splitter type:',
            ('RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'TokenTextSplitter'),
            key=current_parameter_name+'_splitter'
        )

        if st.button('Evaluate Chunk Size'):

            

            chunk_size_step = ( chunk_size_max - chunk_size_min + 1) // chunk_size_data_points
            chunk_size_step = max(chunk_size_step-1 , 1) # Adjust step size to ensure it's at least 1
            chunk_size_range = range(chunk_size_min, chunk_size_max + 1, chunk_size_step)
            embedding_results = []
            st.session_state["embedding_results"] = embedding_results

            for chunk_size in chunk_size_range:
                result = create_embedding_and_evaluate(current_parameter_name, chunk_size, chunk_overlap, number_of_source_documents, search_type, embeddings_option, splitter_type)
                embedding_results.append(result)
            
            display_embedding_evaluation_results(current_parameter_name)
            prepare_charts(current_parameter_name)


            st.session_state["embedding_results"] = embedding_results

    # Chunk Overlap tab
    with tab_chunk_overlap:
        current_parameter_name = "Chunk Overlap"
        st.session_state["parameter_name"] = current_parameter_name

        st.write("Configure the varying parameter")
        chunk_overlap = st.select_slider(
            'Select the range for chunk overlap for text splitting:',
            options=list(range(0, 501, 20)),
            value=(0, 500),
            key = current_parameter_name
        )
        chunk_overlap_min, chunk_overlap_max = chunk_overlap
        chunk_overlap_data_points = st.slider('Number of data points to collect between chunk overlap range:', min_value=2, max_value=10, value=3, step=1)

        st.divider()
        st.write("Set the constant parameters")
        chunk_size = st.number_input('Enter the chunk size for text splitting:', min_value=50, value=1000, key = current_parameter_name+'_size')
        number_of_source_documents = st.slider('Select the number of source documents for retrieval:', min_value=2, max_value=10, value=4, key = current_parameter_name+'_k')
        search_type = st.selectbox('Select the search type:', ('similarity', 'mmr'), key = current_parameter_name+'_search_type')
        embeddings_option = st.selectbox('Select the embeddings to use:', ('CohereEmbeddings', 'OpenAIEmbeddings'), key = current_parameter_name+'_embedding')
        splitter_type = st.selectbox(
            'Select the text splitter type:',
            ('RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'TokenTextSplitter'),
            key=current_parameter_name+'_splitter'
        )


        if st.button('Evaluate Chunk Overlap'):
            chunk_overlap_step = (chunk_overlap_max - chunk_overlap_min + 1) // chunk_overlap_data_points
            chunk_overlap_step = max(chunk_overlap_step, 1)  # Adjust step size to ensure it's at least 1
            chunk_overlap_range = range(chunk_overlap_min, chunk_overlap_max + 1, chunk_overlap_step)
            embedding_results = []
            st.session_state["embedding_results"] = embedding_results

            for chunk_overlap in chunk_overlap_range:
                result = create_embedding_and_evaluate(current_parameter_name, chunk_size, chunk_overlap,  number_of_source_documents, search_type, embeddings_option, splitter_type)
                embedding_results.append(result)

            display_embedding_evaluation_results(current_parameter_name)
            prepare_charts(current_parameter_name)

            st.session_state["embedding_results"] = embedding_results

    # Number of Source Documents tab
    with tab_number_of_source_documents:
        current_parameter_name = "Number of Source Documents"
        st.session_state["parameter_name"] = current_parameter_name

        st.write("Configure the varying parameter")
        number_of_source_documents = st.select_slider(
            'Select the range for number of source documents for retrieval:',
            options=list(range(2, 11, 1)),
            value=(2, 10),
            key = current_parameter_name
        )
        number_of_source_documents_min, number_of_source_documents_max = number_of_source_documents
        number_of_source_documents_data_points = st.slider('Number of data points to collect between number of source documents range:', min_value=2, max_value=10, value=3, step=1)

        st.divider()
        st.write("Set the constant parameters")
        chunk_size = st.number_input('Enter the chunk size for text splitting:', min_value=50, value=1000, key = current_parameter_name+'_size')
        chunk_overlap = st.number_input('Enter the chunk overlap for text splitting:', min_value=0, value=50, key = current_parameter_name+'_overlap')
        search_type = st.selectbox('Select the search type:', ('similarity', 'mmr'), key = current_parameter_name+'_search_type')
        embeddings_option = st.selectbox('Select the embeddings to use:', ('CohereEmbeddings', 'OpenAIEmbeddings'), key = current_parameter_name+'_embedding')
        splitter_type = st.selectbox(
            'Select the text splitter type:',
            ('RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'TokenTextSplitter'),
            key=current_parameter_name+'_splitter'
        )
        if st.button('Evaluate Number of Source Documents'):
            number_of_source_documents_step = (number_of_source_documents_max - number_of_source_documents_min + 1) // number_of_source_documents_data_points
            number_of_source_documents_step = max(number_of_source_documents_step, 1)  # Adjust step size to ensure it's at least 1
            number_of_source_documents_range = range(number_of_source_documents_min, number_of_source_documents_max + 1, number_of_source_documents_step)
            embedding_results = []
            st.session_state["embedding_results"] = embedding_results

            for number_of_source_documents in number_of_source_documents_range:
                result = create_embedding_and_evaluate(current_parameter_name, chunk_size, chunk_overlap,  number_of_source_documents,  search_type, embeddings_option, splitter_type)
                embedding_results.append(result)

            display_embedding_evaluation_results(current_parameter_name)
            prepare_charts(current_parameter_name)

            st.session_state["embedding_results"] = embedding_results

    # Search Type tab
    with tab_search_type:
        current_parameter_name = "Search Type"
        st.session_state["parameter_name"] = current_parameter_name

        st.write("Configure the varying parameter")
        search_types = st.multiselect('Select the search types:', ['similarity', 'mmr'],key = current_parameter_name)

        st.divider()
        st.write("Set the constant parameters")
        chunk_size = st.number_input('Enter the chunk size for text splitting:', min_value=50, value=1000, key = current_parameter_name+'_size')
        chunk_overlap = st.number_input('Enter the chunk overlap for text splitting:', min_value=0, value=50, key = current_parameter_name+'_overlap')
        number_of_source_documents = st.slider('Select the number of source documents for retrieval:', min_value=2, max_value=10, value=4, key = current_parameter_name+'_k')
        embeddings_option = st.selectbox('Select the embeddings to use:', ('CohereEmbeddings', 'OpenAIEmbeddings'), key = current_parameter_name+'_embedding')
        splitter_type = st.selectbox(
            'Select the text splitter type:',
            ('RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'TokenTextSplitter'),
            key=current_parameter_name+'_splitter'
        )
        if st.button('Evaluate Search Type'):
            embedding_results = []
            st.session_state["embedding_results"] = embedding_results

            for search_type in search_types:
                result = create_embedding_and_evaluate(current_parameter_name, chunk_size, chunk_overlap,  number_of_source_documents, search_type,  embeddings_option, splitter_type)
                embedding_results.append(result)

            display_embedding_evaluation_results(current_parameter_name)
            prepare_charts(current_parameter_name)

            st.session_state["embedding_results"] = embedding_results

    # Embeddings Option tab
    with tab_embeddings_option:
        current_parameter_name = "Embeddings Option"
        st.session_state["parameter_name"] = current_parameter_name
        
        st.write("Configure the varying parameter")
        embeddings_options = st.multiselect('Select the embeddings to use:', ['CohereEmbeddings', 'OpenAIEmbeddings'],key = current_parameter_name)

        st.divider()
        st.write("Set the constant parameters")
        chunk_size = st.number_input('Enter the chunk size for text splitting:', min_value=50, value=1000, key = current_parameter_name+'_size')
        chunk_overlap = st.number_input('Enter the chunk overlap for text splitting:', min_value=0, value=50, key = current_parameter_name+'_overlap')
        number_of_source_documents = st.slider('Select the number of source documents for retrieval:', min_value=2, max_value=10, value=4, key = current_parameter_name+'_k')
        search_type = st.selectbox('Select the search type:', ('similarity', 'mmr'), key = current_parameter_name+'_search_type')
        splitter_type = st.selectbox(
            'Select the text splitter type:',
            ('RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'TokenTextSplitter'),
            key=current_parameter_name+'_splitter'
        )
        if st.button('Evaluate Embeddings Option'):
            embedding_results = []
            st.session_state["embedding_results"] = embedding_results
            for embeddings_option in embeddings_options:
                result = create_embedding_and_evaluate(current_parameter_name,chunk_size , chunk_overlap,number_of_source_documents, search_type,embeddings_option , splitter_type )
                embedding_results.append(result)

            display_embedding_evaluation_results(current_parameter_name)
            prepare_charts(current_parameter_name, is_barchart=True)

            st.session_state["embedding_results"] = embedding_results

    with tab_splitter:
        current_parameter_name = "Splitter"
        st.session_state["parameter_name"] = current_parameter_name

        st.write("Configure the varying parameter")
        splitter_options = st.multiselect('Select the text splitters:', ['RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'TokenTextSplitter'], default=['RecursiveCharacterTextSplitter'])
        
        st.divider()
        st.write("Set the constant parameters")
        chunk_size = st.number_input('Enter the chunk size for text splitting:', min_value=50, value=200, key=current_parameter_name+'_chunk_size')
        chunk_overlap = st.number_input('Enter the chunk overlap for text splitting:', min_value=0, value=50, key=current_parameter_name+'_overlap')
        number_of_source_documents = st.slider('Select the number of source documents for retrieval:', min_value=2, max_value=10, value=4, key=current_parameter_name+'_k')
        search_type = st.selectbox('Select the search type:', ('similarity', 'mmr'), key=current_parameter_name+'_search_type')
        embeddings_option = st.selectbox('Select the embeddings to use:', ('CohereEmbeddings', 'OpenAIEmbeddings'), key=current_parameter_name+'_embedding')

        if st.button('Evaluate Splitters'):
            embedding_results = []
            st.session_state["embedding_results"] = embedding_results

            for splitter in splitter_options:
                result = create_embedding_and_evaluate(current_parameter_name, chunk_size, chunk_overlap, number_of_source_documents, search_type, embeddings_option, splitter)
                embedding_results.append(result)

            display_embedding_evaluation_results(current_parameter_name)
            prepare_charts(current_parameter_name, is_barchart=True)

            st.session_state["embedding_results"] = embedding_results

# Visualization code

# current_parameter_name = st.session_state.get("parameter_name", None)

# Call the function with the current parameter name
# st.session_state["embedding_results"]