# from langchain.document_loaders import PyPDFLoader
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
from rerankers import Reranker
from langchain.retrievers import ContextualCompressionRetriever
import os
from dotenv import load_dotenv
from st_pages import add_page_title

add_page_title()

# Load environment variables from .env file
load_dotenv()
# IMPORTANT: Remember to create a .env variable containing: OPENAI_API_KEY=sk-xyz where xyz is your key
# Access the API key from the environment variable
st.sidebar.header("RAGTune")
os.environ['COHERE_API_KEY'] = os.environ.get("COHERE_API_KEY")

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
number_of_source_documents = st.slider('Select the number of source documents for retrieval:', min_value=2, max_value=10, value=4)
search_type = st.selectbox('Select the search type:', ('similarity', 'mmr'))
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
    # Create a retriever with the user-defined search_type and number_of_source_documents
    _retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": number_of_source_documents})
    st.session_state['_retriever'] = _retriever

st.subheader("Step 2: Select default LLM settings")

temperature = st.slider('Select the temperature for the language model:', min_value=0.0, max_value=2.0, value=0.7)
max_tokens = st.number_input('Enter the max tokens for the output:', min_value=0, max_value=4096, value=400)

# Retrieve the prompt from the hub
_prompt = hub.pull("rlm/rag-prompt")

st.write(
        """
        Prompt: You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
        """
        )

# Let's allow the user to select a single LLM and multiple rerankers to evaluate.
llm_options = {
    "Cohere - command-light": lambda: ChatCohere(model_name="command-light", temperature=temperature, max_tokens=max_tokens),
    "Cohere - command": lambda: ChatCohere(model_name="command", temperature=temperature, max_tokens=max_tokens),
    "Cohere - command-r": lambda: ChatCohere(model_name="command-r", temperature=temperature, max_tokens=max_tokens),
    "OpenAI - gpt-3.5-turbo": lambda: ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=max_tokens),
    "OpenAI - gpt-4-turbo-preview": lambda: ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=temperature, max_tokens=max_tokens),
    "OpenAI - gpt-4": lambda: ChatOpenAI(model_name="gpt-4", temperature=temperature, max_tokens=max_tokens),
    "Anthropic - claude-3-opus-20240229": lambda: ChatAnthropic(model_name="claude-3-opus-20240229", temperature=temperature, max_tokens=max_tokens),
    "Anthropic - claude-3-sonnet-20240229": lambda: ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=temperature, max_tokens=max_tokens),
    "Anthropic - claude-3-haiku-20240307": lambda: ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=temperature, max_tokens=max_tokens),
}

reranker_options = {
    "Cross-encoder default": lambda: Reranker('cross-encoder'),
    "MixedBread-AI Cross-encoder": lambda: Reranker('mixedbread-ai/mxbai-rerank-xlarge-v1', model_type='cross-encoder'),
    "Default T5 Seq2Seq reranker": lambda: Reranker("t5"),
    "InRanker-base T5 Seq2Seq reranker": lambda: Reranker("unicamp-dl/InRanker-base", model_type="t5"),
    "Cohere API reranker": lambda: Reranker("cohere", lang='en', api_key=os.environ['COHERE_API_KEY']),
    # "Jina API reranker": lambda: Reranker("jina", api_key=st.secrets["JINA_API_KEY"]),
    # "RankGPT4-turbo": lambda: Reranker("rankgpt", api_key=st.secrets["OPENAI_API_KEY"]),
    # "RankGPT3-turbo": lambda: Reranker("rankgpt3", api_key=st.secrets["OPENAI_API_KEY"]),
    "ColBERTv2 reranker": lambda: Reranker("colbert"),
}

# Ask the user to select the LLM and rerankers they want to evaluate.
st.subheader("Step 3: Select the LLM and rerankers to be evaluated")
selected_llm = st.selectbox('Select the LLM to evaluate:', options=list(llm_options.keys()))
selected_rerankers = st.multiselect('Select the rerankers to evaluate:', options=list(reranker_options.keys()))

@st.cache_data
def evaluate_rerankers(selected_llm, selected_rerankers, eval_questions, eval_answers, _retriever, _prompt):
    # We will store the results in a dictionary for visualization later.
    reranker_results = {}

    # Instantiate the selected LLM
    llm = llm_options[selected_llm]()

    # Iterate over each selected reranker and perform the evaluation.
    for reranker_name in selected_rerankers:
        ranker = reranker_options[reranker_name]()
        compressor = ranker.as_langchain_compressor(k=3)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=_retriever)

        rag_chain = (
            {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
            | _prompt
            | llm
            | StrOutputParser()
        )

        # Run the RAG chain for each question and collect answers and contexts.
        answers = []
        contexts = []
        for question in eval_questions:
            response = rag_chain.invoke(question)
            answers.append(response)
            retrieved_docs = compression_retriever.invoke(question)
            contexts.append([context.page_content for context in retrieved_docs])

        # Create a Hugging Face dataset from the responses.
        from datasets import Dataset
        response_dataset = Dataset.from_dict({
            "question": eval_questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": eval_answers
        })

        # Evaluate the dataset using the specified metrics.
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

        # Convert the results to a pandas DataFrame for easier visualization.
        df_results = result.to_pandas()
        reranker_results[reranker_name] = {
            "overview": result,
            "details": df_results
        }

    return reranker_results

# Add an "Evaluate" button that the user must press to run the evaluation.
if st.button('Evaluate'):
    # Function to format the documents for the RAG chain.
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    if 'reranker_results' not in st.session_state:
        st.session_state['reranker_results'] = {}
    st.session_state['reranker_results'] = evaluate_rerankers(selected_llm, selected_rerankers, st.session_state['eval_questions'], st.session_state['eval_answers'], st.session_state['_retriever'], _prompt)
    
if 'reranker_results' in st.session_state and st.session_state['reranker_results']:
    # Now, let's visualize the results for each reranker.
    with st.expander("See Evaluation Results"):
        for reranker_name, results in st.session_state['reranker_results'].items():
            st.write(f"Overview of Results for {reranker_name}:")
            st.write(results["overview"])
            st.write(f"Details of Results for {reranker_name}:")
            st.dataframe(results["details"], use_container_width=True)
else:
    st.warning("No results available. Please run the evaluation first.")

st.subheader("Step 4: Visualize Data")

if st.button('Prepare Charts'):

    if 'reranker_results' in st.session_state:
        results = st.session_state['reranker_results']

        # Convert the results dictionary to a pandas DataFrame
        data = []
        for reranker_name, reranker_result in results.items():
            result = reranker_result["overview"]
            for metric_name, metric_value in result.items():
                data.append({
                    "Reranker": reranker_name,
                    "Metric": metric_name,
                    "Value": metric_value
                })

        visual_df = pd.DataFrame(data)
        st.session_state["visual_df"] = visual_df
    
    else:
        st.warning("No results available. Please run the evaluation first.")


if 'visual_df' in st.session_state and not st.session_state['visual_df'].empty:
    # Now, let's visualize the results for each reranker.
    with st.expander("See Visualization Results"):
        # Side-by-Side Bar Charts
        st.subheader("Side-by-Side Bar Charts")
        
        import plotly.express as px
        # Create the grouped bar chart with Plotly
        fig = px.bar(st.session_state["visual_df"], x="Metric", y="Value", color="Reranker", barmode='group', height=400)
        # Display the figure in the Streamlit app
        st.plotly_chart(fig)

        # Overlaid Line Charts
        st.subheader("Overlaid Line Charts")
        line_chart = st.line_chart(st.session_state["visual_df"].pivot(index='Metric', columns='Reranker', values='Value'))

        st.write("Dataframe for download")
        st.session_state["visual_df"]

else:
    st.warning("No plots available. Please run the 'Prepare Charts' step first.")