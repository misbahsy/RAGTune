import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
# Load environment variables from .env file
load_dotenv()
# IMPORTANT: Remember to create a .env variable containing: OPENAI_API_KEY=sk-xyz where xyz is your key

# Access the API key from the environment variable
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
os.environ['COHERE_API_KEY'] = os.environ.get("COHERE_API_KEY")
os.environ['ANTHROPIC_API_KEY'] = os.environ.get("ANTHROPIC_API_KEY")


from st_pages import Page, show_pages, add_page_title
st.sidebar.header("RAGTune")

# # Optional -- adds the title and icon to the current page
add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles and icons
show_pages(
    [
        Page("Home.py", "Upload Document and Assign Dataset"),
        Page("pages/1_LLM.py", "Evaluate LLM Models"),
        Page("pages/2_embeddings.py", "Evaluate Embeddings"),
        Page("pages/3_query_tranformations.py", "Evaluate Query Transformations"),
        Page("pages/4_rerankers.py", "Evaluate Rerankers"),
        # Page("pages/5_prompt_optimizer.py", "Prompt Optimization using DSPy"), coming soon

    ]
)

# Initialize doc_path with a default value
doc_path = "docs/constitution.pdf"

# Initialize session state keys if they don't exist
if 'eval_questions' not in st.session_state:
    st.session_state['eval_questions'] = []
if 'eval_answers' not in st.session_state:
    st.session_state['eval_answers'] = []


# Check if the user wants to use the default document or upload their own
st.header('Document Selection')
document_option = st.radio("Choose your document source", ('Upload a file', 'Use default test document'))

if document_option == 'Upload a file':
    st.session_state['eval_questions'] = [""]
    st.session_state['eval_answers'] = [""]
    # Allow multiple files to be uploaded including pdf, csv, doc, docx, ppt, pptx
    uploaded_files = st.file_uploader("Choose files", type=['pdf', 'csv', 'doc', 'docx', 'ppt', 'pptx'], accept_multiple_files=True)
    if uploaded_files:
        # Ensure the 'uploaded_docs' directory exists before saving the files
        upload_dir = "uploaded_docs"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        # Save the uploaded files and collect their paths
        for uploaded_file in uploaded_files:
            with open(os.path.join(upload_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Update session state with the directory name of uploaded documents
        st.session_state['doc_path'] = upload_dir

        # User input for eval_questions and eval_answers
        st.subheader('Provide Evaluation Questions and Answers')

        data = {
            'Questions': st.session_state['eval_questions'],
            'Ground Truth': st.session_state['eval_answers']
        }
        qa_df = pd.DataFrame(data)
        edited_qa_df = st.data_editor(data, num_rows="dynamic", use_container_width=True, hide_index=True)

        eval_questions_list = edited_qa_df['Questions']
        eval_answers_list = edited_qa_df['Ground Truth']

        if st.button("Save eval Q&As"):
            # Check if the number of questions matches the number of answers
            st.session_state['eval_questions'] = eval_questions_list
            st.session_state['eval_answers'] = eval_answers_list
            st.success("Evaluation questions and answers saved successfully!")

else:
    # Use the default document
    doc_path = "docs"
    st.write("Using the default document: Constitution.pdf")

    # Default eval_questions and eval_answers
    eval_questions = [
        "what is article I of the constitution of the US about?",
        "How many sections does ARTICLE. IV have?",
        "Who is elegible to be the President of the US?",
        "What majority is needed to amend the constitution",
        "How many states are sufficient for ratification of the constitution?",
    ]
    eval_answers = [
        "Article I of the United States Constitution establishes the legislative branch of the federal government, known as the United States Congress. It outlines that all legislative powers are vested in Congress, which is divided into two parts: the House of Representatives and the Senate. The bicameral Congress was created as a compromise between large and small states, with representation based on population and equal representation for states. Article I grants Congress enumerated powers and the authority to pass laws necessary for carrying out those powers. It also sets out procedures for passing bills and imposes limits on Congress's authority. Additionally, Article I's Vesting Clause ensures that all federal legislative power belongs to Congress, emphasizing the separation of powers among the three branches of government",
        "4 Sections",
        "No Person except a natural born Citizen, or a Citizen of the United States, at the time of the Adoption of this Constitution, shall be eligible to the Office of President; neither shall any Per- son be eligible to that Office who shall not have attained to the Age of thirty five Years, and been fourteen Years a Resident within the United States.",
        "The Congress, whenever two thirds of both Houses shall deem it necessary, shall propose Amendments to this Constitution, or, on the Ap- plication of the Legislatures of two thirds of the several States, shall call a Convention for pro- posing Amendments",
        "The Ratification of the Conventions of nine States, shall be sufficient for the Establishment of this Constitution between the States so rati- fying the Same.",
    ]

    # Assign the default questions and answers to the state
    st.session_state['eval_questions'] = eval_questions
    st.session_state['eval_answers'] = eval_answers
    st.session_state['doc_path'] = doc_path

# Display eval questions and answers if available
if st.session_state.get('eval_questions') and st.session_state.get('eval_answers'):
    st.subheader('Saved Evaluation Questions and Answers')
    # Convert eval_questions and eval_answers to a DataFrame and display it
    eval_qa_df = pd.DataFrame({
        'Questions': st.session_state['eval_questions'],
        'Ground Truth': st.session_state['eval_answers']
    })
    st.dataframe(eval_qa_df, use_container_width=True , hide_index=True)
    if len(eval_qa_df["Questions"]) >= 4:
        st.subheader('Proceed to one of the tabs on the left to perform Evaluations')
        st.page_link("pages/1_LLM.py", label="LLM")
        st.page_link("pages/2_embeddings.py", label="Embeddings")
        st.page_link("pages/3_query_tranformations.py", label="Query Tranformations")
        st.page_link("pages/4_rerankers.py", label="Rerankers")

    else:
        st.warning('Please add at least 4 rows of data for evaluation')
    
else:
    st.header('No evaluation questions and answers provided.')
