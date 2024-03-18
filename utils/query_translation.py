from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain import hub


def generate_queries_multi(llm,retriever, question):
    # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    def get_unique_union(documents: list[list]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]

    # Retrieve
    # question = "What is task decomposition for LLM agents?"
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":question})
    return docs, retrieval_chain

def vanilla_rag(question, llm, retrieval_chain):
    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0)

    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    return(final_rag_chain.invoke({"question":question}))

def run_multi_query( llm, retriever, question):
    docs, retrieval_chain = generate_queries_multi(llm, retriever, question)
    answer = vanilla_rag(question, llm, retrieval_chain)
    return answer, docs

def generate_queries_rag_fusion(llm, retriever, question):
    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_rag_fusion
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    def reciprocal_rank_fusion(results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula """

        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # Return the reranked results as a list of tuples, each containing the document and its fused score
        return reranked_results

    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    docs = retrieval_chain_rag_fusion.invoke({"question": question})
    return docs, retrieval_chain_rag_fusion

def vanilla_rag_fusion(question, llm, retrieval_chain_rag_fusion):
    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, 
         "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    return final_rag_chain.invoke({"question":question})

def run_rag_fusion(llm, retriever, question):
    docs, retrieval_chain_rag_fusion = generate_queries_rag_fusion(llm, retriever, question)
    contexts = [doc[0] for doc in docs]
    answer = vanilla_rag_fusion(question, llm, retrieval_chain_rag_fusion)
    return answer, contexts

def generate_queries_decomposition(llm, question):
    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)

    # Chain
    generate_queries_decomposition_chain = (prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

    # Run
    questions = generate_queries_decomposition_chain.invoke({"question":question})
    return questions, generate_queries_decomposition_chain

def answer_recursively(llm, retriever, question, questions):
    # Prompt
    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    decomposition_prompt = ChatPromptTemplate.from_template(template)

    def format_qa_pair(question, answer):
        """Format Q and A pair"""

        formatted_string = ""
        formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
        return formatted_string.strip()

    # 
    all_contexts = []
    q_a_pairs = ""
    for q in questions:
        context = retriever.invoke(q)
        # Add the formatted context to the list
        all_contexts.append(context)

        rag_chain = (
        {"context": itemgetter("question") | retriever, 
         "question": itemgetter("question"),
         "q_a_pairs": itemgetter("q_a_pairs")} 
        | decomposition_prompt
        | llm
        | StrOutputParser())

        answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
        q_a_pair = format_qa_pair(q,answer)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
    all_context_flat_list = [context for contexts in all_contexts for context in contexts]
    return answer, all_context_flat_list

def run_recursive_decomposition(llm, retriever, question):
    questions, _ = generate_queries_decomposition(llm, question)
    answer, contexts = answer_recursively(llm, retriever, question, questions)
    return answer, contexts

def retrieve_and_rag(llm, retriever, question, sub_question_generator_chain):
    """RAG on each sub-question"""

    # Use our decomposition / 
    sub_questions = sub_question_generator_chain.invoke({"question":question})

    # Initialize a list to hold RAG chain results
    rag_results = []

    retrieved_docs_list = []

    # RAG prompt
    prompt_rag = hub.pull("rlm/rag-prompt")

    for sub_question in sub_questions:

        # Retrieve documents for each sub-question
        retrieved_docs = retriever.get_relevant_documents(sub_question)

        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs, 
                                                                "question": sub_question})
        rag_results.append(answer)
        retrieved_docs_list.append(retrieved_docs)

    retrieved_docs_flat_list = [context for contexts in retrieved_docs_list for context in contexts]
    return rag_results, sub_questions, retrieved_docs_flat_list

def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""

    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

def answer_from_qa_pairs(llm, question, context):
    # Prompt
    template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return final_rag_chain.invoke({"context":context,"question":question})

def run_individual_decomposition(llm, retriever, question):
    prompt_rag = hub.pull("rlm/rag-prompt")
    _, generate_queries_decomposition_chain = generate_queries_decomposition(llm, question)
    answers, questions, retrieved_docs_flat_list = retrieve_and_rag(llm, retriever, question, generate_queries_decomposition_chain)
    context = format_qa_pairs(questions, answers)
    answer = answer_from_qa_pairs(llm, question, context)
    return answer, retrieved_docs_flat_list

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

def generate_queries_step_back(llm, question):
    # Few Shot Examples
    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel's was born in what country?",
            "output": "what is Jan Sindel's personal history?",
        },
    ]
    # We now transform these to example messages
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            # Few shot examples
            few_shot_prompt,
            # New question
            ("user", "{question}"),
        ]
    )
    generate_queries_step_back_chain = prompt | llm | StrOutputParser()
    queries = generate_queries_step_back_chain.invoke({"question": question})
    return queries, generate_queries_step_back_chain

def step_back_rag(llm, retriever, question, queries, generate_queries_step_back_chain):
    # Response prompt 
    response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

    # {normal_context}
    # {step_back_context}

    # Original Question: {question}
    # Answer:"""
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
    normal_context = retriever.invoke(question)
    # print(normal_context)
    step_back_context = retriever.invoke(generate_queries_step_back_chain.invoke({"question": question}))
    # print(step_back_context)
    chain = (
        {
            # Retrieve context using the normal question
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
            # Retrieve context using the step-back question
            "step_back_context": generate_queries_step_back_chain | retriever,
            # Pass on the question
            "question": lambda x: x["question"],
        }
        | response_prompt
        | llm
        | StrOutputParser()
    )
    answer =  chain.invoke({"question": question})
    combined_context = normal_context + step_back_context
    # print('normal', normal_context)
    # print('step', step_back_context)
    # print('combined', combined_context)
    return answer, combined_context

def run_step_back_rag(llm, retriever, question):
    queries, generate_queries_step_back_chain = generate_queries_step_back(llm, question)
    # print(queries)
    answer, contexts = step_back_rag(llm, retriever, question, queries, generate_queries_step_back_chain)
    # print(contexts)
    return answer, contexts

def generate_docs_for_retrieval(llm, question):
    # HyDE document genration
    template = """Please write a scientific paper passage to answer the question
    Question: {question}
    Passage:"""
    prompt_hyde = ChatPromptTemplate.from_template(template)

    generate_docs_for_retrieval_chain = (
        prompt_hyde | llm | StrOutputParser() 
    )

    # Run
    # return generate_docs_for_retrieval_chain.invoke({"question":question})
    return generate_docs_for_retrieval_chain

def retrieve_hyde(retriever, question, generate_docs_for_retrieval_chain):
    # Retrieve
    retrieval_chain = generate_docs_for_retrieval_chain | retriever 
    retrieved_docs = retrieval_chain.invoke({"question":question})
    return retrieved_docs

def hyde_rag(llm, question, retireved_docs):
    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return final_rag_chain.invoke({"context":retireved_docs,"question":question})

def run_hyde(llm, retriever, question):
    generate_docs_for_retrieval_chain = generate_docs_for_retrieval(llm, question)
    retireved_docs = retrieve_hyde(retriever, question, generate_docs_for_retrieval_chain)
    answer = hyde_rag(llm, question, retireved_docs)
    return answer, retireved_docs