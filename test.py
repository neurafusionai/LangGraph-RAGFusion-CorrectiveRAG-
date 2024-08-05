import io
import os
import operator
from typing import List, TypedDict, Sequence, Annotated
from langchain_core.messages import BaseMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import TokenTextSplitter
import streamlit as st

# EMBEDDING_MODEL defines the embedding strategy for transforming textual input into high-dimensional vector representations.
# The chosen model, "text-embedding-3-small", is optimized for balance between performance and efficiency, making it ideal for real-time systems.
# In enterprise applications, this model might be replaced or fine-tuned with domain-specific data to enhance relevance.
EMBEDDING_MODEL = "text-embedding-3-small"

# TOP_K parameter governs the retrieval depth during similarity search operations.
# Setting TOP_K = 5 ensures that the system balances recall (finding enough relevant documents) with precision (excluding irrelevant data).
# This parameter may require dynamic adjustment based on empirical evidence from usage patterns.
TOP_K = 5

# MAX_DOCS_FOR_CONTEXT limits the number of documents passed into the context for generation.
# This constraint is crucial for managing computational resources and ensuring that the LLM (Large Language Model) can process the input efficiently.
MAX_DOCS_FOR_CONTEXT = 8

# DOCUMENT_PDF specifies the input file name, serving as the primary data source.
# The document ID or file path should ideally be dynamically assigned or stored in a document management system for large-scale applications.
DOCUMENT_PDF = "2402.03367v2.pdf"

# Environment variables are utilized to securely store API keys.
# In production, these should be managed by a secrets manager or a secure environment configuration system.
os.environ["SEARCHAPI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

# Streamlit is leveraged here to create an interactive interface for the chatbot.
# The use of Streamlit ensures rapid deployment and visualization capabilities, making it an excellent choice for prototyping in research and industry environments.
st.title("Multi-PDF ChatBot using RAG Fusion & Corrective")
user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')

# Sidebar allows users to upload multiple PDF files, facilitating multi-document analysis.
# This design decision supports complex queries that require information from diverse sources.
with st.sidebar:
    uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
    process = st.button("Process")

# GraphState is a TypedDict representing the state maintained throughout the processing pipeline.
# This structure enables type safety and clarity, essential for ensuring consistency across different processing stages.
# Annotations and strong typing enhance the robustness of the code, reducing the likelihood of runtime errors.
class GraphState(TypedDict):
    llm_opus: ChatOpenAI  # Claude 3「Haiku」 model, a specialized LLM optimized for generating contextually relevant queries and answers.
    emb_model: OpenAIEmbeddings  # The embedding model that vectorizes text data, enabling semantic search.
    question: str  # The initial query posed by the user, forming the basis of the subsequent processing steps.
    generate_querys: List[str]  # List of generated queries derived from the original question to enhance document retrieval scope.
    generate_query_num: int  # Number of queries to generate, balancing between comprehensiveness and computational efficiency.
    integration_question: str  # Consolidated question derived from multiple generated queries, improving the relevance of results.
    transform_question: str  # The transformed query, optimized for web search engines to retrieve additional context if necessary.
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Historical sequence of messages, ensuring context continuity during LLM operations.
    fusion_documents: List[List[Document]]  # Hierarchical structure storing documents retrieved for each query, crucial for subsequent fusion and ranking.
    documents: List[Document]  # Final set of documents passed to the LLM for answer generation, representing the most relevant content.
    is_search: bool  # Boolean flag indicating whether additional web search is required, optimizing resource usage.

# generate_query function generates multiple variations of the user's query.
# This step is critical for expanding the search space, thereby increasing the likelihood of retrieving relevant documents.
# The function leverages the ChatOpenAI model to create semantically diverse queries while maintaining the original intent.
def generate_query(state: GraphState) -> GraphState:
    print("\n--- __start__ ---")
    print("--- generate_query ---")
    llm = state["llm_opus"]
    question = state["question"]
    generate_query_num = state["generate_query_num"]
    
    # The system_prompt establishes the LLM's role as a query generation assistant, setting clear expectations for its output.
    system_prompt = "You are an assistant that generates multiple search queries based on a single input query."
    
    # The human_prompt provides specific instructions on how to generate the queries, ensuring that the model produces output aligned with the task requirements.
    human_prompt = """When creating queries, output each query on a new line without significantly changing the original query's meaning.
    Input query: {question}
    {generate_query_num} output queries: 
    """
    
    # ChatPromptTemplate orchestrates the interaction with the LLM, ensuring a structured and consistent approach to query generation.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
    )
    
    # questions_chain represents the processing pipeline, where the LLM generates and the output parser refines the list of queries.
    questions_chain = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    
    # The invoke method executes the LLM, generating the list of queries, which are then inserted into the state.
    generate_querys = questions_chain.invoke(
        {
            "question": question, 
            "generate_query_num": generate_query_num
        }
    )
    
    # The original query is preserved by inserting it at the beginning of the list, ensuring it is considered during the retrieval phase.
    generate_querys.insert(0, "0. " + question)
    print("\nOriginal Question + Generated Questions==========================")
    for i, query in enumerate(generate_querys):
        print(f"\n{query}")
    print("\n===========================================================\n")
    
    return {"generate_querys": generate_querys}

# retrieve function is responsible for retrieving relevant documents based on the generated queries.
# This function is central to the RAG (Retrieval-Augmented Generation) pipeline, enabling the system to pull in pertinent information.
# It employs vector search techniques, which are essential for handling large-scale document repositories efficiently.
def retrieve(state: GraphState) -> GraphState:
    print("--- retrieve ---")
    print(state)
    emb_model = state['emb_model']  # Embedding model used for transforming text into vector representations, facilitating similarity search.
    generate_querys = state["generate_querys"]  # The list of queries generated from the original question.
    
    # PyPDFLoader is utilized to load the PDF document, ensuring compatibility with a wide range of document formats.
    raw_documents = PyPDFLoader(DOCUMENT_PDF).load()
    
    # TokenTextSplitter is employed to break the document into smaller chunks, optimizing retrieval performance.
    # Chunk size and overlap are tuned to balance between retrieval precision and recall, critical for maintaining context.
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents)
    print("Original document: ", len(documents), " docs")

    # Chroma vector store is initialized with document embeddings, enabling efficient similarity searches against the queries.
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(documents, embeddings)

    # The fusion_documents list aggregates documents retrieved for each query, setting the stage for subsequent fusion and ranking.
    fusion_documents = []
    for question in generate_querys:
        docs = vectordb.similarity_search(question, k=3)  # The top k documents are selected based on their similarity to each query.
        fusion_documents.append(docs)
    return {"fusion_documents": fusion_documents}

# fusion function aggregates and ranks documents based on their relevance across multiple queries.
# The fusion mechanism enhances the likelihood of selecting documents that are consistently relevant, improving the final output quality.
# This step is essential for filtering and prioritizing information, especially in multi-document scenarios.
def fusion(state):
    print("--- fusion ---")
    fusion_documents = state["fusion_documents"]  # Documents retrieved from vector store searches.
    k = 60  # Hyperparameter controlling the weight assigned to document rank during the fusion process.
    documents = []
    fused_scores = {}
    
    # Each document is scored based on its rank across different queries, allowing for a fusion of relevance scores.
    # This fusion approach ensures that documents consistently ranked across multiple queries are prioritized,
    # which enhances the quality and relevance of the final document selection.
    for docs in fusion_documents:
        for rank, doc in enumerate(docs, start=1):
            # If a document's content has not been encountered before, it is initialized in the fused_scores dictionary.
            # This prevents duplicate scoring and ensures that each document is scored only once.
            if doc.page_content not in fused_scores:
                fused_scores[doc.page_content] = 0
                documents.append(doc)
            # The scoring mechanism inversely weights the rank, giving higher importance to documents that appear earlier in the list.
            # The hyperparameter k controls the influence of rank, where lower ranks (higher importance) have a more significant impact.
            fused_scores[doc.page_content] += 1 / (rank + k)
    
    # The documents are then reranked based on their accumulated scores, selecting the top 3 for further processing.
    # This reranking process is critical for filtering out less relevant documents and focusing on the most promising candidates.
    reranked_results = {doc_str: score for doc_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:3]}
    print("\nTop 3 search scores ========================================")
    for i, score in enumerate(reranked_results.values(), start=1):
        print(f"\nDocument {i}: {score}")
    print("\n===========================================================\n")

    # The filtered_documents list is populated with the final set of documents that match the top 3 reranked scores.
    # This filtering ensures that only the most relevant documents proceed to the next stages of processing, optimizing efficiency and output quality.
    filterd_documents = []
    for doc in documents:
        if doc.page_content in reranked_results:
            filterd_documents.append(doc)
    documents = filterd_documents
    return {"documents": documents}


def integration_query(state):
    print("--- integration_query ---")
    llm = state["llm_opus"]
    generate_querys = state["generate_querys"]
    
    # The system_prompt establishes the task for the LLM to consolidate multiple queries into one.
    # This step is vital for simplifying and focusing the search intent, which improves the relevance of the final response.
    system_prompt = """You are a question rewriter that consolidates multiple input questions into one question."""
    
    # The human_prompt guides the LLM to output only the integrated question, ensuring clarity and precision.
    human_prompt = """Please output only the integrated question.
    Multiple questions: {query}
    Integrated question: """
    
    # The integration_chain processes the list of queries and produces a single, coherent question.
    # This integration is crucial for reducing redundancy and ensuring that the final query captures the core search intent.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    integration_chain = prompt | llm | StrOutputParser()
    questions = "\n".join(generate_querys)
    integration_query = integration_chain.invoke({"query": questions})
    
    # The integrated question is logged and returned, ready for the next phase of processing.
    print(f"\nIntegrated question: {integration_query}\n")
    return {"integration_question": integration_query}


def grade_documents(state):
    print("--- grade_documents ---")
    llm = state["llm_opus"]
    integration_question = state["integration_question"]
    documents = state["documents"]
    
    # system_prompt sets the LLM's role as an evaluator of document relevance, providing clear criteria for relevance assessment.
    # This process is essential for filtering out irrelevant information, which could dilute the quality of the final response.
    system_prompt = """You are an assistant that evaluates the relevance between searched documents and user questions.
    If the document contains keywords or semantic content related to the question, you evaluate it as relevant.
    Respond with "Yes" for relevance and "No" for no relevance."""
    
    # human_prompt structures the relevance evaluation request, guiding the LLM in comparing each document with the integrated question.
    human_prompt = """
    Document: {context}
    Question: {query}
    Relevance ("Yes" or "No"): """
    
    # grade_chain processes each document and determines its relevance to the integrated question.
    # This filtering is crucial to ensure that only the most pertinent documents are passed on for response generation.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    filtered_docs = []
    is_search = False
    grade_chain = prompt | llm | StrOutputParser()
    
    # Each document's relevance is evaluated and logged, with only relevant documents being retained.
    # Documents that do not meet the relevance criteria are excluded, optimizing the response generation process.
    print("\nEvaluation of relevance for each document =============================")
    for doc in documents:
        grade = grade_chain.invoke({"context": doc.page_content, "query": integration_question})
        print(f"\nRelevance: {grade}")
        if "Yes" in grade:
            filtered_docs.append(doc)
        else:
            is_search = True
    print("\n===========================================================\n")
    return {"documents": filtered_docs, "is_search": is_search}


def decide_to_generate(state):
    print("--- decide_to_generate ---")
    is_search = state['is_search']
    
    # This function determines the next action based on the relevance of the documents.
    # If the documents are deemed insufficient (is_search = True), the system proceeds to optimize the query for a web search.
    # Otherwise, it directly generates the response based on the available documents.
    if is_search == True:
        return "transform_query"
    else:
        return "create_message"
    
def transform_query(state):
    print("--- transform_query ---")
    llm = state["llm_opus"]
    integration_question = state["integration_question"]
    
    # system_prompt instructs the LLM to convert the integrated question into a format optimized for web search.
    # This transformation is crucial for ensuring that the query aligns with search engine capabilities and maximizes the retrieval of relevant information.
    system_prompt = """You are a rewriter that converts input questions into queries optimized for web search."""
    
    # human_prompt guides the LLM to focus on the core intent of the question, ensuring the output is a precise and effective web search query.
    human_prompt = """Look at the question and infer the fundamental meaning/intent to output only the web search query.
    Question: {query}
    Web search query: """
    
    # The transform_chain processes the integrated question and refines it into a query suitable for web search engines.
    # This step is vital for retrieving additional or missing context from the web, which may not be present in the initial document set.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    transform_chain = prompt | llm | StrOutputParser()
    transform_query = transform_chain.invoke({"query": integration_question})
    
    # The transformed query is logged and stored in the state for subsequent web search operations.
    print(f"\nWeb search query: {transform_query}\n")
    state["transform_question"] = transform_query
    return {"transform_question": transform_query}

def web_search(state):
    print("--- web_search ---")
    transform_question = state["transform_question"]
    documents = state["documents"]
    
    # The web_search function uses the SearchApiAPIWrapper to retrieve documents from external sources using the transformed query.
    # This is essential for supplementing the existing document set with up-to-date or missing information that enhances the final response quality.
    retriever = SearchApiAPIWrapper()
    docs = retriever.run(transform_question)
    
    # The new documents retrieved from the web are appended to the existing document list, ensuring a comprehensive context for the final response.
    documents.extend(docs)
    return {"documents": documents}


def create_message(state):
    print("--- create_message ---")
    documents = state["documents"]
    question = state["question"]
    
    # system_message instructs the LLM to generate all responses in English, which could be crucial in maintaining consistency across multilingual data sources.
    system_message = "You will always respond in English."
    
    # human_message structures the context and the query in a way that guides the LLM to produce a coherent and contextually relevant answer.
    # The context is partitioned with separators to delineate different documents, ensuring clarity in the generated response.
    human_message = """Refer to the context separated by '=' signs below to answer the question.
    {context}
    Question: {query}
    """
    
    # The prompt is constructed to integrate the context from the selected documents with the user's original question.
    # This integration is critical for ensuring that the LLM has all necessary information to generate an accurate and relevant response.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", human_message),
        ]
    )
    partition = "\n" + "=" * 20 + "\n"
    valid_documents = [doc for doc in documents if hasattr(doc, 'page_content')]
    documents_context = partition.join([doc.page_content for doc in valid_documents])
    messages = prompt.format_messages(context=documents_context, query=question)

    return {"messages": messages}


def generate(state):
    print("--- generate ---")
    llm = state["llm_opus"]
    messages = state["messages"]
    
    # The generate function invokes the LLM to produce the final response based on the prepared messages.
    # This is the culmination of the process, where all the preceding steps come together to answer the user's query.
    response = llm.invoke(messages)
    print("--- end ---\n")
    
    # The generated response is returned and stored in the state for output, completing the processing pipeline.
    return {"messages": [response]}


def get_compile_graph():
    graph = StateGraph(GraphState)
    graph.set_entry_point("generate_query")
    
    # The graph represents the workflow as a sequence of nodes and edges, each corresponding to a function in the processing pipeline.
    # This graph-based design enhances modularity, making the system easier to debug, extend, and optimize.
    graph.add_node("generate_query", generate_query)
    graph.add_edge("generate_query", "retrieve")
    graph.add_node("retrieve", retrieve)
    graph.add_edge("retrieve", "fusion")
    graph.add_node("fusion", fusion)
    graph.add_edge("fusion", "integration_query")
    graph.add_node("integration_query", integration_query)
    graph.add_edge("integration_query", "grade_documents")
    graph.add_node("grade_documents", grade_documents)
    
    # Conditional edges introduce decision points in the workflow, allowing the process to adapt based on intermediate results.
    # This conditional logic is crucial for optimizing resource usage and ensuring that the system responds intelligently to different scenarios.
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "create_message": "create_message"
        },
    )
    
    # The final stages of the workflow guide the process from query transformation to web search and ultimately to response generation.
    # Each node in the graph corresponds to a specific function, ensuring a clear and organized flow of operations.
    graph.add_node("transform_query", transform_query)
    graph.add_edge("transform_query", "web_search")
    graph.add_node("web_search", web_search)
    graph.add_edge("web_search", "create_message")
    graph.add_node("create_message", create_message)
    graph.add_edge("create_message", "generate")
    graph.add_node("generate", generate)
    graph.add_edge("generate", END)

    # The graph is compiled into a callable object, ready for execution or further analysis.
    # This compilation step is essential for transforming the high-level design into an executable workflow.
    compile_graph = graph.compile()
    
    return compile_graph

if process:

    llm_opus = ChatOpenAI(model_name="gpt-4o")  # Initializes the LLM for processing user queries.

    emb_model = OpenAIEmbeddings(model="text-embedding-3-small")  # Initializes the embedding model for document vectorization.

    compile_graph = get_compile_graph()  # Compiles the graph-based workflow for processing.
    print(compile_graph)
    
    # The state dictionary is initialized with the required inputs for processing the user's query.
    # This setup is crucial for ensuring that all necessary components are available when invoking the workflow.
    state = {
        "llm_opus": llm_opus,  # The LLM instance used for query processing.
        "question": "What is Rag Fusion",  # The user's query, serving as the input for the workflow.
        "generate_query_num": 2  # Specifies the number of alternative queries to generate.
    }

    print("State dictionary before invoking compile_graph:", state)
    
    # The compiled graph is invoked, processing the input state through the defined workflow to generate the final response.
    output = compile_graph.invoke(
        input={
            "llm_opus": state["llm_opus"],
            "question": state["question"],
            "generate_query_num": state["generate_query_num"],
        }
    )
    
    # The output, containing the generated response, is displayed in the Streamlit interface, providing the user with the final answer.
    st.write("output:")
    st.write(output["messages"][-1].content)


