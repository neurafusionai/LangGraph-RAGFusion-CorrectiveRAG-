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
import pprint

# EMBEDDING_MODEL serves as a crucial hyperparameter, representing the embedding model used for text vectorization.
# In a production-grade system, this should be adaptive or fine-tuned based on domain-specific data.
EMBEDDING_MODEL = "text-embedding-3-small"

# TOP_K determines the number of top similar documents to retrieve from the vector store.
# This parameter could be dynamically adjusted depending on context sensitivity and computational resources.
TOP_K = 5

# MAX_DOCS_FOR_CONTEXT is the upper bound on the number of documents considered during query augmentation.
# This prevents context overflow in large document sets, which can degrade model performance.
MAX_DOCS_FOR_CONTEXT = 8

# DOCUMENT_PDF specifies the input document, which is loaded and processed.
# Future iterations could support multiple document formats or even streams.
DOCUMENT_PDF = "graphrag-ms.pdf"

# Environment variables for API keys are set here. These should ideally be managed using a secure secrets manager in production.
os.environ["SEARCHAPI_API_KEY"] = " " 
os.environ["OPENAI_API_KEY"] = " "

# The GraphState TypedDict provides a structured definition for maintaining state across the process lifecycle.
# This design leverages strong typing to prevent runtime errors and ensures clarity in state management.
class GraphState(TypedDict):
    gpt_4o: ChatOpenAI  # Claude 3 「Haiku」 model from OpenAI; a lightweight model optimized for generation tasks.
    emb_model: OpenAIEmbeddings  # Embedding model to vectorize the textual data.
    question: str  # The core question or query input by the user, serving as the focal point for subsequent operations.
    generate_querys: List[str]  # List of queries generated to refine or expand the original question.
    generate_query_num: int  # Number of queries to generate for expanded search capability.
    integration_question: str  # Final consolidated question after integrating multiple queries.
    transform_question: str  # Transformed query, optimized for web search engines.
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Sequential history of messages exchanged, aiding in context continuity.
    fusion_documents: List[List[Document]]  # Documents fetched from multiple queries before fusion.
    documents: List[Document]  # Final set of documents selected for answer generation.
    is_search: bool  # Boolean flag indicating if a web search is necessary for resolving the query.

# The state dictionary initializes the GraphState structure.
# The use of such a dictionary allows for extensibility and modifications with minimal code disruption.
state = {
    "llm_opus": ChatOpenAI(model_name="gpt-4o"),  # High-level LLM interface for executing prompt templates and generating responses.
    "question": "what is RAG-FUSION?",  # Example query to explore a complex concept.
    "generate_query_num": 2,  # Set to generate two queries in addition to the original; an optimal number for efficient search.
    "emb_model": "text-embedding-3-small",  # Embedding model specified for vectorization tasks.
    "generate_querys": [],  # Placeholder for storing generated queries.
    "integration_question": '',  # Placeholder for the integrated question.
    "transform_question": '',  # Placeholder for the transformed query.
    "documents": List[Document]  # Placeholder for the documents to be processed.
}

# generate_query function orchestrates the generation of multiple queries based on the original user query.
# It ensures that search coverage is broad, enhancing the chances of retrieving relevant documents.
def generate_query(state: GraphState) -> GraphState:
    print("\n--- __start__ ---")
    print("--- generate_query ---")
    llm = state["llm_opus"]
    question = state["question"]
    generate_query_num = state["generate_query_num"]
    
    # The system_prompt provides a clear directive to the model, ensuring that it understands its role in query generation.
    system_prompt = "You are an assistant that generates multiple search queries based on a single input query."
    
    # human_prompt guides the model in generating semantically consistent but varied queries, crucial for diversified search results.
    human_prompt = """When creating queries, output each query on a new line without significantly changing the original query's meaning.
    Input query: {question}
    {generate_query_num} output queries: 
    """
    
    # ChatPromptTemplate encapsulates the system and human prompts into a coherent template, ensuring that the model understands the task structure.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
    )
    
    # StrOutputParser processes the model's output into a list of queries, ready for downstream use.
    questions_chain = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    
    # The invoke method triggers the generation process and stores the queries in the state.
    generate_querys = questions_chain.invoke(
        {
            "question": question, 
            "generate_query_num": generate_query_num
        }
    )
    
    # The original question is prepended to ensure it remains a primary reference during retrieval.
    generate_querys.insert(0, "0. " + question)
    
    print("\nOriginal Question + Generated Questions==========================")
    for i, query in enumerate(generate_querys):
        print(f"\n{query}")
    print("\n===========================================================\n")
    
    # Updates the state with the newly generated queries.
    state["generate_querys"] = generate_querys
    return state

state = generate_query(state)

# retrieve function handles document retrieval, leveraging embeddings and vector stores to fetch relevant information.
# It is a critical component of any RAG (Retrieval-Augmented Generation) pipeline.
def retrieve(state: GraphState) -> GraphState:
    print("--- retrieve ---")
    print(state)
    
    emb_model = state['emb_model']  # Embedding model specified in the state.
    generate_querys = state["generate_querys"]  # Generated queries to be used for document retrieval.
    
    # PyPDFLoader is employed to load PDF documents, a common format in enterprise knowledge bases.
    raw_documents = PyPDFLoader(DOCUMENT_PDF).load()
    
    # TokenTextSplitter ensures that documents are broken down into manageable chunks for efficient retrieval.
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=24)
    
    # The split_documents method processes raw documents into smaller chunks.
    documents = text_splitter.split_documents(raw_documents)
    print("Original document: ", len(documents), " docs")
    
    # Chroma vector store is initialized with the document embeddings, enabling efficient similarity search.
    embeddings = OpenAIEmbeddings(model=emb_model)
    vectordb = Chroma.from_documents(documents, embeddings)
  
    fusion_documents = []
    
    # For each generated query, relevant documents are retrieved from the vector store.
    for question in generate_querys:
        docs = vectordb.similarity_search(question, k=3)
        fusion_documents.append(docs)
    
    # The fusion documents are stored in the state for further processing.
    state["fusion_documents"] = fusion_documents
    return state

state = retrieve(state)

# fusion function combines documents from multiple queries to form a coherent set of documents for final processing.
# It uses a fusion strategy that prioritizes relevance across different queries.
def fusion(state):
    print("--- fusion ---")
    fusion_documents = state["fusion_documents"]
    k = 60  # A hyperparameter in the fusion ranking formula; controls the impact of rank on the fusion score.
    documents = []
    fused_scores = {}
    
    # Documents are ranked and scored based on their appearance across multiple queries.
    for docs in fusion_documents:
        for rank, doc in enumerate(docs, start=1):
            if doc.page_content not in fused_scores:
                fused_scores[doc.page_content] = 0
                documents.append(doc)
            fused_scores[doc.page_content] += 1 / (rank + k)
    
    # The top 3 documents are selected based on their cumulative fusion score.
    reranked_results = {doc_str: score for doc_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:3]}
    print("\nTop 3 search scores ========================================")
    for i, score in enumerate(reranked_results.values(), start=1):
        print(f"\nDocument {i}: {score}")
    print("\n===========================================================\n")
    
    # The filtered documents are retained for the next stage of processing.
    filterd_documents = []
    for doc in documents:
        if doc.page_content in reranked_results:
            filterd_documents.append(doc) 
    documents = filterd_documents
    state["documents"] = documents   
    return state

state = fusion(state)

# integration_query function consolidates multiple queries into a single, integrated query.
# This is essential for scenarios where a unified query provides a clearer search intent.
def integration_query(state):
    print("--- integration_query ---")
    llm = state["llm_opus"]
    generate_querys = state["generate_querys"]
    
    # system_prompt instructs the LLM to consolidate multiple queries into one, ensuring it grasps the task's goal.
    system_prompt = """You are a question rewriter that consolidates multiple input questions into one question."""
    
    # human_prompt specifies the format of the output, emphasizing the need for a singular, coherent question.
    human_prompt = """Please output only the integrated question.
    Multiple questions: {query}
    Integrated question: """
    
    # The prompt is constructed and used to drive the LLM towards producing the integrated query.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    
    # The integration_chain generates the integrated query.
    integration_chain = prompt | llm | StrOutputParser()
    questions = "\n".join(generate_querys)
    integration_query = integration_chain.invoke({"query": questions})
    
    # Logs and updates the state with the integrated query.
    print(f"\nIntegrated question: {integration_query}\n")
    state["integration_question"] = integration_query
    return state

state = integration_query(state)

# grade_documents function evaluates the relevance of each retrieved document in relation to the integrated question.
# This step ensures only the most pertinent documents are considered for final generation.
def grade_documents(state):
    print("--- grade_documents ---")
    llm = state["llm_opus"]
    integration_question = state["integration_question"]
    documents = state["documents"]
    
    # system_prompt directs the LLM to evaluate document relevance, a critical step in filtering out noise.
    system_prompt = """You are an assistant that evaluates the relevance between searched documents and user questions.
    If the document contains keywords or semantic content related to the question, you evaluate it as relevant.
    Respond with "Yes" for relevance and "No" for no relevance."""
        
    # human_prompt formats the relevance evaluation request for each document.
    human_prompt = """
        
    Document: {context} 
        
    Question: {query}
    Relevance ("Yes" or "No"): """

    # The grading chain processes each document and determines its relevance.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    filtered_docs = []
    is_search = False
    grade_chain = prompt | llm | StrOutputParser()
    
    # Logs and filters documents based on the relevance evaluation.
    print("\nEvaluation of relevance for each document =============================")
    for doc in documents:
        grade = grade_chain.invoke({"context": doc.page_content, "query": integration_question})
        print(f"\nRelevance: {grade}")
        if "Yes" in grade:
            filtered_docs.append(doc)
        else:
            is_search = True
    print("\n===========================================================\n")
    
    # Updates the state with filtered documents and search necessity.
    state["documents"] = filtered_docs
    state["is_search"] = is_search
    return state

state = grade_documents(state)

# decide_to_generate function determines whether a web search is needed or if the system can proceed to message generation.
# This decision point is crucial for optimizing the balance between precision and resource utilization.
def decide_to_generate(state):
    print("--- decide_to_generate ---")
    is_search = state['is_search']
    
    # Returns the next step based on whether additional searching is required.
    if is_search == True:
        return "transform_query"
    else:
        return "create_message"

# transform_query function refines the integrated question into a format suitable for web searches.
# This process is pivotal in enhancing search engine compatibility and improving result quality.
def transform_query(state):
    print("--- transform_query ---")
    llm = state["llm_opus"]
    integration_question = state["integration_question"]
    
    # system_prompt instructs the LLM to optimize the question for web search, emphasizing intent capture.
    system_prompt = """You are a rewriter that converts input questions into queries optimized for web search."""
    
    # human_prompt guides the transformation, ensuring that the query is clear and search-engine friendly.
    human_prompt = """Look at the question and infer the fundamental meaning/intent to output only the web search query.
    Question: {query}
    Web search query: """
    
    # The transformation chain handles the conversion to a search query.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    transform_chain = prompt | llm | StrOutputParser()
    transform_query = transform_chain.invoke({"query": integration_question})
    
    # Logs and updates the state with the transformed query.
    print(f"\nWeb search query: {transform_query}\n")
    state["transform_question"] = transform_query
    return state

state = transform_query(state)

# web_search function performs an online search using the transformed query and integrates the results with existing documents.
# This step is essential for enriching the document set with the latest information.
def web_search(state):
    print("--- web_search ---")
    transform_question = state["transform_question"]
    documents = state["documents"]
    
    # SearchApiAPIWrapper is used to interact with a web search API, retrieving relevant documents.
    retriever = SearchApiAPIWrapper()
    docs = retriever.run(transform_question)
    
    # The retrieved documents are appended to the existing set for further processing.
    documents.extend(docs)
    state["documents"] = documents
    return state

state = web_search(state)

# create_message function constructs the final message based on the selected documents and the original question.
# This function is critical for synthesizing a coherent and contextually accurate response.
def create_message(state):
    print("--- create_message ---")
    documents = state["documents"]
    question = state["question"]
    
    # system_message ensures the LLM responds in English, which may be important for cross-cultural applications.
    system_message = "You will always respond in English."
    
    # human_message structures the context and question in a way that the LLM can generate a targeted response.
    human_message = """Refer to the context separated by '=' signs below to answer the question.
    {context}
    Question: {query}
    """
    
    # The prompt is used to format the message, integrating the selected documents and question.
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
    
    # The final messages are stored in the state for generation.
    state["messages"] = messages
    return state

state = create_message(state)

# generate function produces the final response based on the constructed message.
# This is the culmination of the process, where the LLM generates the answer.
def generate(state):
    print("--- generate ---")
    llm = state["llm_opus"]
    messages = state["messages"]
    
    # The invoke method triggers the LLM to generate the final response.
    response = llm.invoke(messages)
    print("--- end ---\n")
    
    # The response is stored in the state, completing the process.
    state["messages"] = [response]
    return state

state = generate(state)

# Final output of the generated message.
print("Updated state with generated queries:")
print(state["messages"])

# get_compile_graph function creates and compiles a StateGraph that defines the sequence of operations in this process.
# This graph structure enhances modularity, making it easier to manage, debug, and extend.
def get_compile_graph():
    graph = StateGraph(GraphState)
    graph.set_entry_point("generate_query")  # Entry point is set to the first operation.
    
    # Nodes are added to the graph representing each function, ensuring a clear and organized workflow.
    graph.add_node("generate_query", generate_query)
    graph.add_edge("generate_query", "retrieve")
    graph.add_node("retrieve", retrieve)
    graph.add_edge("retrieve", "fusion")
    graph.add_node("fusion", fusion)
    graph.add_edge("fusion", "integration_query")
    graph.add_node("integration_query", integration_query)
    graph.add_edge("integration_query", "grade_documents")
    graph.add_node("grade_documents", grade_documents)
    
    # Conditional edges are introduced to handle branching logic based on document relevance evaluation.
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "create_message": "create_message"
        },
    )
    
    # The remaining nodes are added, ensuring that the process can transition smoothly from search to generation.
    graph.add_node("transform_query", transform_query)
    graph.add_edge("transform_query", "web_search")
    graph.add_node("web_search", web_search)
    graph.add_edge("web_search", "create_message")
    graph.add_node("create_message", create_message)
    graph.add_edge("create_message", "generate")
    graph.add_node("generate", generate)
    
    # The final edge points to the END, marking the completion of the process.
    graph.add_edge("generate", END)

    # The graph is compiled, ensuring that it is ready for execution or further analysis.
    compile_graph = graph.compile()
