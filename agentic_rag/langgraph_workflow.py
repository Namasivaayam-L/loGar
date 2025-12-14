from typing import Dict, Any, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import logging

# Import our custom agent
from .agent_core import LogAnalysisAgent, create_log_analysis_agent

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state for our graph
WorkflowState = Dict[str, Any]

def create_log_analysis_workflow(vector_store):
    """
    Create a simple LangGraph workflow for log analysis.

    The workflow consists of:
    1. Query Analysis - Analyze and refine the query
    2. Retrieval - Get relevant log chunks from vector store
    3. Answer Generation - Use Gemini to generate response

    Returns:
        LangGraph workflow object ready to be compiled and invoked
    """
    # Initialize agent
    agent = create_log_analysis_agent(vector_store)

    # Create the workflow graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("query_analysis", query_analysis_node(agent))
    workflow.add_node("retrieval", retrieval_node(agent))
    workflow.add_node("answer_generation", answer_generation_node(agent))

    # Set the entry point
    workflow.set_entry_point("query_analysis")

    # Add edges
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "answer_generation")
    workflow.add_edge("answer_generation", END)

    # Compile the workflow
    compiled_workflow = workflow.compile()

    return compiled_workflow, agent

def query_analysis_node(agent: LogAnalysisAgent):
    """Node for analyzing and refining user queries."""

    def analyze_query(state: WorkflowState) -> WorkflowState:
        """Analyze the user query and prepare for retrieval."""
        query = state.get("query", "")
        logger.info(f"Analyzing query: {query}")

        # For simple agent, we can do basic query preprocessing
        # In a more advanced agent, this could involve query expansion,
        # intent detection, or reformulation

        # Basic query preparation
        analyzed_query = query.strip()

        # Add metadata about query type
        query_metadata = {
            "original_query": query,
            "processed_query": analyzed_query,
            "query_length": len(analyzed_query),
            "contains_error_terms": any(term in analyzed_query.lower() for term in [
                "error", "fail", "exception", "crash", "timeout", "denied", "unauthorized"
            ]),
            "contains_progress_terms": any(term in analyzed_query.lower() for term in [
                "process", "progress", "step", "workflow", "completed", "success"
            ]),
        }

        updated_state = state.copy()
        updated_state.update({
            "analyzed_query": analyzed_query,
            "query_metadata": query_metadata,
            "step": "query_analysis_completed"
        })

        logger.info("Query analysis completed")
        return updated_state

    return analyze_query

def retrieval_node(agent: LogAnalysisAgent):
    """Node for retrieving relevant log chunks."""

    def retrieve_documents(state: WorkflowState) -> WorkflowState:
        """Retrieve relevant log chunks based on analyzed query."""
        query = state.get("analyzed_query", "")
        query_metadata = state.get("query_metadata", {})

        logger.info(f"Retrieving documents for query: {query}")

        # Determine retrieval parameters based on query analysis
        top_k = 15 if query_metadata.get("contains_error_terms") else 10

        # Perform retrieval
        try:
            retrieved_docs = agent.vector_store.similarity_search(query, k=top_k)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # Format context for generation
            context = agent._format_context(retrieved_docs)

            updated_state = state.copy()
            updated_state.update({
                "retrieved_documents": retrieved_docs,
                "context": context,
                "top_k_retrieved": len(retrieved_docs),
                "step": "retrieval_completed"
            })

            return updated_state

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            updated_state = state.copy()
            updated_state.update({
                "retrieved_documents": [],
                "context": "",
                "retrieval_error": str(e),
                "step": "retrieval_completed"
            })
            return updated_state

    return retrieve_documents

def answer_generation_node(agent: LogAnalysisAgent):
    """Node for generating answers using Gemini."""

    def generate_answer(state: WorkflowState) -> WorkflowState:
        """Generate final answer using retrieved context."""
        query = state.get("analyzed_query", "")
        context = state.get("context", "")
        retrieved_docs = state.get("retrieved_documents", [])
        query_metadata = state.get("query_metadata", {})

        logger.info("Generating answer with Gemini...")

        try:
            # Generate response using agent's analysis method
            response = agent._generate_analysis(query, context)

            # Structure the final response
            final_response = {
                "analysis": response,
                "query": state.get("query", ""),
                "retrieved_chunks": len(retrieved_docs),
                "query_metadata": query_metadata,
                "top_relevant_chunks": [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "doc_id": doc.metadata.get("doc_id", "unknown"),
                        "chunk_idx": doc.metadata.get("chunk_idx", "unknown"),
                        "score": doc.metadata.get("score", 0.0)
                    } for doc in retrieved_docs[:5]
                ],
                "processing_steps": [
                    "query_analysis",
                    "retrieval",
                    "answer_generation"
                ]
            }

            updated_state = state.copy()
            updated_state.update({
                "final_response": final_response,
                "step": "answer_generation_completed",
                "success": True
            })

            logger.info("Answer generation completed successfully")
            return updated_state

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")

            final_response = {
                "analysis": f"Failed to generate analysis: {str(e)}",
                "query": state.get("query", ""),
                "retrieved_chunks": len(retrieved_docs),
                "error": True,
                "error_message": str(e)
            }

            updated_state = state.copy()
            updated_state.update({
                "final_response": final_response,
                "step": "answer_generation_completed",
                "success": False,
                "error": str(e)
            })

            return updated_state

    return generate_answer

# Convenience function to run the workflow
def run_agentic_log_analysis(query: str, vector_store, workflow=None) -> Dict[str, Any]:
    """
    Convenience function to run the complete agentic workflow.

    Args:
        query: User's log analysis query
        vector_store: MongoDB vector store instance
        workflow: Pre-compiled workflow (optional)

    Returns:
        Dict containing the analysis results
    """
    if workflow is None:
        # Create workflow if not provided
        workflow, agent = create_log_analysis_workflow(vector_store)

    # Initialize state
    initial_state = {
        "query": query,
        "step": "initialization"
    }

    # Run the workflow
    final_state = workflow.invoke(initial_state)

    # Return the final response
    return final_state.get("final_response", {})
