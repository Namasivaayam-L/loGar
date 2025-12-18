import os
import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv

from vector_store.mongo_vector_store import MongoDBVectorStore

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogAnalysisAgent:
    """Simple Agent for log analysis using retrieval and generative AI."""

    def __init__(self,
                 vector_store: VectorStore = None,
                 gemini_model: str = "gemini-2.5-flash"):
        """Initialize the log analysis agent."""
        self.vector_store = vector_store
        self.gemini_model = gemini_model

        # Initialize Gemini LLM
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        self.llm = ChatGoogleGenerativeAI(
            model=self.gemini_model,
            google_api_key=google_api_key,
            temperature=0.1,  # Low temperature for consistency
            max_retries=3,
        )

        # Setup prompts
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup prompts for log analysis tasks."""
        self.analysis_prompt = PromptTemplate(
            template="""
                You are an expert log analysis assistant. Analyze the following log information and answer the user's query.
                CONTEXT (Retrieved Log Chunks):
                {context}

                USER QUERY: {query}

                INSTRUCTIONS:
                1. Focus specifically on the retrieved log chunks provided above
                2. Provide detailed, technical analysis of errors and issues
                3. If the logs show progress or workflow, summarize the key steps
                4. Be precise about timestamps, error messages, and root causes
                5. If the information is insufficient, say so clearly
                6. Structure your response for clarity with headings if multiple issues are found

                ANALYSIS:
                """,
            input_variables=["context", "query"]
        )

    def analyze_query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze a log query using retrieval and generative analysis.

        Args:
            query: User's log analysis query
            top_k: Number of similar documents to retrieve

        Returns:
            Dict containing analysis, retrieved docs, and metadata
        """
        try:
            logger.info(f"Starting agentic analysis for query: {query}")

            # Step 1: Retrieve relevant log chunks
            retrieved_docs = self.vector_store.similarity_search(query, k=top_k)
            logger.info(f"Retrieved {len(retrieved_docs)} relevant log chunks")

            # Step 2: Format context from retrieved documents
            context = self._format_context(retrieved_docs)

            # Step 3: Generate analysis using Gemini
            analysis = self._generate_analysis(query, context)

            # Step 4: Structure response
            response = {
                "analysis": analysis,
                "retrieved_chunks": len(retrieved_docs),
                "query": query,
                "top_relevant_chunks": [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "doc_id": doc.metadata.get("doc_id", "unknown"),
                        "chunk_idx": doc.metadata.get("chunk_idx", "unknown"),
                        "score": doc.metadata.get("score", 0.0)
                    } for doc in retrieved_docs[:5]  # Top 5 most relevant
                ]
            }

            logger.info("Agentic analysis completed successfully")
            return response

        except Exception as e:
            logger.error(f"Error in agent analysis: {e}")
            return {
                "analysis": f"An error occurred during analysis: {str(e)}",
                "retrieved_chunks": 0,
                "error": True,
                "query": query
            }

    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            chunk_info = f"CHUNK {i}:\nFile: {doc.metadata.get('file_path', 'unknown')}\nDoc ID: {doc.metadata.get('doc_id', 'unknown')}\nChunk ID: {doc.metadata.get('chunk_idx', 'unknown')}\n\nContent:\n{doc.page_content}\n\n"
            context_parts.append(chunk_info)

        return "\n---\n".join(context_parts)

    def _generate_analysis(self, query: str, context: str) -> str:
        """Generate analysis using Gemini LLM."""
        try:
            # Create the formatted prompt
            formatted_prompt = self.analysis_prompt.format(
                query=query,
                context=context
            )

            # Generate response
            response = self.llm.invoke(formatted_prompt)

            return response.content

        except Exception as e:
            logger.error(f"Error generating analysis with Gemini: {e}")
            return f"Failed to generate analysis: {str(e)}"


def create_log_analysis_agent(vector_store: MongoDBVectorStore):
    """Factory function to create agent with configured vector store."""
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    return LogAnalysisAgent(
        vector_store=vector_store,
        gemini_model=gemini_model
    )
