from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from typing import List, Dict, Optional
import os
import tempfile
from pypdf import PdfReader
import docx
import re
import logging
import requests
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, Mapping
import weaviate
from functools import lru_cache
import hashlib
import json

# Disable CrewAI telemetry
os.environ["CREWAI_TELEMETRY"] = "false"

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Gemini API configuration
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Initialize Weaviate client
client = None
try:
    client = weaviate.Client(
        url="http://localhost:8080"
    )
    logger.info("Weaviate client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Weaviate client: {str(e)}")

@lru_cache(maxsize=100)
def get_document_hash(file_path: str) -> str:
    """Generate a hash of the document for caching"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

@lru_cache(maxsize=100)
def search_legal_documents(query: str, doc_type: str, limit: int = 3) -> List[Dict]:
    """Search for legal documents in the vector database with caching"""
    try:
        if not client:
            raise Exception("Weaviate client not initialized")
            
        # Search with filter for legal documents
        result = (
            client.query
            .get("Document", ["document_name", "content", "doc_type"])
            .with_hybrid(
                query=query,
                properties=["content"],
                alpha=0.5  # Balance between keyword and semantic search
            )
            .with_where({
                "operator": "And",
                "operands": [
                    {
                        "path": ["doc_type"],
                        "operator": "Equal",
                        "valueString": doc_type
                    }
                ]
            })
            .with_limit(limit)
            .do()
        )
        
        if result and "data" in result and "Get" in result["data"] and "Document" in result["data"]["Get"]:
            return result["data"]["Get"]["Document"]
        return []
        
    except Exception as e:
        logger.error(f"Error searching legal documents: {str(e)}")
        return []

class GeminiLLM(LLM):
    """Custom LLM class for Gemini API"""
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            headers = { 
                "Content-Type": "application/json" 
            }
            params = {
                "key": GEMINI_API_KEY
            }
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            response = requests.post(
                GEMINI_API_URL,
                headers=headers,
                params=params,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    logger.error("No valid response from Gemini API")
                    return ""
            else:
                logger.error(f"Gemini API request failed with status {response.status_code}: {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return ""
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": "gemini-2.0-flash"}

class DocumentReviewCrew:
    def __init__(self, file_path: str, doc_type: str):
        self.file_path = file_path
        self.doc_type = doc_type
        self.text = self._extract_text()
        self.document_hash = get_document_hash(file_path)
        self.relevant_docs = self._get_relevant_documents()
        self.llm = GeminiLLM()
        
    def _get_relevant_documents(self) -> List[Dict]:
        """Get relevant legal documents for review with caching"""
        try:
            # Use first 500 chars for search to reduce processing time
            search_text = self.text[:500]
            relevant_docs = search_legal_documents(search_text, self.doc_type)
            logger.info(f"Found {len(relevant_docs)} relevant legal documents")
            return relevant_docs
        except Exception as e:
            logger.error(f"Error getting relevant documents: {str(e)}")
            return []
        
    def _extract_text(self) -> str:
        """Extract text from the document file"""
        if not self.file_path:
            raise ValueError("File path is required")
            
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        if not self.file_path.lower().endswith(('.pdf', '.docx', '.doc')):
            raise ValueError("Unsupported file type. Only PDF, DOCX, and DOC files are supported")
            
        try:
            if self.file_path.endswith('.pdf'):
                reader = PdfReader(self.file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            elif self.file_path.endswith(('.docx', '.doc')):
                doc = docx.Document(self.file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from document: {str(e)}")
    
    def create_agents(self) -> List[Agent]:
        """Create specialized agents for document review"""
        return [
            Agent(
                role='Document Analyst',
                goal='Analyze document format, structure, and legal content',
                backstory="""You are an expert in legal document analysis who ensures documents follow 
                proper formatting standards, use correct legal terminology, and meet compliance requirements.""",
                verbose=True,
                llm=self.llm
            )
        ]
    
    def create_tasks(self) -> List[Task]:
        """Create tasks for the document review process"""
        # Prepare context from relevant documents
        context = "\n\n".join([doc["content"] for doc in self.relevant_docs])
        
        return [
            Task(
                description=f"""Analyze this {self.doc_type} document and provide a comprehensive review:
                1. Format and structure analysis
                2. Legal terminology and clause consistency
                3. Compliance with organizational standards
                4. Potential risks and improvements
                
                Document text: {self.text[:1000]}...
                
                Relevant legal documents for reference:
                {context[:1000]}...""",
                agent=self.create_agents()[0]
            )
        ]
    
    def review_document(self) -> dict:
        """Review the document using the CrewAI framework"""
        try:
            logger.info(f"Starting document review for {self.file_path}")
            
            # Create crew with single agent
            crew = Crew(
                agents=self.create_agents(),
                tasks=self.create_tasks(),
                process=Process.sequential,
                verbose=True
            )
            
            # Execute the crew
            result = crew.kickoff()
            
            logger.info("Document review completed successfully")
            
            # Process results
            return {
                "analysis": result,
                "compliance": "Compliance check results...",
                "suggestions": [
                    "Suggestion 1: Improve document structure",
                    "Suggestion 2: Update legal terminology",
                    "Suggestion 3: Add missing clauses"
                ],
                "recommendations": "Overall recommendations for improvement...",
                "reference_documents": [doc["document_name"] for doc in self.relevant_docs]
            }
            
        except Exception as e:
            logger.error(f"Error in document review: {str(e)}")
            raise Exception(f"Error in document review: {str(e)}")

def review_document(file_path: str, doc_type: str) -> dict:
    """Main function to review a document using CrewAI agents"""
    try:
        # Validate inputs
        if not file_path:
            raise ValueError("File path is required")
        if not doc_type:
            raise ValueError("Document type is required")
            
        # Create and run the document review crew
        crew = DocumentReviewCrew(file_path, doc_type)
        return crew.review_document()
        
    except Exception as e:
        logger.error(f"Error in document review: {str(e)}")
        raise Exception(f"Error in document review: {str(e)}") 