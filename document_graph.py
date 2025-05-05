from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentState(TypedDict):
    """State for document generation workflow"""
    title: str
    requirements: str
    doc_type: str
    vector_knowledge: str
    research_analysis: str
    document_outline: str
    draft_document: str
    final_document: str
    messages: Sequence[HumanMessage | AIMessage]

class DocumentGraph:
    def __init__(self, vector_knowledge: str):
        self.vector_knowledge = vector_knowledge
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        
    def create_research_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an experienced legal research specialist with expertise in analyzing 
            legal documents and extracting key information. Your role is to identify relevant legal 
            principles, precedents, and requirements from the provided knowledge base."""),
            ("human", """Analyze the following vector knowledge and extract relevant legal information 
            for a {doc_type} document titled '{title}' with requirements: {requirements}
            
            Vector Knowledge:
            {vector_knowledge}
            
            Focus on:
            1. Legal principles and precedents
            2. Relevant case law
            3. Statutory requirements
            4. Industry standards
            5. Best practices
            
            Provide a detailed analysis of the relevant information.""")
        ])
    
    def create_outline_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a document structure expert specializing in legal document formatting.
            You ensure documents follow proper legal formatting standards and include all necessary sections."""),
            ("human", """Based on the research analysis, create a detailed outline for a {doc_type} 
            document titled '{title}'. Ensure the outline follows standard legal document structure 
            and includes all necessary sections.
            
            Research Analysis:
            {research_analysis}
            
            Consider the following document types:
            - Contracts and Agreements
            - Wills and Trusts
            - Legal Opinions and Memorandums
            - Court Documents
            - Corporate Documents
            - Client Forms and Letters
            
            Provide a comprehensive outline with main sections and subsections.""")
        ])
    
    def create_draft_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a skilled legal content writer with experience in drafting various 
            types of legal documents. You ensure the content is precise, legally sound, and follows 
            proper legal terminology."""),
            ("human", """Using the research analysis and document outline, draft the complete 
            {doc_type} document titled '{title}'. Ensure:
            1. Use precise legal terminology
            2. Follow the provided structure
            3. Include all necessary legal formalities
            4. Maintain consistency with the vector knowledge
            5. Use proper legal document formatting
            
            Research Analysis:
            {research_analysis}
            
            Document Outline:
            {document_outline}
            
            Format the document using HTML tags for proper display.""")
        ])
    
    def create_review_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a meticulous legal review specialist who ensures documents are 
            complete, accurate, and meet all legal requirements. You catch any inconsistencies or 
            missing elements."""),
            ("human", """Review the generated {doc_type} document titled '{title}' and ensure:
            1. All legal requirements are met
            2. The document is complete and accurate
            3. There are no inconsistencies
            4. The formatting is correct
            5. All necessary sections are included
            
            Draft Document:
            {draft_document}
            
            Provide the final, polished document with any necessary revisions.""")
        ])
    
    def research_agent(self, state: DocumentState) -> DocumentState:
        """Agent for legal research analysis"""
        prompt = self.create_research_prompt()
        chain = prompt | self.llm
        response = chain.invoke({
            "title": state["title"],
            "requirements": state["requirements"],
            "doc_type": state["doc_type"],
            "vector_knowledge": state["vector_knowledge"]
        })
        state["research_analysis"] = response.content
        return state
    
    def outline_agent(self, state: DocumentState) -> DocumentState:
        """Agent for creating document outline"""
        prompt = self.create_outline_prompt()
        chain = prompt | self.llm
        response = chain.invoke({
            "title": state["title"],
            "doc_type": state["doc_type"],
            "research_analysis": state["research_analysis"]
        })
        state["document_outline"] = response.content
        return state
    
    def draft_agent(self, state: DocumentState) -> DocumentState:
        """Agent for drafting the document"""
        prompt = self.create_draft_prompt()
        chain = prompt | self.llm
        response = chain.invoke({
            "title": state["title"],
            "doc_type": state["doc_type"],
            "research_analysis": state["research_analysis"],
            "document_outline": state["document_outline"]
        })
        state["draft_document"] = response.content
        return state
    
    def review_agent(self, state: DocumentState) -> DocumentState:
        """Agent for reviewing and finalizing the document"""
        prompt = self.create_review_prompt()
        chain = prompt | self.llm
        response = chain.invoke({
            "title": state["title"],
            "doc_type": state["doc_type"],
            "draft_document": state["draft_document"]
        })
        state["final_document"] = response.content
        return state
    
    def create_graph(self) -> StateGraph:
        """Create the document generation workflow graph"""
        workflow = StateGraph(DocumentState)
        
        # Add nodes
        workflow.add_node("research", self.research_agent)
        workflow.add_node("outline", self.outline_agent)
        workflow.add_node("draft", self.draft_agent)
        workflow.add_node("review", self.review_agent)
        
        # Add edges
        workflow.add_edge("research", "outline")
        workflow.add_edge("outline", "draft")
        workflow.add_edge("draft", "review")
        workflow.add_edge("review", END)
        
        # Set entry point
        workflow.set_entry_point("research")
        
        return workflow.compile()
    
    def generate_document(self, title: str, requirements: str, doc_type: str) -> str:
        """Generate a document using the LangGraph workflow"""
        try:
            logger.info(f"Starting document generation for {title}")
            
            # Initialize state
            initial_state = DocumentState(
                title=title,
                requirements=requirements,
                doc_type=doc_type,
                vector_knowledge=self.vector_knowledge,
                research_analysis="",
                document_outline="",
                draft_document="",
                final_document="",
                messages=[]
            )
            
            # Create and run the graph
            graph = self.create_graph()
            final_state = graph.invoke(initial_state)
            
            logger.info("Document generation completed successfully")
            return final_state["final_document"]
            
        except Exception as e:
            logger.error(f"Error in document generation: {str(e)}")
            return f"Error generating document: {str(e)}" 