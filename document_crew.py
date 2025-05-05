from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentCrew:
    def __init__(self, vector_knowledge: str):
        self.vector_knowledge = vector_knowledge
        
    def create_agents(self) -> List[Agent]:
        """Create specialized agents for document generation"""
        return [
            Agent(
                role='Legal Research Specialist',
                goal='Analyze and extract relevant legal information from vector knowledge',
                backstory="""You are an experienced legal research specialist with expertise in analyzing 
                legal documents and extracting key information. Your role is to identify relevant legal 
                principles, precedents, and requirements from the provided knowledge base.""",
                verbose=True
            ),
            Agent(
                role='Document Structure Expert',
                goal='Create a well-structured document outline based on document type',
                backstory="""You are a document structure expert specializing in legal document formatting.
                You ensure documents follow proper legal formatting standards and include all necessary sections.""",
                verbose=True
            ),
            Agent(
                role='Legal Content Writer',
                goal='Generate comprehensive legal content based on research and structure',
                backstory="""You are a skilled legal content writer with experience in drafting various 
                types of legal documents. You ensure the content is precise, legally sound, and follows 
                proper legal terminology.""",
                verbose=True
            ),
            Agent(
                role='Legal Review Specialist',
                goal='Review and refine the generated document',
                backstory="""You are a meticulous legal review specialist who ensures documents are 
                complete, accurate, and meet all legal requirements. You catch any inconsistencies or 
                missing elements.""",
                verbose=True
            )
        ]
    
    def create_tasks(self, title: str, requirements: str, doc_type: str) -> List[Task]:
        """Create tasks for the document generation process"""
        return [
            Task(
                description=f"""Analyze the following vector knowledge and extract relevant legal information 
                for a {doc_type} document titled '{title}' with requirements: {requirements}
                
                Vector Knowledge:
                {self.vector_knowledge}
                
                Focus on:
                1. Legal principles and precedents
                2. Relevant case law
                3. Statutory requirements
                4. Industry standards
                5. Best practices
                
                Provide a detailed analysis of the relevant information.""",
                agent=self.create_agents()[0]
            ),
            Task(
                description=f"""Based on the research analysis, create a detailed outline for a {doc_type} 
                document titled '{title}'. Ensure the outline follows standard legal document structure 
                and includes all necessary sections.
                
                Consider the following document types:
                - Contracts and Agreements
                - Wills and Trusts
                - Legal Opinions and Memorandums
                - Court Documents
                - Corporate Documents
                - Client Forms and Letters
                
                Provide a comprehensive outline with main sections and subsections.""",
                agent=self.create_agents()[1]
            ),
            Task(
                description=f"""Using the research analysis and document outline, draft the complete 
                {doc_type} document titled '{title}'. Ensure:
                1. Use precise legal terminology
                2. Follow the provided structure
                3. Include all necessary legal formalities
                4. Maintain consistency with the vector knowledge
                5. Use proper legal document formatting
                
                Format the document using HTML tags for proper display.""",
                agent=self.create_agents()[2]
            ),
            Task(
                description=f"""Review the generated {doc_type} document titled '{title}' and ensure:
                1. All legal requirements are met
                2. The document is complete and accurate
                3. There are no inconsistencies
                4. The formatting is correct
                5. All necessary sections are included
                
                Provide the final, polished document with any necessary revisions.""",
                agent=self.create_agents()[3]
            )
        ]
    
    def generate_document(self, title: str, requirements: str, doc_type: str) -> str:
        """Generate a document using the CrewAI framework"""
        try:
            logger.info(f"Starting document generation for {title}")
            
            # Create crew
            crew = Crew(
                agents=self.create_agents(),
                tasks=self.create_tasks(title, requirements, doc_type),
                process=Process.sequential
            )
            
            # Execute the crew
            result = crew.kickoff()
            
            logger.info("Document generation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in document generation: {str(e)}")
            return f"Error generating document: {str(e)}" 