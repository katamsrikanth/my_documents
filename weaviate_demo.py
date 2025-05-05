import weaviate
import json
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import textwrap

# Load environment variables
load_dotenv()

# Initialize Weaviate client
client = weaviate.Client(
    url="http://localhost:8080"
)

# Define a simple schema
class_obj = {
    "class": "Document",
    "description": "A collection of documents",
    "properties": [
        {
            "name": "title",
            "dataType": ["string"],
            "description": "The title of the document"
        },
        {
            "name": "content",
            "dataType": ["text"],
            "description": "The content of the document"
        }
    ]
}

def create_schema():
    """Create the schema in Weaviate"""
    try:
        # Delete existing schema if it exists
        try:
            client.schema.delete_class("Document")
            print("Deleted existing Document class")
        except Exception as e:
            print(f"No existing schema to delete: {e}")

        # Create new schema
        client.schema.create_class(class_obj)
        print("Schema created successfully!")
        
        # Verify schema was created
        schema = client.schema.get()
        print("Current schema:", schema)
        
    except Exception as e:
        print(f"Error creating schema: {e}")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def create_chunks(text, chunk_size=1000):
    """Split text into chunks of approximately equal size"""
    return textwrap.wrap(text, chunk_size, break_long_words=False, break_on_hyphens=False)

def process_and_upload_pdf(pdf_path, chunk_size=1000):
    """Process a PDF file and upload its chunks to Weaviate"""
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return False

    # Create chunks
    chunks = create_chunks(text, chunk_size)
    filename = os.path.basename(pdf_path)
    
    print(f"\nProcessing PDF: {filename}")
    print(f"Created {len(chunks)} chunks")

    # Upload chunks to Weaviate
    for i, chunk in enumerate(chunks):
        try:
            client.data_object.create(
                class_name="Document",
                data_object={
                    "title": filename,
                    "content": chunk
                }
            )
            print(f"Uploaded chunk {i+1}/{len(chunks)}")
        except Exception as e:
            print(f"Error uploading chunk {i}: {e}")
            return False
    
    return True

def perform_search(query):
    """Perform a semantic search on the collection"""
    try:
        print(f"\nPerforming search for: '{query}'")
        
        # Create the query
        where_filter = {
            "operator": "Like",
            "path": ["content"],
            "valueText": f"*{query}*"
        }
        
        result = (
            client.query
            .get("Document", ["title", "content"])
            .with_where(where_filter)
            .with_limit(3)  # Show more results for document chunks
            .do()
        )
        
        if "data" in result and "Get" in result["data"] and "Document" in result["data"]["Get"]:
            chunks = result["data"]["Get"]["Document"]
            if chunks:
                print("\nSearch Results:")
                for chunk in chunks:
                    print(f"\nFile: {chunk['title']}")
                    print(f"Content: {chunk['content'][:200]}...")  # Show first 200 chars
                    print("-" * 50)
            else:
                print("No matching documents found.")
        else:
            print("No results found in the expected format.")
            print("Raw response:", result)
    except Exception as e:
        print(f"Error performing search: {e}")

def main():
    print("Creating Weaviate schema...")
    create_schema()
    
    # Example usage with a PDF file
    pdf_path = input("Enter the path to your PDF file (or press Enter to skip): ").strip()
    if pdf_path:
        if os.path.exists(pdf_path):
            success = process_and_upload_pdf(pdf_path)
            if success:
                print("\nPDF processed and uploaded successfully!")
        else:
            print("Error: PDF file not found!")
    
    # Perform a sample search
    search_query = input("\nEnter your search query: ").strip()
    if search_query:
        perform_search(search_query)

if __name__ == "__main__":
    main() 