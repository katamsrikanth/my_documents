# Weaviate Vector Database Demo

This is a simple Python application that demonstrates how to:
1. Connect to a Weaviate vector database
2. Create a collection (class)
3. Add sample data
4. Perform semantic search

## Prerequisites

- Python 3.7 or higher
- Weaviate instance running (local or cloud)
- pip (Python package manager)

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have a Weaviate instance running. By default, the application connects to `http://localhost:8080`. If your Weaviate instance is running at a different URL, modify the `url` parameter in `weaviate_demo.py`.

## Running the Application

Simply run:
```bash
python weaviate_demo.py
```

The application will:
1. Create a schema for the "Article" collection
2. Add sample articles about AI and machine learning
3. Perform a sample semantic search

## Customization

You can modify the sample data in the `add_sample_data()` function and change the search query in the `main()` function to experiment with different searches.

## Notes

- The application uses the `text2vec-transformers` vectorizer for text embedding
- The search results are limited to 2 matches by default
- Error handling is implemented for all major operations 