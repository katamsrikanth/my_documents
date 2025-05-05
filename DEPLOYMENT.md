# Production Deployment Guide

## Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Setup Instructions

1. Clean up the project:
```bash
python cleanup.py
```

2. Create a new virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On Unix or MacOS
source venv/bin/activate
```

4. Install production dependencies:
```bash
pip install -r requirements.txt
```

5. Set up environment variables:
Create a `.env` file with the following variables:
```
GOOGLE_API_KEY=your_api_key
FLASK_ENV=production
FLASK_APP=app.py
```

## Running in Production

1. Using Gunicorn (recommended):
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. Using Flask development server (not recommended for production):
```bash
flask run --host=0.0.0.0 --port=5000
```

## Project Structure
```
doc-search-app/
├── app.py                 # Main application file
├── document_review_crew.py # Document review implementation
├── document_crew.py       # Document generation implementation
├── document_graph.py      # Document graph implementation
├── requirements.txt       # Production dependencies
├── cleanup.py            # Cleanup script
├── templates/            # HTML templates
├── temp/                # Temporary files (auto-cleaned)
└── venv/                # Virtual environment
```

## Maintenance

1. Regular cleanup:
```bash
python cleanup.py
```

2. Update dependencies:
```bash
pip freeze > requirements.txt
```

## Security Considerations

1. Never commit the `.env` file
2. Keep API keys secure
3. Use HTTPS in production
4. Implement proper access controls
5. Regular security updates

## Monitoring

1. Check logs regularly
2. Monitor disk space
3. Monitor API usage
4. Set up error alerts

## Backup

1. Regular database backups
2. Document storage backups
3. Configuration backups 