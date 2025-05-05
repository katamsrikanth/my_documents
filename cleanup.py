import os
import shutil
import glob

def cleanup_project():
    """Clean up the project for production deployment"""
    # Remove __pycache__ directories
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            shutil.rmtree(os.path.join(root, '__pycache__'))
    
    # Clean up temp directory
    if os.path.exists('temp'):
        shutil.rmtree('temp')
        os.makedirs('temp')
    
    # Remove .pyc files
    for pyc_file in glob.glob('**/*.pyc', recursive=True):
        os.remove(pyc_file)
    
    # Remove .DS_Store files (if any)
    for ds_file in glob.glob('**/.DS_Store', recursive=True):
        os.remove(ds_file)
    
    # Remove old virtual environment
    if os.path.exists('venv'):
        shutil.rmtree('venv')
    
    print("Project cleaned up successfully!")

if __name__ == "__main__":
    cleanup_project() 