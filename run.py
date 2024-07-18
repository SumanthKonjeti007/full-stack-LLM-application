import os
from app import create_app
from dotenv import load_dotenv

def create_directories():
    # Define the directory structure
    directories = [
        'app',
        'app/api',
        'app/services',
        'app/utils'
    ]

    # Create the directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("Directory structure created successfully.")

# Load environment variables
load_dotenv()


# Create the application
app = create_app()

if __name__ == '__main__':
    #create_directories()
    app.run(debug=True)
