from models.user import User

def create_sample_user():
    # Create a test user with email as username
    test_user = User(username="test1234@gmail.com", password="test1234")
    
    try:
        if test_user.save():
            print("Successfully created test user!")
            print("Username: test1234@gmail.com")
            print("Password: test1234")
        else:
            print("User already exists!")
    except Exception as e:
        print(f"Error creating user: {str(e)}")

if __name__ == "__main__":
    create_sample_user() 