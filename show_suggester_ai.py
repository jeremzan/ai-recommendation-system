from fuzzywuzzy import process
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv()


def ask_user(user_input):
    shows_list = user_input.split(",")
    # Check if the user entered more than 1 show or if there is an empty string/whitespaces only
    if len(shows_list) < 2 or not all(show.strip() for show in shows_list):
        return None
    return shows_list

def get_favorite_tv_shows(shows_list):
    known_shows = ["Game of Thrones", "Lupin", "The Witcher", "Breaking Bad", "Stranger Things"]

    if shows_list:
        matched_shows = [process.extractOne(show, known_shows)[0] for show in shows_list]
        return matched_shows
    return None

def extract_description(file_path):
    try:
        df = pd.read_csv(file_path)
        descriptions = df['Description'].tolist()
        return descriptions
    except Exception as e:
        print(f"Error reading CSV file: {e}")



def generate_embeddings(descriptions, client, model="text-embedding-ada-002"):
    with open('embeddings.txt', 'a') as file:  # Open file in append mode
        for description in descriptions:
            response = client.embeddings.create(
                input=description,
                model=model
            )
            embedding = response.data[0].embedding
            file.write(str(embedding) + '\n')  # Convert embedding to string and write to file




if __name__ == "__main__":
    while True:
        user_input = input("Which TV shows did you love watching? Separate them by a comma. Make sure to enter more than 1 show: \n")
        # user_inputs.append(user_input)
        shows_list = ask_user(user_input)
        favorite_shows = get_favorite_tv_shows(shows_list)
        if favorite_shows:
            print(f"Just to make sure, do you mean {', '.join(favorite_shows)}? (y/n) ")
            confirmation = input()
            if confirmation.lower() == 'y':
                print("Great! Generating recommendations...")
                break
            else:
                print("Sorry about that. Let's try again, please make sure to write the names of the TV shows correctly.\n")
        else:
                print("Please enter at least 2 TV shows and don't leave any empty spaces between commas.\n")   

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) #Jeremy, oublie pas dde rentrer ton API key dans un .env :)
    if not os.path.exists('./embeddings.txt'):
        descriptions = extract_description('./imdb_tvshows - imdb_tvshows.csv') 
        generate_embeddings(descriptions, client)

   

# Example usage
# favorite_shows = get_favorite_tv_shows()
