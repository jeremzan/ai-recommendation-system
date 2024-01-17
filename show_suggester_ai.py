from fuzzywuzzy import process
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv()

def ask_user():

    flag = True

    while flag :
        # Prompt the user for input
        user_input = input("Which TV shows did you love watching? Separate them by a comma. Make sure to enter more than 1 show: \n")
        shows_list = user_input.split(",")

        #Check if the user entered 3 TV shows
        if len(shows_list) < 2:
            print("Please enter at least 2 TV shows.\n")

        #Check if the user entered whitespaces or empty strings
        elif not all(show.strip() for show in shows_list):
            print("Please enter non-empty TV shows.\n")

        else :
            flag = False

    return shows_list


def get_favorite_tv_shows():
    # Sample list of TV shows for fuzzy matching
    known_shows = ["Game of Thrones", "Lupin", "The Witcher", "Breaking Bad", "Stranger Things"]

    while True:
        shows = ask_user()

        matched_shows = [process.extractOne(show, known_shows)[0] for show in shows]
        confirmation = input(f"Just to make sure, do you mean {', '.join(matched_shows)}? (y/n) \n")

        if confirmation.lower() == 'y':
            print("Great! Generating recommendations...")
            return matched_shows
        else:
            print("Sorry about that. Let's try again, please make sure to write the names of the TV shows correctly.\n")
    

def extract_description(file_path):
    try:
        df = pd.read_csv(file_path)
        descriptions = df['Description'].tolist()
        return descriptions
    except Exception as e:
        print(f"Error reading CSV file: {e}")



def generate_embeddings(file_path, client, model="text-embedding-ada-002"):
    descriptions = extract_description(file_path)
    with open('embeddings.txt', 'a') as file:  # Open file in append mode
        for description in descriptions:
            response = client.embeddings.create(
                input=description,
                model=model
            )
            embedding = response.data[0].embedding
            file.write(str(embedding) + '\n')  # Convert embedding to string and write to file




if __name__ == "__main__":
   client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) #Jeremy, oublie pas dde rentrer ton API key dans un .env :)
   generate_embeddings('./imdb_tvshows - imdb_tvshows.csv', client)
   

# Example usage
# favorite_shows = get_favorite_tv_shows()
