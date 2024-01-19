from fuzzywuzzy import process
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import pickle
import numpy as np



load_dotenv()


def ask_user(user_input):
    shows_list = user_input.split(",")
    # Check if the user entered more than 1 show or if there is an empty string/whitespaces only
    if len(shows_list) < 2 or not all(show.strip() for show in shows_list):
        return None
    return shows_list

def get_favorite_tv_shows(shows_list, known_shows):
    if shows_list:
        matched_shows = [process.extractOne(show, known_shows)[0] for show in shows_list]
        return matched_shows
    return None

def read_csv_file(file_path):
    try:
        tv_shows = pd.read_csv(file_path)
        return tv_shows
    except Exception as e:
        print(f"Error reading CSV file: {e}")



def generate_embeddings(tv_shows, client, model="text-embedding-ada-002"):
    embeddings_dict = {}
    titles = tv_shows['Title'].tolist()
    descriptions = tv_shows['Description'].tolist()
    counter = 0
    for index, description in enumerate(descriptions):
        response = client.embeddings.create(
            input=description,
            model=model
        )
        embedding = response.data[0].embedding
        embeddings_dict[titles[index]] = embedding
        print(counter)
        counter +=1

    with open('embeddings.pkl', 'wb') as file:  # Open file in binary write mode
        pickle.dump(embeddings_dict, file)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_matching_shows(favorite_shows, embeddings):
    favorite_embeddings = []
    for show in favorite_shows:
        favorite_embeddings.append(embeddings[show])
    
    #Compute average of favorite shows
    average = np.mean(favorite_embeddings, axis=0)

    #Compute distance 
    all_distances = compute_distances(favorite_shows, embeddings, average)

    smallest_elements = sorted(all_distances.items(), key=lambda x: x[1])[:5]
    best_matches = [x[0] for x in smallest_elements]
    print(smallest_elements)
    print(best_matches)

def compute_distances(favorite_shows, embeddings, average):
    all_distances = {}
    for key, element in embeddings.items():
        if key in favorite_shows:
            continue
        distance = cosine_similarity(average, element)
        all_distances[key] = distance
    return all_distances


if __name__ == "__main__":
    while True:
        user_input = input("Which TV shows did you love watching? Separate them by a comma. Make sure to enter more than 1 show: \n")
        # user_inputs.append(user_input)
        shows_list = ask_user(user_input)
        known_tv_shows = read_csv_file('./imdb_tvshows - imdb_tvshows.csv') 
        favorite_shows = get_favorite_tv_shows(shows_list, known_tv_shows['Title'])
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


    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    if not os.path.exists('./embeddings.pkl'):
        generate_embeddings(known_tv_shows, client)
        
    with open('./embeddings.pkl', 'rb') as embedding_file:
        embeddings = pickle.load(embedding_file)

    find_matching_shows(favorite_shows, embeddings) 


# Example usage
# favorite_shows = get_favorite_tv_shows()
