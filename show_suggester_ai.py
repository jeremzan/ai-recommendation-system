from fuzzywuzzy import process
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import pickle
import numpy as np
from PIL import Image
from io import BytesIO


load_dotenv()


def user_input_to_shows_list(user_input):
    shows_list = user_input.split(",")    
    shows_list_stripped = [show.strip() for show in shows_list]
    return shows_list_stripped

def get_favorite_tv_shows(shows_list, known_shows):

    if shows_list:
        # Check  if there is an empty string/whitespaces only
        if not all(shows_list):
            return None
        
        matched_shows = [process.extractOne(show, known_shows)[0] for show in shows_list]

        # Check if the user entered more than 1 show 
        if len(set(matched_shows)) < 2 :
            return None

        return list(set(matched_shows)) # Remove repeated shows
    
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

    all_distances = {}
    for key, element in embeddings.items():
        if key in favorite_shows:
            continue
        distance = cosine_similarity(average, element)
        all_distances[key] = distance

    smallest_elements = sorted(all_distances.items(), key=lambda x: x[1], reverse=True)[:5]
    recommended_shows = [x[0] for x in smallest_elements]
    # print(smallest_elements)
    # print(recommended_shows)
    return recommended_shows



def generate_show_descriptions(favorite_shows, recommended_shows, client, model="gpt-3.5-turbo"):

    # Convert lists to comma-separated strings
    favorite_shows_str = ', '.join(favorite_shows)
    recommended_shows_str = ', '.join(recommended_shows)

    # Create prompts and call OpenAI ChatGPT API
    prompt1 = f"""Create a concept for a new TV show based on: {favorite_shows_str}. 
    The format of you response should be:
    Title : <title of the show>
    Concept: <description of the show>
    Plot: <description of the plot>"""
    prompt2 = f"""Create a concept for a new TV show based on: {recommended_shows_str}. 
    The format of you response should be:
    Title : <title of the show>
    Concept: <description of the show>
    Plot: <description of the plot>"""

    response1 = client.chat.completions.create(
        model=model, 
        messages=[
            {   "role": "user",
                "content": prompt1  }])
    response2 = client.chat.completions.create(
        model=model, 
        messages=[
            {   "role": "user",
                "content": prompt2  }])
    
    content1 = response1.choices[0].message.content
    content2 = response2.choices[0].message.content

    return content1, content2

def extract_show_name(description):
    # Extract the show name from the line starting with "Title: "
    for line in description.split('\n'):
        if line.startswith("Title: "):
            return line.replace("Title: ", "").strip()
    return "Unknown Show"  # Fallback in case the title line is not found

def extract_concept(description):
    concept_start = description.find("Concept:")
    plot_start = description.find("Plot:", concept_start)

    if concept_start != -1 and plot_start != -1:
        concept_text = description[concept_start + 7:plot_start].strip()  # +7 to skip "Concept:" part
    elif concept_start != -1:
        concept_text = description[concept_start + 7:].strip()  # +7 to skip "Concept:" part
    else:
        return "Concept not available"  # Fallback if "Concept:" is not found

    return concept_text

def generate_show_ads(description1, description2):
    # Create prompts and call OpenAI DALL-E API
    prompt1 = f"Create an advertisement for a TV show based on the description: {description1}"
    prompt2 = f"Create an advertisement for a TV show based on the description: {description2}"

    image_data1 = client.images.generate(
        model="dall-e-3",
        prompt=prompt1,
        size="1024x1024",
        quality="standard",
        n=1,)
    image_data2 = client.images.generate(
        model="dall-e-3",
        prompt=prompt2,
        size="1024x1024",
        quality="standard",
        n=1,)

    # Decode and save or display images
    image1 = Image.open(BytesIO(image_data1))
    image2 = Image.open(BytesIO(image_data2))

    image1.show()  # This will open the image using the default viewer
    image2.show()  

    # # If you want to save the images
    # image1.save("show1_ad.jpg")
    # image2.save("show2_ad.jpg")

    # # If you want to open saved images using OS default viewer
    # os.system("open show1_ad.jpg")  # For MacOS
    # os.system("start show1_ad.jpg")  # For Windows



if __name__ == "__main__":

    while True:
        user_input = input("Which TV shows did you love watching? Separate them by a comma. Make sure to enter more than 1 show: \n")
        shows_list = user_input_to_shows_list(user_input)

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


    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")
        exit()

    if not os.path.exists('./embeddings.pkl'):
        generate_embeddings(known_tv_shows, client)
        
    with open('./embeddings.pkl', 'rb') as embedding_file:
        embeddings = pickle.load(embedding_file)

    recommended_shows = find_matching_shows(favorite_shows, embeddings)

    content1, content2 = generate_show_descriptions(favorite_shows, recommended_shows, client)

    show1name = extract_show_name(content1)
    show2name = extract_show_name(content2)
    concept1 = extract_concept(content1)
    concept2 = extract_concept(content2)
    # generate_show_ads(content1, content2)
   


    # Final message
    final_message = (
        f"I have also created just for you two shows which I think you would love. "
        f"Show #1 is based on the fact that you loved the input shows that you gave me. "
        f"Its name is {show1name} and it is about {concept1}\n"
        f"Show #2 is based on the shows that I recommended for you. "
        f"Its name is {show2name} and it is about {concept2} "
        f"Here are also the 2 tv show ads. Hope you like them!"
    )
    print(final_message)
