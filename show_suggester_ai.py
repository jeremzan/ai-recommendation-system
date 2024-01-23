import logging
from fuzzywuzzy import process
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
import math
import requests
from colorama import Fore, init
init(autoreset=True)


logging.basicConfig(level=logging.INFO, format='%(message)s')
for logger_name, logger_obj in logging.Logger.manager.loggerDict.items():
    if hasattr(logger_obj, 'setLevel'):
        logger_obj.setLevel(logging.WARNING)


load_dotenv()

def user_input_to_shows_list(user_input):
    """
    Convert user input string to a list of TV show names.

    Args:
        user_input (str): User input string with TV show names separated by commas.

    Returns:
        list: List of TV show names with leading and trailing whitespaces removed and without empty strings.
    """
    shows_list = user_input.split(",")    
    shows_list_stripped = [show.strip() for show in shows_list]
    shows_list_stripped = list(filter(None, shows_list_stripped))   # Remove empty strings from the list
    return shows_list_stripped

def get_favorite_tv_shows(shows_list, known_shows):
    """
    Get the user's favorite TV shows based on matching with known shows.

    Args:
        shows_list (list): List of user-provided TV show names.
        known_shows (list): List of known TV show names.

    Returns:
        list: List of favorite TV show names that match the known TV shows, or None if criteria are not met.
    """
    
    matched_shows = [process.extractOne(show, known_shows)[0] for show in shows_list] 
    matched_shows_without_rep = list(set(matched_shows))    # Remove repeated shows

    # Check if the user entered more than 1 show 
    if  len(matched_shows_without_rep) < 2 :
        return None

    return matched_shows_without_rep
    
def read_csv_file(file_path):
    """
    Read data from a CSV file into a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing data from the CSV file, or None if there is an error reading the file.
    """
    try:
        tv_shows = pd.read_csv(file_path)
        return tv_shows
    except Exception as e:
        logging.error(Fore.RED + f"Error reading CSV file: {e}")

def generate_embeddings(tv_shows, client, model="text-embedding-ada-002"):
    """
    Generate text embeddings for TV show descriptions and save them to a file.

    Args:
        tv_shows (pd.DataFrame): DataFrame containing TV show data with 'Title' and 'Description' columns.
        client (OpenAI): OpenAI API client.
        model (str, optional): OpenAI text embedding model (default is "text-embedding-ada-002").

    Returns:
        None
    """
    embeddings_dict = {}
    titles = tv_shows['Title'].tolist()
    descriptions = tv_shows['Description'].tolist()
    for index, description in enumerate(descriptions):
        response = client.embeddings.create(
            input=description,
            model=model
        )
        embedding = response.data[0].embedding
        embeddings_dict[titles[index]] = embedding


    with open('embeddings.pkl', 'wb') as file:  # Open file in binary write mode
        pickle.dump(embeddings_dict, file)

def cosine_similarity(a, b):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        a (np.array): First vector.
        b (np.array): Second vector.

    Returns:
        float: Cosine similarity between vectors a and b.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_distances(favorite_shows, embeddings, average):
    """
    Calculate cosine distances between the average embedding and other TV show embeddings.

    Args:
        favorite_shows (list): List of user's favorite TV show names.
        embeddings (dict): Dictionary mapping TV show names to embeddings.
        average (np.array): Average embedding of the user's favorite shows.

    Returns:
        dict: A dictionary containing TV show names as keys and cosine distances as values.
    """
    distances_dict = {}
    for show, embedding in embeddings.items():
        if show in favorite_shows:
            continue
        distance = cosine_similarity(average, embedding)
        distances_dict[show] = distance
    return distances_dict

def find_matching_shows(favorite_shows, embeddings):
    """
    Find TV shows that match the user's favorite shows based on embeddings.
    Apply linear transformation to the cosine similarity scores.

    Args:
        favorite_shows (list): List of user's favorite TV show names.
        embeddings (dict): Dictionary mapping TV show names to embeddings.

    Returns:
        list: List of recommended TV show names with adjusted percentages.
    """
    favorite_embeddings = [embeddings[show] for show in favorite_shows]
    average_array = np.mean(favorite_embeddings, axis=0)
    distances_dict = compute_distances(favorite_shows, embeddings, average_array)

    # Get the minimum and maximum distances
    min_distance = min(distances_dict.values())
    max_distance = max(distances_dict.values())

    # Linearly scale distances to a range of 10% to 100%
    #scaled_distances = {
    #    show: int(10 + 90 * (distance - min_distance) / (max_distance - min_distance))
    #    for show, distance in distances_dict.items()
    #}

    # # Linearly scale distances to a range of 10% to 98%
    scaled_distances = {
         show: int(10 + 88 * (distance - min_distance) / (max_distance - min_distance))
         for show, distance in distances_dict.items()
     }

    # Sort and select top 5 shows
    best_recommendation_shows = sorted(scaled_distances.items(), key=lambda x: x[1], reverse=True)[:5]
    return dict(best_recommendation_shows)

def generate_show_descriptions(favorite_shows, recommended_shows, client, model="gpt-3.5-turbo"):
    """
    Generate TV show concepts and plots based on favorite and recommended shows.

    Args:
        favorite_shows (list): List of user's favorite TV show names.
        recommended_shows (list): List of recommended TV show names.
        client (OpenAI): OpenAI API client.
        model (str, optional): OpenAI language model (default is "gpt-3.5-turbo").

    Returns:
        tuple: Two generated TV show descriptions as strings.
    """

    # Convert lists to comma-separated strings
    favorite_shows_str = ', '.join(favorite_shows)
    recommended_shows_str = ', '.join(recommended_shows)

    # Create prompts and call OpenAI ChatGPT API
    prompt1 = f"""Create a concept for a new TV show based on: {favorite_shows_str}. 
    The format of you response should be:
    Title : <title of the show>
    Concept: <title of the show> is a <description of the show>
    Plot: <description of the plot>"""
    prompt2 = f"""Create a concept for a new TV show based on: {recommended_shows_str}. 
    The format of you response should be:
    Title : <title of the show>
    Concept: <title of the show> is a <description of the show>
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
    """
    Extract the TV show name from a description.

    Args:
        description (str): TV show description.

    Returns:
        str: Extracted TV show name or "Unknown Show" if not found.
    """
    # Extract the show name from the line starting with "Title: "
    for line in description.split('\n'):
        if line.startswith("Title: "):
            return line.replace("Title: ", "").strip()
    return "Unknown Show"  # Fallback in case the title line is not found

def extract_concept(description):
    """
    Extract the TV show concept from a description.

    Args:
        description (str): TV show description.

    Returns:
        str: Extracted TV show concept or "Concept not available" if not found.
    """
    concept_label = "Concept:"
    plot_label = "Plot:"
    key_phrase = "is a"

    # Find the start of the "Concept:" section
    concept_start = description.find(concept_label)

    if concept_start != -1:
        # Find the start of the "Plot:" section
        plot_start = description.find(plot_label, concept_start)

        # Extract everything from "Concept:" to "Plot:"
        if plot_start != -1:
            concept_text = description[concept_start:plot_start].strip()
        else:
            concept_text = description[concept_start:].strip()

        # Find and skip past "is a" in the concept text
        key_phrase_index = concept_text.find(key_phrase)
        if key_phrase_index != -1:
            # Start extraction from just after "is a"
            start_index = key_phrase_index + len(key_phrase)
            concept_text = concept_text[start_index:].strip()
        else:
            return "Detailed concept description not available"

        return concept_text

    return "Concept not available"  # Fallback if "Concept:" is not found

def extract_plot(description):
    plot_label = "Plot:"
    plot_start = description.find(plot_label)

    if plot_start != -1:
        # Start extraction right after "Plot:"
        start_index = plot_start + len(plot_label)
        plot_text = description[start_index:].strip()
        return plot_text

    return "Plot not available"  # Fallback if "Plot:" is not found

def generate_show_ads(plot1, plot2, client):
    """
    Generate advertisements for TV shows based on their descriptions using DALL-E API.

    Args:
        description1 (str): TV show description for the first show.
        description2 (str): TV show description for the second show.

    Returns:
        None (displays generated images).
    """
    prompt1 = f"Create a visually striking movie poster based on the following description: {plot1}. The poster should feature the main characters, key setting, and central theme of the show. Emphasize the mood and genre through color scheme and composition. Include a subtle title in a font style that matches the show's atmosphere. Minimize text to ensure the focus remains on the visual elements. The poster should vividly represent the essence of the show without relying heavily on text."
    prompt2 = f"Create a visually striking movie poster based on the following description: {plot2}. The poster should feature the main characters, key setting, and central theme of the show. Emphasize the mood and genre through color scheme and composition. Include a subtle title in a font style that matches the show's atmosphere. Minimize text to ensure the focus remains on the visual elements. The poster should vividly represent the essence of the show without relying heavily on text."

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


    image_url1 = image_data1.data[0].url
    image_url2 = image_data2.data[0].url

     # Download and display the first image
    response1 = requests.get(image_url1)
    if response1.status_code == 200:
        image1 = Image.open(BytesIO(response1.content))
        image1.show()

    # Download and display the second image
    response2 = requests.get(image_url2)
    if response2.status_code == 200:
        image2 = Image.open(BytesIO(response2.content))
        image2.show()

    return image_url1, image_url2

def create_openai_client():
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    except Exception as e:
        logging.error(f"Error creating OpenAI client: {e}")
        exit()
    return client

if __name__ == "__main__":

    while True:
        user_input = input("Which TV shows did you love watching? Separate them by a comma. Make sure to enter more than 1 show: \n")
        shows_list = user_input_to_shows_list(user_input)

        known_tv_shows = read_csv_file('./imdb_tvshows - imdb_tvshows.csv') 
        favorite_shows = get_favorite_tv_shows(shows_list, known_tv_shows['Title'])

        if favorite_shows:
            logging.info(f"Just to make sure, do you mean {', '.join(favorite_shows)}? (y/n) ")
            confirmation = input()
            if confirmation.lower() == 'y':
                logging.info(Fore.GREEN + "Great! Generating recommendations...")
                break
            else:
                logging.info("Sorry about that. Let's try again, please make sure to write the names of the TV shows correctly.\n")
        else:
            logging.warning(Fore.RED + "Please enter at least 2 different TV shows.\n")   

    client = create_openai_client()

    if not os.path.exists('./embeddings.pkl'):
        generate_embeddings(known_tv_shows, client)
        
    with open('./embeddings.pkl', 'rb') as embedding_file:
        embeddings = pickle.load(embedding_file)

    recommended_shows = find_matching_shows(favorite_shows, embeddings)

    #Print recommanded shows and percentage
    for show, percentage in recommended_shows.items():
        logging.info(Fore.YELLOW + f'{show} ({percentage}%)')

    content1, content2 = generate_show_descriptions(favorite_shows, recommended_shows, client)

    show1name = extract_show_name(content1)
    show2name = extract_show_name(content2)
    concept1 = extract_concept(content1)
    concept2 = extract_concept(content2)
    plot1 = extract_plot(content1)
    plot2 = extract_plot(content2)
    
    # Final message
    final_message = f"""
I have also created just for you two shows which I think you would love.
Show #1 is based on the fact that you loved the input shows that you gave me. Its name is {show1name} and it is a {concept1}
Show #2 is based on the shows that I recommended for you. Its name is {show2name} and it is a {concept2}
Here are also the 2 TV show ads. Hope you like them!
"""
    logging.info(Fore.CYAN + final_message)

    images = generate_show_ads(plot1, plot2, client)

    logging.info(f"""To open the images in the browser click here :          
Show #1 : {images[0]}

Show #2 : {images[1]}""")