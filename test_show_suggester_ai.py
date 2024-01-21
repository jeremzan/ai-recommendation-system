import pickle
from unittest.mock import ANY, Mock, mock_open, patch 
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import pytest
import logging
import openai as OpenAI
import os
import pandas as pd
from show_suggester_ai import generate_show_ads, generate_embeddings, user_input_to_shows_list, get_favorite_tv_shows, read_csv_file, generate_show_descriptions, find_matching_shows

load_dotenv()

def test_ask_user_valid_input():
    assert user_input_to_shows_list("Gem of thrunes,    witch,  ") == ["Gem of thrunes","witch"]
    assert user_input_to_shows_list("Game of ,   ") == ["Game of"]

def test_ask_user_empty_input():
    assert user_input_to_shows_list("") == []

def test_ask_user_whitespace_input():
    assert user_input_to_shows_list("   ") == []

def test_get_favorite_tv_shows():
    known_shows = ["Game of Thrones", "The Witcher", "Breaking Bad", "Stranger Things"]

    user_input = "Ge of Thrs, Witcer    ,  witcer, Breaking bud,"
    shows_list = user_input_to_shows_list(user_input)
    assert set(get_favorite_tv_shows(shows_list, known_shows)) == set(["Game of Thrones", "The Witcher", "Breaking Bad"])

def test_get_favorite_tv_shows_empty_list():
    known_shows = ["Game of Thrones", "The Witcher", "Breaking Bad", "Stranger Things"]
    shows_list = []
    assert get_favorite_tv_shows(shows_list, known_shows) == None

def test_get_favorite_tv_shows_single_show():
    known_shows = ["Game of Thrones", "The Witcher", "Breaking Bad", "Stranger Things"]
    shows_list = ["Game of Thrones"]
    assert get_favorite_tv_shows(shows_list, known_shows) == None

def test_get_favorite_tv_shows_duplicate_shows():
    known_shows = ["Game of Thrones", "The Witcher", "Breaking Bad", "Stranger Things"]
    shows_list = ["Game of Thrones", "Game of Thrones"]
    assert get_favorite_tv_shows(shows_list, known_shows) == None

def test_get_favorite_tv_shows_invalid_shows():
    known_shows = ["Game of Thrones", "The Witcher", "Breaking Bad", "Stranger Things"]
    shows_list = ["Invalid Show"]
    assert get_favorite_tv_shows(shows_list, known_shows) == None

def test_get_favorite_tv_shows_valid_shows():
    known_shows = ["Game of Thrones", "The Witcher", "Breaking Bad", "Stranger Things"]
    shows_list = ["Game of Thrones", "The Witcher", "Breaking Bad"]
    assert set(get_favorite_tv_shows(shows_list, known_shows)) == set(["Game of Thrones", "The Witcher", "Breaking Bad"])

def test_read_csv_file_valid_file():
    file_path = "./imdb_tvshows - imdb_tvshows.csv"
    tv_shows = read_csv_file(file_path)
    assert isinstance(tv_shows, pd.DataFrame)

def test_read_csv_file_invalid_file():
    file_path = "path/to/invalid/file.csv"
    tv_shows = read_csv_file(file_path)
    assert tv_shows is None


def test_generate_show_descriptions():
    try :
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")

    favorite_shows = ["Game of Thrones", "The Witcher"]
    recommended_shows = ["Breaking Bad", "Stranger Things"]
    model = "gpt-3.5-turbo"
    try :
        description1, description2 = generate_show_descriptions(favorite_shows, recommended_shows, client, model)
        print(description1)
        print(description2)
    except Exception as e:
        print(f"Error generating show descriptions: {e}")

def test_len_find_matching_shows():
    favorite_shows = ['Show 1', 'Show 2', 'Show 3']
    embeddings = {
        'Show 1': [1,6,-1,3],
        'Show 2': [0,2,1,2],
        'Show 3': [0,1,3,-5],
        'Show 4': [0,0,1,2],
        'Show 5': [3,-1,-2,1],
        'Show 6': [1,0,7,-3],
        'Show 7': [4,2,3,4],
        'Show 8': [2,-2,5,6],
        'Show 9': [1,3,3-1,-2],
        'Show 10': [3,1,2,-3],
    }
    recommanded_show = find_matching_shows(favorite_shows, embeddings)
    assert len(recommanded_show.items()) == 5

def test_type_find_matching_shows():
    favorite_shows = ['Show 1', 'Show 2', 'Show 3']
    embeddings = {
        'Show 1': [1,6,-1,3],
        'Show 2': [0,2,1,2],
        'Show 3': [0,1,3,-5],
        'Show 4': [0,0,1,2],
        'Show 5': [3,-1,-2,1],
        'Show 6': [1,0,7,-3],
        'Show 7': [4,2,3,4],
        'Show 8': [2,-2,5,6],
        'Show 9': [1,3,3-1,-2],
        'Show 10': [3,1,2,-3],
    }
    recommanded_show = find_matching_shows(favorite_shows, embeddings)
    assert type(recommanded_show) == dict

def test_valid_matching_shows():
    favorite_shows = ['Show 1', 'Show 2', 'Show 3']
    embeddings = {
        'Show 1': [1,6,-1,3],
        'Show 2': [0,2,1,2],
        'Show 3': [0,1,3,-5],
        'Show 4': [0,0,1,2],
        'Show 5': [3,-1,-2,1],
        'Show 6': [1,0,7,-3],
        'Show 7': [4,2,3,4],
        'Show 8': [2,-2,5,6],
        'Show 9': [1,3,3-1,-2],
        'Show 10': [3,1,2,-3],
    }
    recommanded_show = find_matching_shows(favorite_shows, embeddings)
    assert any(show in recommanded_show.keys() for show in embeddings.keys())


    
def test_find_matching_shows():
    favorite_shows = ['Show 1', 'Show 2', 'Show 3']
    embeddings = {
        'Show 1': [1,6,-1,3],
        'Show 2': [0,2,1,2],
        'Show 3': [0,1,3,-5],
        'Show 4': [0,0,1,2],
        'Show 5': [3,-1,-2,1],
        'Show 6': [1,0,7,-3],
        'Show 7': [4,2,3,4],
        'Show 8': [2,-2,5,6],
        'Show 9': [1,3,3-1,-2],
        'Show 10': [3,1,2,-3],
    }
    recommanded_show = find_matching_shows(favorite_shows, embeddings)
    print(recommanded_show.keys())
    assert set(recommanded_show.keys()) == set(['Show 9', 'Show 10', 'Show 7', 'Show 4', 'Show 6'])
    

def test_generate_embeddings():
    # Create a mock DataFrame with sample TV show data
    tv_shows = pd.DataFrame({
        'Title': ['Show 1', 'Show 2', 'Show 3'],
        'Description': ['Description 1', 'Description 2', 'Description 3']
    })

    # Create a mock OpenAI client
    class MockOpenAIClient:
        def __init__(self):
            self.embeddings = MockEmbeddings()

    # Create a mock OpenAI embeddings API
    class MockEmbeddings:
        def create(self, input, model):
            return MockResponse()

    # Create a mock OpenAI embeddings API response
    class MockResponse:
        def __init__(self):
            self.data = [MockEmbedding()]

    # Create a mock embedding
    class MockEmbedding:
        def __init__(self):
            self.embedding = [0.1, 0.2, 0.3]

    # Mock the 'open' function to avoid writing to a file
    with patch('builtins.open', mock_open()) as mock_file:
        # Mock the OpenAI client
        with patch('openai.OpenAI', MockOpenAIClient):
            # Call the generate_embeddings function
            generate_embeddings(tv_shows, MockOpenAIClient())

            # Read the written data from the mock file
            written_data = mock_file().write.call_args[0][0]
            embeddings_dict_from_file = pickle.loads(written_data)

    # Assert that the embeddings dictionary is correctly generated
    expected_embeddings_dict = {
        'Show 1': [0.1, 0.2, 0.3],
        'Show 2': [0.1, 0.2, 0.3],
        'Show 3': [0.1, 0.2, 0.3]
    }
    assert embeddings_dict_from_file == expected_embeddings_dict

    # Assert that the file is opened and written to
    mock_file.assert_any_call('embeddings.pkl', 'wb')
    mock_file().write.assert_called_once()


def test_generate_show_ads():
    client = None
    images = None

    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")

    if client is not None:
        try:
            images = generate_show_ads("A bird on a tree", "A shark in the sea", client)
            # Assert the URLs are as expected
            assert len(images) == 2  # Adjust this to match the expected number of images
        except Exception as e:
            print(f"Error generating show ads: {e}")
            assert False  # Fail the test if an exception occurs

    else:
        assert False  # Fail the test if client is not created

   
