import pickle
from unittest.mock import Mock, patch 
from PIL import Image
from io import BytesIO
import pytest
import pandas as pd
from show_suggester_ai import generate_show_ads, generate_embeddings, user_input_to_shows_list, get_favorite_tv_shows, read_csv_file, generate_show_descriptions, find_matching_shows


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

@patch('show_suggester_ai.create_openai_client')
def test_generate_show_descriptions(mock_create_client):
    # Mock data
    favorite_shows = ['Friends', 'Breaking Bad']
    recommended_shows = ['Game of Thrones', 'The Witcher']

    mock_client = Mock()
    mock_create_client.return_value = mock_client

    # Mock response
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='Show 1 Description')), 
                             Mock(message=Mock(content='Show 2 Description'))]

    mock_client.chat.completions.create.return_value = mock_response

    # Call the function with the mock client
    with patch('show_suggester_ai.create_openai_client', mock_client):  
        result = generate_show_descriptions(favorite_shows, recommended_shows, mock_client)

    # Check if the result is a tuple and contains two elements
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, "Result should contain two elements"
    assert isinstance(result[0], str), "First element of the result should be a string"
    assert isinstance(result[1], str), "Second element of the result should be a string"

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
    

@patch('show_suggester_ai.create_openai_client')
@patch('builtins.open')  # Mock the open function
@patch('pickle.dump')  # Mock the pickle.dump function
def test_generate_embeddings(mock_pickle_dump, mock_open, mock_create_client):
    # Create a mock OpenAI client
    mock_client = Mock()
    mock_create_client.return_value = mock_client

    # Prepare sample TV shows data
    sample_data = pd.DataFrame({
        'Title': ['Show 1', 'Show 2'],
        'Description': ['Description of Show 1', 'Description of Show 2']
    })

    # Mock embeddings response
    mock_response1 = Mock()
    mock_response1.data = [Mock(embedding=[0.1, 0.2, 0.3])]

    mock_response2 = Mock()
    mock_response2.data = [Mock(embedding=[0.4, 0.5, 0.6])]

    # Configure the mock client to return these mock responses
    mock_client.embeddings.create.side_effect = [mock_response1, mock_response2]

    # Call the function
    generate_embeddings(sample_data, mock_client)

    # Check that embeddings are correctly formed and pickle.dump is called
    expected_embeddings = {
        'Show 1': [0.1, 0.2, 0.3],
        'Show 2': [0.4, 0.5, 0.6]
    }
    mock_pickle_dump.assert_called_once_with(expected_embeddings, mock_open.return_value.__enter__.return_value)

    # Check that the file is opened in binary write mode
    mock_open.assert_called_once_with('embeddings.pkl', 'wb')


@patch('show_suggester_ai.create_openai_client')
def test_generate_show_ads(mock_create_client):
    # Create a dummy image and get its byte stream
    dummy_image = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = BytesIO()
    dummy_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Mock the OpenAI client
    mock_client = Mock()
    mock_create_client.return_value = mock_client

    # Mock response for DALL-E API calls
    mock_image_data1 = Mock()
    mock_image_data1.data = [Mock(url='https://mockurl1.com/image1')]
    mock_image_data2 = Mock()
    mock_image_data2.data = [Mock(url='https://mockurl2.com/image2')]

    # Set up side effect to return different data for each call
    mock_client.images.generate.side_effect = [mock_image_data1, mock_image_data2]

    # Mock the responses for the image downloads
    mock_response1 = Mock()
    mock_response1.status_code = 200
    mock_response1.content = img_byte_arr

    mock_response2 = Mock()
    mock_response2.status_code = 200
    mock_response2.content = img_byte_arr

    # Setup patches for requests.get
    with patch('requests.get', side_effect=[mock_response1, mock_response2]):
        # Example inputs
        plot1 = 'Plot description for Show 1'
        plot2 = 'Plot description for Show 2'

        # Call the function
        result = generate_show_ads(plot1, plot2, mock_client)

        # Print result for debugging
        print("Result URLs:", result)

        # Assertions
        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 2, "Result should contain two image URLs"
        assert result[0] == 'https://mockurl1.com/image1', "First image URL does not match expected output"
        assert result[1] == 'https://mockurl2.com/image2', "Second image URL does not match expected output"


   
