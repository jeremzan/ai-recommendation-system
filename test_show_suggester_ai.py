import pytest
import logging
import openai as OpenAI
import os
from show_suggester_ai import user_input_to_shows_list, get_favorite_tv_shows, generate_show_descriptions, find_matching_shows

def test_ask_user_valid_input():
    assert user_input_to_shows_list("Gem of thrunes,    witch,  ") == ["Gem of thrunes","witch"]
    assert user_input_to_shows_list("Game of") == ["Game of"]

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