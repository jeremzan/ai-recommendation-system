import pytest
from show_suggester_ai import user_input_to_shows_list, get_favorite_tv_shows

def test_ask_user_valid_input():
    assert user_input_to_shows_list("Gem of thrunes,    witch,  ") == ["Gem of thrunes","witch",""]
    assert user_input_to_shows_list("Game of") == ["Game of"]

def test_ask_user_empty_input():
    assert user_input_to_shows_list("") == [""]

def test_ask_user_whitespace_input():
    assert user_input_to_shows_list("   ") == [""]

def test_get_favorite_tv_shows():
    known_shows = ["Game of Thrones", "The Witcher", "Breaking Bad", "Stranger Things"]

    user_input = "Ge of Thrs, Witcer    ,  witcer, Breaking bud"
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