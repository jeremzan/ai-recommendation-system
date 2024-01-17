import pytest
from show_suggester_ai import ask_user, get_favorite_tv_shows

def test_ask_user_valid_input():
    assert ask_user("Game of Thrones, The Witcher") == ["Game of Thrones", " The Witcher"]

def test_ask_user_invalid_input():
    assert ask_user("Game of Thrones") is None

def test_get_favorite_tv_shows():
    shows_list = ["Game of Thrones, The Witcher", "Breaking Bad, Stranger Things"]
    assert get_favorite_tv_shows(shows_list) == ["Game of Thrones", "The Witcher"]