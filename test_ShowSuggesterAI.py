import pytest
from ShowSuggesterAI import ask_user

def test_len_input():
    user_input = ask_user()
    shows_list = user_input.split(",")
    assert len(shows_list) == 3

def test_valid_input():
    user_input = ask_user()
    shows_list = user_input.split(",")
    for show in shows_list:
        assert not show.strip() == ""


