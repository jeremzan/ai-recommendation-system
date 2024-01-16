from show_suggester_ai import ask_user
import pytest

def test_len_input():
    shows_list = ask_user()   
    assert len(shows_list) >= 2

def test_valid_input():
    shows_list = ask_user()
    for show in shows_list:
        assert not show.strip() == ""

