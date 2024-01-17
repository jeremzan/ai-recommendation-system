# show_suggester_ai.py

from fuzzywuzzy import process

def ask_user(user_input):
    shows_list = user_input.split(",")
    # Check if the user entered more than 1 show or if there is an empty string/whitespaces only
    if len(shows_list) < 2 or not all(show.strip() for show in shows_list):
        return None
    return shows_list

def get_favorite_tv_shows(shows_list):
    known_shows = ["Game of Thrones", "Lupin", "The Witcher", "Breaking Bad", "Stranger Things"]

    if shows_list:
        matched_shows = [process.extractOne(show, known_shows)[0] for show in shows_list]
        return matched_shows
    return None

# Example usage in main script
if __name__ == "__main__":
    # user_inputs = []
    while True:
        user_input = input("Which TV shows did you love watching? Separate them by a comma. Make sure to enter more than 1 show: \n")
        # user_inputs.append(user_input)
        shows_list = ask_user(user_input)
        favorite_shows = get_favorite_tv_shows(shows_list)
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
