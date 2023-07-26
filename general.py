import string

CASING_LOOKUP = {
    'numeric':0,
    'all_upper':1,
    'all_lower':2,
    'initial_upper':3,
    'other':4,
    'PAD':5
}

chars = list(string.ascii_letters + string.punctuation + string.digits)
CHAR_LOOKUP = {char: idx for char, idx in zip(chars, range(1, len(chars) + 1))}