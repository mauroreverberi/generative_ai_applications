"""Character tokenizer for the company names, adapted from the CharTokenizer of the course."""

PAD = "<pad>"
# a line break never appears in a name, so it marks the start and the end of every name
END = "\n"


class CharTokenizer:
    """Map characters to ids and back, the padding token gets id 0 and the end marker id 1."""

    def __init__(self, vocabulary):
        """Build the two lookup tables from the list of tokens."""
        self.token_id_for_char = {char: token_id for token_id, char in enumerate(vocabulary)}
        self.char_for_token_id = {token_id: char for token_id, char in enumerate(vocabulary)}

    @staticmethod
    def train_from_names(names):
        """Create a tokenizer with every character that appears in the names."""
        return CharTokenizer([PAD, END] + sorted(set("".join(names))))

    def encode(self, text):
        """Return the ids of the characters of a text as a list."""
        return [self.token_id_for_char[char] for char in text]

    def decode(self, token_ids):
        """Return the text for a list of ids, without padding and end markers."""
        return "".join(self.char_for_token_id[token_id] for token_id in token_ids
                       if token_id > 1)

    def vocabulary_size(self):
        """Return the number of tokens, including the two special tokens."""
        return len(self.token_id_for_char)
