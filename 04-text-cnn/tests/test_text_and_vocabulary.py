from disaster_tweets.text import normalize_tweet, tokenize
from disaster_tweets.vocabulary import build_vocabulary, encode_text


def test_normalize_tweet_replaces_urls_and_mentions():
    text = normalize_tweet("WOW @someone https://example.com/a?b=1  #StaySafe")
    assert "<user>" in text
    assert "<url>" in text
    assert "#staysafe" in text


def test_tokenizer_preserves_special_markers_and_hashtags():
    tokens = tokenize("Follow @news at https://example.com #storm!")
    assert "<user>" in tokens
    assert "<url>" in tokens
    assert "#storm" in tokens


def test_vocabulary_is_deterministic_and_encoding_is_padded():
    texts = ["fire in the hills", "fire near the hills"]
    vocab_a = build_vocabulary(texts, min_frequency=1)
    vocab_b = build_vocabulary(texts, min_frequency=1)
    assert vocab_a == vocab_b
    assert vocab_a["<pad>"] == 0
    assert vocab_a["<unk>"] == 1

    encoded = encode_text("fire unknownword", vocab_a, max_length=5)
    assert len(encoded) == 5
    assert encoded[-1] == 0
    assert encoded[1] == 1
