import pandas as pd

from disaster_tweets.data import stratified_split
from disaster_tweets.text import normalize_tweet


def test_stratified_split_keeps_duplicate_tweets_in_one_partition():
    rows = []
    for label, prefix in [(0, "movie review"), (1, "emergency response")]:
        for group_index in range(20):
            text = f"{prefix} unique example {group_index}"
            for duplicate_index in range(2):
                rows.append({"text": text, "target": label})

    frame = pd.DataFrame(rows)
    x_train, x_valid, y_train, y_valid = stratified_split(frame, random_state=11)

    train_groups = {normalize_tweet(text) for text in x_train}
    valid_groups = {normalize_tweet(text) for text in x_valid}
    assert train_groups.isdisjoint(valid_groups)
    assert set(y_train) == {0, 1}
    assert set(y_valid) == {0, 1}
    assert abs(sum(y_valid) / len(y_valid) - 0.5) <= 0.05
