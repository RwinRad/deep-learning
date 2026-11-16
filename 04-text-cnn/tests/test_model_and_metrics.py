import torch

from disaster_tweets.metrics import classification_metrics
from disaster_tweets.model import TextCNN


def test_textcnn_returns_one_logit_per_example():
    model = TextCNN(vocab_size=50, embedding_dim=16, num_filters=8, dropout=0.1)
    batch = torch.randint(0, 50, (4, 20))
    output = model(batch)
    assert output.shape == (4,)


def test_metrics_use_probability_scores_for_auc():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0
    assert result["roc_auc"] == 1.0
