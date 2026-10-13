import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from voice_of_customer.pipeline import LABELS, ROOT, load_analyzer
from voice_of_customer.text import mask_pii, normalize


def test_dataset_split_integrity_and_label_balance():
    data = pd.read_csv(ROOT / 'data/feedback.csv')
    assert data.id.is_unique
    assert set(data.split) == {'train', 'validation', 'test'}
    assert set(data.label) == set(LABELS)
    assert not data.text.map(normalize).duplicated().any()
    assert data.groupby(['split', 'label']).size().min() >= 10
    assert set(data.lang) == {'ar', 'en'}


def test_training_vocabulary_does_not_include_test_only_text():
    model = load_analyzer()
    # This test-only idiom is absent from both train and validation.
    features = model.classifier.named_steps['features'].transformer_list[0][1]
    assert 'بهذلة' not in features.vocabulary_
    assert model.reducer.named_steps['svd'].n_components == 32


def test_local_masking_preserves_negation():
    masked = mask_pii('مو سيئ، البريد test@example.com والجوال 0501234567')
    assert 'test@example.com' not in masked and '0501234567' not in masked
    assert 'مو سيئ' in normalize(masked)
    assert normalize('أَفضل ـ تطبيق') == 'افضل تطبيق'


def test_end_to_end_sentiment_clustering_and_summary():
    texts = ['التوصيل سريع وكل الطلب كامل', 'التطبيق يعلق والدفع ما يتم', 'كيف أوصل لخدمة المحادثة']
    model = load_analyzer()
    output = model.analyze(texts)
    assert [p['sentiment'] for p in output['predictions']] == ['positive', 'negative', 'neutral']
    assert output['total'] == 3
    assert sum(c['size'] for c in output['clusters']) == 3
    assert sum(output['sentiment_counts'].values()) == 3
    assert all(c['representative'] in texts for c in output['clusters'])
    vectors = model.embed(texts)
    assert vectors.shape == (3, 32) and np.isfinite(vectors).all()
    assert output == model.analyze(texts)


def test_private_contact_never_appears_in_output():
    result = load_analyzer().analyze(['المنتج ممتاز تواصلوا معي test@example.com 0501234567'])
    serialized = json.dumps(result)
    assert 'test@example.com' not in serialized and '0501234567' not in serialized


@pytest.mark.parametrize('comments', [[], [''], [' '], [None], ['x' * 4001], ['x'] * 1001, 'ممتاز', None])
def test_invalid_input_is_rejected(comments):
    with pytest.raises(ValueError):
        load_analyzer().analyze(comments)


def test_duplicate_batch_is_supported():
    result = load_analyzer().analyze(['المنتج ممتاز'] * 4)
    assert result['total'] == 4 and len(result['clusters']) == 1


def test_saved_metrics_match_fixed_test_predictions():
    predictions = pd.read_csv(ROOT / 'reports/predictions.csv')
    metrics = json.loads((ROOT / 'reports/metrics.json').read_text())
    for model in ['baseline', 'improved']:
        actual = f1_score(predictions.label, predictions[f'{model}_pred'], labels=LABELS, average='macro')
        assert actual == pytest.approx(metrics[model]['macro_f1'])
        assert metrics[model]['n'] == len(predictions) == 66
    errors = pd.read_csv(ROOT / 'reports/error_analysis.csv')
    assert len(errors) >= 10
    assert (errors.label != errors.improved_pred).all()
    assert set(errors.id) == set(predictions.loc[predictions.label != predictions.improved_pred, 'id'])
