"""Regression checks for uploaded feedback being lost or misread."""

import gradio as gr
import pytest

from app import analyze_csv


@pytest.mark.parametrize('content', [
    b'',
    b'other\nreview\n',
    b'text\n\xff\n',
    b'text\n"unterminated\n',
    'text\nالمنتج ممتاز\n" "\n'.encode(),
    'text\nالمنتج ممتاز\n\nالتطبيق يعلق\n'.encode(),
])
def test_invalid_csv_is_rejected_without_losing_rows(tmp_path, content):
    path = tmp_path / 'invalid.csv'
    path.write_bytes(content)
    with pytest.raises(gr.Error):
        analyze_csv(str(path))


def test_literal_na_is_preserved_and_multiline_review_is_one_comment(tmp_path):
    path = tmp_path / 'valid.csv'
    path.write_text('text\nNA\n"المنتج ممتاز\nوالتوصيل سريع"\n', encoding='utf-8')
    _, _, output = analyze_csv(str(path))
    assert output['total'] == 2
    assert output['predictions'][0]['text'] == 'NA'
    assert output['predictions'][1]['text'] == 'المنتج ممتاز والتوصيل سريع'
