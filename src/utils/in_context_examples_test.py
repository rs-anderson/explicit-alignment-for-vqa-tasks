
from in_context_examples import InContextExampleFormatter
from easydict import EasyDict
import pytest


@pytest.fixture
def in_context_examples():
    return [
        EasyDict(
            {
                "question_id": 508840006,
                "img_key": 508840,
                "question": "What color is the boys hat?",
                "gold_answer": "red",
            }
        ),
        EasyDict(
            {
                "question_id": 135938002,
                "img_key": 135938,
                "question": "Is the man wearing a shirt?",
                "gold_answer": "no",
            }
        ),
    ]


@pytest.fixture
def test_sample():
    return EasyDict(
        {
            "question_id": 262148000,
            "question": "Where is he looking?",
            "img_key_full": "000000262148",
            "img": [],
            "gold_answer": "down",
            "answers": [
                "down",
                "down",
                "at table",
                "skateboard",
                "down",
                "table",
                "down",
                "down",
                "down",
            ],
            "clip_embedding": None,
            "in_context_examples": None,
        }
    )


@pytest.mark.parametrize("format_type, expected_formatted_input", [
    ("default", '<extra_id_0>\nWhat color is the boys hat?\nred\n<extra_id_1>\nIs the man wearing a shirt?\nno\n<extra_id_2>\nWhere is he looking?\n'),
    ("hotpotqa", '<extra_id_0>\nCombine facts and answer this:\nWhat color is the boys hat?\nred\n<extra_id_1>\nCombine facts and answer this:\nIs the man wearing a shirt?\nno\n<extra_id_2>\nCombine facts and answer this:\nWhere is he looking?\n'),
    ("hotpotqa_no_prefix", 'Combine facts and answer this:\nWhat color is the boys hat?\nred\nCombine facts and answer this:\nIs the man wearing a shirt?\nno\nCombine facts and answer this:\nWhere is he looking?\n'),
])
def test_example_formatter(in_context_examples, test_sample, format_type, expected_formatted_input):

    example_formatter = InContextExampleFormatter(format_type=format_type)
    formatted_input = example_formatter.format_input(
        in_context_examples, test_sample
    )
    assert formatted_input == expected_formatted_input


@pytest.mark.parametrize("format_type, expected_formatted_input", [
    ("default", '<extra_id_0>\nWhere is he looking?\n'),
    ("hotpotqa", '<extra_id_0>\nCombine facts and answer this:\nWhere is he looking?\n'),
    ("hotpotqa_no_prefix", 'Combine facts and answer this:\nWhere is he looking?\n'),
])
def test_example_formatter_when_no_in_context_examples(in_context_examples, test_sample, format_type, expected_formatted_input):
    in_context_examples = []
    example_formatter = InContextExampleFormatter(format_type=format_type)
    formatted_input = example_formatter.format_input(
        in_context_examples, test_sample
    )
    
    assert formatted_input == expected_formatted_input
