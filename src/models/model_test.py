from vct0 import VCT0Prefix
import torch
import pytest


@pytest.fixture
def test_few_shot_prefix_projections():
    return torch.tensor(
        [
            [
                [[-100.0, -101.0, -102.0], [-103.0, -104.0, -105.0]],
                [[-106.0, -107.0, -108.0], [-109.0, -110.0, -111.0]],
                [[-130.0, -131.0, -132.0], [-133.0, -134.0, -135.0]],
            ],
            [
                [[-112.0, -113.0, -114.0], [-115.0, -116.0, -117.0]],
                [[-117.0, -118.0, -119.0], [-120.0, -121.0, -122.0]],
                [[-140.0, -141.0, -142.0], [-143.0, -144.0, -145.0]],
            ],
        ],
    )


@pytest.fixture
def test_zero_shot_prefix_projections():
    return torch.tensor(
        [
            [
                [[-100.0, -101.0, -102.0], [-103.0, -104.0, -105.0]],
            ],
            [
                [[-112.0, -113.0, -114.0], [-115.0, -116.0, -117.0]],
            ],
        ],
    )


@pytest.fixture
def test_text_embeddings():
    return torch.tensor(
        [
            [
                [100.0, 101.0, 102.0],
                [103.0, 104.0, 105.0],
                [106.0, 107.0, 108.0],
                [109.0, 110.0, 111.0],
                [130.0, 131.0, 132.0],
                [133.0, 134.0, 135.0],
                [99.0, 98.0, 97.0],
            ],
            [
                [112.0, 113.0, 114.0],
                [115.0, 116.0, 117.0],
                [117.0, 118.0, 119.0],
                [120.0, 121.0, 122.0],
                [140.0, 141.0, 142.0],
                [143.0, 144.0, 145.0],
                [96.0, 95.0, 94.0],
            ],
        ],
    )


@pytest.fixture
def test_model():
    model_args = {
        "prefix_length": 2,
        "prefix_size": 768,  # dimensions of clip embedding
        "mapping_type": "mlp",  # "perceiver" or "transformer" or "mlp"
        "model_version": "bigscience/T0_3B",
    }

    model = VCT0Prefix(**model_args)
    model.lm_embedding_size = 3
    model.prefix_length = 2
    return model


def test_few_shot_inserting_prefix(
    test_zero_shot_prefix_projections, test_text_embeddings, test_model
):

    batch_size = 2
    num_shots = 0

    question_masks = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    question_tokens = torch.tensor(
        [
            [32099, 20414, 11, 11, 11, 48, 0],
            [20414, 32099, 11, 48, 48, 48, 10],
        ],
        dtype=int,
    )

    joint_embeddings, joint_attention_masks = test_model.insert_prefix_into_input(
        batch_size,
        num_shots,
        question_tokens,
        test_text_embeddings,
        test_zero_shot_prefix_projections,
        question_masks,
    )

    expected_joint_embeddings = torch.tensor(
        [
            [
                *test_zero_shot_prefix_projections[0][0].tolist(),
                test_text_embeddings[0, 1],
                test_text_embeddings[0, 2],
                test_text_embeddings[0, 3],
                test_text_embeddings[0, 4],
                test_text_embeddings[0, 5],
                test_text_embeddings[0, 6],
            ],
            [
                test_text_embeddings[1, 0],
                *test_zero_shot_prefix_projections[1][0].tolist(),
                test_text_embeddings[1, 2],
                test_text_embeddings[1, 3],
                test_text_embeddings[1, 4],
                test_text_embeddings[1, 5],
                test_text_embeddings[1, 6],
            ],
        ],
    )

    expected_joint_attention_masks = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=int,
    )

    assert (joint_embeddings != expected_joint_embeddings).sum() == 0 and (
        joint_attention_masks != expected_joint_attention_masks
    ).sum() == 0



def test_zero_shot_inserting_prefix(
    test_few_shot_prefix_projections, test_text_embeddings, test_model
):

    batch_size = 2
    num_shots = 2

    question_masks = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    question_tokens = torch.tensor(
        [
            [32099, 20414, 32098, 11, 32097, 48, 0],
            [20414, 32099, 11, 32098, 48, 32097, 10],
        ],
        dtype=int,
    )

    joint_embeddings, joint_attention_masks = test_model.insert_prefix_into_input(
        batch_size,
        num_shots,
        question_tokens,
        test_text_embeddings,
        test_few_shot_prefix_projections,
        question_masks,
    )

    expected_joint_embeddings = torch.tensor(
        [
            [
                *test_few_shot_prefix_projections[0][0].tolist(),
                test_text_embeddings[0, 1],
                *test_few_shot_prefix_projections[0][1].tolist(),
                test_text_embeddings[0, 3],
                *test_few_shot_prefix_projections[0][2].tolist(),
                test_text_embeddings[0, 5],
                test_text_embeddings[0, 6],
            ],
            [
                test_text_embeddings[1, 0],
                *test_few_shot_prefix_projections[1][0].tolist(),
                test_text_embeddings[1, 2],
                *test_few_shot_prefix_projections[1][1].tolist(),
                test_text_embeddings[1, 4],
                *test_few_shot_prefix_projections[1][2].tolist(),
                test_text_embeddings[1, 6],
            ],
        ],
    )

    expected_joint_attention_masks = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=int,
    )

    assert (joint_embeddings != expected_joint_embeddings).sum() == 0 and (
        joint_attention_masks != expected_joint_attention_masks
    ).sum() == 0
