"""Tests for trained-point judgment-source declarations."""

import didactic.api as dx
import pytest

from lofi.spec.judgment_sources import decode_predicted_judgment_source


def test_predicted_source_names_producer_point_item_set_and_response_space() -> None:
    settings = decode_predicted_judgment_source(
        b"""{
          "kind": "predicted",
          "producer_run": "runs/synthetic",
          "producer_point": "point-17",
          "item_set": "entailment-seed/en",
          "inputs": {"premise": "context", "hypothesis": "claim"},
          "factors": {"verb": "item['verb']"},
          "question": "inference",
          "labels": ["non-entails", "entails"],
          "participant": "lofi-point-17"
        }"""
    ).settings()

    assert settings.producer_run == "runs/synthetic"
    assert settings.producer_point == "point-17"
    assert settings.item_set == "entailment-seed/en"
    assert settings.inputs == {"premise": "context", "hypothesis": "claim"}
    assert settings.factors == {"verb": "item['verb']"}
    assert settings.labels == ("non-entails", "entails")


def test_predicted_source_refuses_a_repeated_response_label() -> None:
    with pytest.raises(dx.ValidationError, match="distinct labels"):
        decode_predicted_judgment_source(
            b"""{
              "kind": "predicted",
              "producer_run": "runs/synthetic",
              "producer_point": "point-17",
              "item_set": "entailment-seed/en",
              "inputs": {"premise": "context", "hypothesis": "claim"},
              "factors": {},
              "question": "inference",
              "labels": ["entails", "entails"]
            }"""
        )
