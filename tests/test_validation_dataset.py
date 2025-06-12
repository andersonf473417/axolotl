"""Module for testing the validation module for the dataset config"""

import warnings
from typing import Optional

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.datasets import ChatTemplate

warnings.filterwarnings("error")


@pytest.fixture(name="minimal_cfg")
def fixture_cfg():
    return DictDefault(
        {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
            "learning_rate": 0.000001,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
        }
    )


# pylint: disable=too-many-public-methods (duplicate-code)
class BaseValidation:
    """
    Base validation module to setup the log capture
    """

    _caplog: Optional[pytest.LogCaptureFixture] = None

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog


class TestValidationCheckDatasetConfig(BaseValidation):
    """
    Test the validation for the dataset config to ensure no correct parameters are dropped
    """

    def test_dataset_config_no_drop_param(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                        "shards": 10,
                    }
                ]
            }
        )

        self._validate_and_check_config(
            cfg,
            expected_attrs={
                "path": cfg.datasets[0].path,
                "type": cfg.datasets[0].type,
                "shards": cfg.datasets[0].shards,
            },
        )

    def test_dataset_default_chat_template_no_drop_param(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "LDJnr/Puffin",
                        "type": "chat_template",
                        "field_messages": "conversations",
                        "shards": 10,
                        "message_field_role": "from",
                        "message_field_content": "value",
                    }
                ],
            }
        )

        self._validate_and_check_config(
            cfg,
            expected_attrs={
                "path": cfg.datasets[0].path,
                "type": cfg.datasets[0].type,
                "chat_template": ChatTemplate.tokenizer_default,
                "field_messages": cfg.datasets[0].field_messages,
                "shards": cfg.datasets[0].shards,
                "message_field_role": cfg.datasets[0].message_field_role,
                "message_field_content": cfg.datasets[0].message_field_content,
            },
            global_chat_template=None,
        )

    def test_dataset_partial_default_chat_template_no_drop_param(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "LDJnr/Puffin",
                        "type": "chat_template",
                        "field_messages": "conversations",
                        "shards": 10,
                        "message_field_role": "from",
                        "message_field_content": "value",
                    }
                ],
            }
        )

        self._validate_and_check_config(
            cfg,
            expected_attrs={
                "path": cfg.datasets[0].path,
                "type": cfg.datasets[0].type,
                "chat_template": ChatTemplate.tokenizer_default,
                "field_messages": cfg.datasets[0].field_messages,
                "shards": cfg.datasets[0].shards,
                "message_field_role": cfg.datasets[0].message_field_role,
                "message_field_content": cfg.datasets[0].message_field_content,
            },
            global_chat_template=ChatTemplate.chatml,
        )

    def test_dataset_chatml_chat_template_no_drop_param(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "LDJnr/Puffin",
                        "type": "chat_template",
                        "chat_template": "gemma",
                        "field_messages": "conversations",
                        "shards": 10,
                        "message_field_role": "from",
                        "message_field_content": "value",
                    }
                ],
            }
        )

        self._validate_and_check_config(
            cfg,
            expected_attrs={
                "path": cfg.datasets[0].path,
                "type": cfg.datasets[0].type,
                "chat_template": cfg.datasets[0].chat_template,
                "field_messages": cfg.datasets[0].field_messages,
                "shards": cfg.datasets[0].shards,
                "message_field_role": cfg.datasets[0].message_field_role,
                "message_field_content": cfg.datasets[0].message_field_content,
            },
            global_chat_template=cfg.chat_template,
        )

    def test_dataset_sharegpt_deprecation(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "LDJnr/Puffin",
                        "type": "sharegpt",
                        "conversation": "chatml",
                    }
                ],
            }
        )

        # Check sharegpt deprecation is raised
        with pytest.raises(ValueError, match=r".*type: sharegpt.*` is deprecated.*"):
            validate_config(cfg)

        # Check that deprecation is not thrown for non-str type
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": {
                            "field_instruction": "instruction",
                            "field_output": "output",
                            "field_system": "system",
                            "format": "<|user|> {instruction} {input} <|model|>",
                            "no_input_format": "<|user|> {instruction} <|model|>",
                            "system_prompt": "",
                        },
                    }
                ],
            }
        )

        validate_config(cfg)

        # Check that deprecation is not thrown for non-sharegpt type
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    }
                ],
            }
        )

        validate_config(cfg)

    def test_message_property_mappings(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                        "message_property_mappings": {
                            "role": "role",
                            "content": "content",
                        },
                    }
                ],
            }
        )

        validate_config(cfg)


class TestOptimizerValidation(BaseValidation):
    """
    Test muon optimizer validation
    """

    def test_muon_deepspeed(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    }
                ],
                "optimizer": "muon",
                "deepspeed": "deepspeed_configs/zero3.json",
            }
        )

        with pytest.raises(ValueError, match=r".*is currently incompatible with*"):
            validate_config(cfg)

    def test_muon_fsdp(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    }
                ],
                "optimizer": "muon",
                "fsdp": ["full_shard"],
                "fsdp_config": {
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                },
            }
        )

        with pytest.raises(ValueError, match=r".*is currently incompatible with*"):
            validate_config(cfg)
