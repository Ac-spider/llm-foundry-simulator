"""Tests for alignment training modules."""
import json
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn.functional as F


class TestTokenizePromptAndOutput:
    """Test tokenize_prompt_and_output function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(side_effect=lambda text, add_special_tokens=False: [
            ord(c) for c in text[:10]]  # Simple encoding
        )
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        return tokenizer

    def test_basic_tokenization(self, mock_tokenizer):
        """Test basic prompt-output tokenization."""
        from llm_foundry.stage5_align.utils import tokenize_prompt_and_output

        result = tokenize_prompt_and_output(
            prompt_strs=["What is 2+2?"],
            output_strs=["The answer is 4."],
            tokenizer=mock_tokenizer,
        )

        assert "input_ids" in result
        assert "labels" in result
        assert "response_mask" in result
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["labels"], torch.Tensor)
        assert isinstance(result["response_mask"], torch.Tensor)

    def test_response_mask_shape(self, mock_tokenizer):
        """Test response mask has correct shape."""
        from llm_foundry.stage5_align.utils import tokenize_prompt_and_output

        result = tokenize_prompt_and_output(
            prompt_strs=["Prompt"],
            output_strs=["Response"],
            tokenizer=mock_tokenizer,
        )

        # labels should be length of prompt + response
        assert len(result["labels"]) == len(result["input_ids"])
        # response_mask should match labels length
        assert len(result["response_mask"]) == len(result["labels"])


class MockLogits:
    """Mock logits object that mimics model output."""
    def __init__(self, logits):
        self.logits = logits


class TestGetResponseLogProbs:
    """Test get_response_log_probs function."""

    @pytest.fixture
    def mock_model(self):
        """Create a simple mock model."""
        def mock_forward(input_ids):
            batch_size, seq_len = input_ids.shape
            vocab_size = 100
            # Return MockLogits object with tensor inside
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return MockLogits(logits)

        model = Mock()
        model.side_effect = mock_forward
        return model

    def test_log_probs_shape(self, mock_model):
        """Test log probs have correct shape."""
        from llm_foundry.stage5_align.utils import get_response_log_probs

        input_ids = torch.randint(0, 100, (2, 5))
        labels = torch.randint(0, 100, (2, 5))

        result = get_response_log_probs(mock_model, input_ids, labels)

        assert isinstance(result, dict)
        assert "log_probs" in result
        assert isinstance(result["log_probs"], torch.Tensor)
        assert result["log_probs"].dim() == 2
        assert result["log_probs"].shape == (2, 5)  # batch size, seq_len


class TestSFTDataset:
    """Test SFTDataset class."""

    @pytest.fixture
    def temp_data_file(self):
        """Create a temporary data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"prompt": "What is 2+2?", "response": "4"}) + '\n')
            f.write(json.dumps({"prompt": "Capital of France?", "response": "Paris"}) + '\n')
            f.flush()
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[10, 20, 30])
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = "<pad>"
        return tokenizer

    def test_dataset_length(self, temp_data_file, mock_tokenizer):
        """Test dataset returns correct length."""
        from llm_foundry.stage5_align.sft import SFTDataset

        dataset = SFTDataset(temp_data_file, mock_tokenizer, max_length=128)
        assert len(dataset) == 2

    def test_dataset_getitem(self, temp_data_file, mock_tokenizer):
        """Test dataset returns correct items."""
        from llm_foundry.stage5_align.sft import SFTDataset

        dataset = SFTDataset(temp_data_file, mock_tokenizer, max_length=128)
        item = dataset[0]

        assert "input_ids" in item
        assert "labels" in item
        assert "response_mask" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
        assert isinstance(item["response_mask"], torch.Tensor)


class TestSFTTrainer:
    """Test SFTTrainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.return_value = torch.randn(2, 10, 100)
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        return tokenizer

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer."""
        optimizer = Mock()
        optimizer.zero_grad = Mock()
        optimizer.step = Mock()
        return optimizer

    def test_initialization(self, mock_model, mock_tokenizer, mock_optimizer):
        """Test trainer initialization."""
        from llm_foundry.stage5_align.sft import SFTTrainer

        trainer = SFTTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            optimizer=mock_optimizer,
            device="cpu"
        )

        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.optimizer == mock_optimizer

    def test_loss_computation(self, mock_model, mock_tokenizer, mock_optimizer):
        """Test SFT loss computation."""
        from llm_foundry.stage5_align.sft import SFTTrainer

        trainer = SFTTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            optimizer=mock_optimizer,
            device="cpu"
        )

        input_ids = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2, 10))
        response_mask = torch.ones((2, 10))

        loss = trainer._compute_loss(input_ids, labels, response_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar


class TestDPODataset:
    """Test DPODataset class."""

    @pytest.fixture
    def temp_data_file(self):
        """Create a temporary data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({
                "prompt": "What is 2+2?",
                "chosen": "The answer is 4.",
                "rejected": "I don't know."
            }) + '\n')
            f.flush()
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[10, 20, 30, 40, 50])
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        return tokenizer

    def test_dataset_length(self, temp_data_file, mock_tokenizer):
        """Test dataset returns correct length."""
        from llm_foundry.stage5_align.dpo import DPODataset

        dataset = DPODataset(temp_data_file, mock_tokenizer, max_length=128)
        assert len(dataset) == 1

    def test_dataset_getitem(self, temp_data_file, mock_tokenizer):
        """Test dataset returns correct items."""
        from llm_foundry.stage5_align.dpo import DPODataset

        dataset = DPODataset(temp_data_file, mock_tokenizer, max_length=128)
        item = dataset[0]

        assert "prompt" in item
        assert "chosen" in item
        assert "rejected" in item
        assert "prompt_input_ids" in item
        assert "chosen_input_ids" in item
        assert "rejected_input_ids" in item


class TestDPOTrainer:
    """Test DPOTrainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        def mock_forward(input_ids):
            batch_size, seq_len = input_ids.shape
            vocab_size = 100
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return MockLogits(logits)

        model = Mock()
        model.side_effect = mock_forward
        model.to = Mock(return_value=model)
        model.parameters = Mock(return_value=[])
        model.state_dict = Mock(return_value={})
        return model

    @pytest.fixture
    def mock_ref_model(self):
        """Create a mock reference model."""
        def mock_forward(input_ids):
            batch_size, seq_len = input_ids.shape
            vocab_size = 100
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return MockLogits(logits)

        model = Mock()
        model.side_effect = mock_forward
        model.to = Mock(return_value=model)
        model.parameters = Mock(return_value=[])
        model.eval = Mock()
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        return tokenizer

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer."""
        optimizer = Mock()
        optimizer.zero_grad = Mock()
        optimizer.step = Mock()
        optimizer.state_dict = Mock(return_value={})
        return optimizer

    def test_initialization(self, mock_model, mock_ref_model, mock_tokenizer, mock_optimizer):
        """Test DPO trainer initialization."""
        from llm_foundry.stage5_align.dpo import DPOTrainer

        trainer = DPOTrainer(
            model=mock_model,
            ref_model=mock_ref_model,
            tokenizer=mock_tokenizer,
            optimizer=mock_optimizer,
            device="cpu"
        )

        assert trainer.model == mock_model
        assert trainer.ref_model == mock_ref_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.beta == 0.1


class TestGRPOTrainer:
    """Test GRPOTrainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        def mock_forward(input_ids):
            batch_size, seq_len = input_ids.shape
            vocab_size = 100
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return logits

        model = Mock()
        model.side_effect = mock_forward
        model.to = Mock(return_value=model)
        model.parameters = Mock(return_value=[])
        return model

    @pytest.fixture
    def mock_inference_backend(self):
        """Create a mock inference backend."""
        backend = Mock()
        backend.generate = Mock(return_value="Generated response")
        return backend

    @pytest.fixture
    def mock_reward_fn(self):
        """Create a mock reward function."""
        def reward_fn(response, gt):
            return {"reward": 1.0 if gt in response else 0.0}
        return reward_fn

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer."""
        optimizer = Mock()
        optimizer.zero_grad = Mock()
        optimizer.step = Mock()
        return optimizer

    def test_initialization(self, mock_model, mock_inference_backend, mock_reward_fn, mock_optimizer):
        """Test GRPO trainer initialization."""
        from llm_foundry.stage5_align.grpo import GRPOTrainer

        trainer = GRPOTrainer(
            model=mock_model,
            optimizer=mock_optimizer,
            inference_backend=mock_inference_backend,
            reward_fn=mock_reward_fn,
            device="cpu"
        )

        assert trainer.model == mock_model
        assert trainer.optimizer == mock_optimizer
        assert trainer.group_size == 8
        assert trainer.beta == 0.1

    def test_compute_group_advantages(self, mock_model, mock_inference_backend, mock_reward_fn, mock_optimizer):
        """Test group advantage computation."""
        from llm_foundry.stage5_align.grpo import GRPOTrainer

        trainer = GRPOTrainer(
            model=mock_model,
            optimizer=mock_optimizer,
            inference_backend=mock_inference_backend,
            reward_fn=mock_reward_fn,
            device="cpu",
            group_size=4
        )

        # Create rewards for 2 groups of 4
        rewards = torch.tensor([1.0, 0.5, 0.0, 0.5,  # Group 1
                               0.8, 0.2, 0.4, 0.6])  # Group 2

        advantages = trainer.compute_group_advantages(rewards, group_size=4)

        assert isinstance(advantages, torch.Tensor)
        assert advantages.shape == (8,)
        # Check that advantages sum to 0 within each group
        assert abs(advantages[0:4].sum().item()) < 1e-5
        assert abs(advantages[4:8].sum().item()) < 1e-5


class TestUtils:
    """Test utility functions."""

    def test_compute_group_normalized_rewards(self):
        """Test group normalized rewards computation."""
        from llm_foundry.stage5_align.utils import compute_group_normalized_rewards

        def mock_reward_fn(resp, gt):
            return {"reward": 1.0}

        responses = ["resp1", "resp2", "resp3", "resp4"]
        gts = ["gt1", "gt2", "gt3", "gt4"]

        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn=mock_reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=gts,
            group_size=2,
            advantage_eps=1e-6,
            normalize_by_std=True
        )

        assert isinstance(advantages, torch.Tensor)
        assert advantages.shape == (4,)
        assert isinstance(raw_rewards, torch.Tensor)
        assert raw_rewards.shape == (4,)
        assert "reward_mean" in metadata

    def test_compute_grpo_clip_loss(self):
        """Test GRPO clip loss computation."""
        from llm_foundry.stage5_align.utils import compute_grpo_clip_loss

        advantages = torch.randn(2)
        policy_log_probs = torch.randn(2, 10)
        old_log_probs = torch.randn(2, 10)

        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=0.2
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 2  # (batch, seq_len)
        assert "clip_fraction" in metadata

    def test_compute_entropy(self):
        """Test entropy computation."""
        from llm_foundry.stage5_align.utils import compute_entropy

        logits = torch.randn(2, 10, 100)
        entropy = compute_entropy(logits)

        assert isinstance(entropy, torch.Tensor)
        assert entropy.shape == (2, 10)
