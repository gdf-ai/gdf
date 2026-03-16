"""Tests for OnlineTrainer."""

from gdf.model import TinyTransformer
from gdf.trainer import OnlineTrainer, TrainerConfig


def test_train_step_good():
    model = TinyTransformer()
    trainer = OnlineTrainer(model)
    result = trainer.train_step("hello world", feedback="good")
    assert result["loss"] is not None
    assert result["loss"] > 0
    assert trainer.step_count >= 1
    assert len(trainer.replay_buffer) == 1


def test_train_step_bad():
    model = TinyTransformer()
    trainer = OnlineTrainer(model)
    result = trainer.train_step("bad text", feedback="bad")
    assert result["loss"] is None
    assert len(trainer.replay_buffer) == 0


def test_train_step_correction():
    model = TinyTransformer()
    trainer = OnlineTrainer(model)
    result = trainer.train_step(
        "helo wrld", feedback="correction", correction="hello world"
    )
    assert result["loss"] is not None
    assert result["loss"] > 0
    assert len(trainer.replay_buffer) == 1


def test_replay_buffer():
    model = TinyTransformer()
    config = TrainerConfig(replay_buffer_size=5, replay_samples=2)
    trainer = OnlineTrainer(model, config)
    for i in range(10):
        trainer.train_step(f"sample text number {i}", feedback="good")
    assert len(trainer.replay_buffer) == 5  # maxlen=5


def test_loss_decreases():
    model = TinyTransformer()
    trainer = OnlineTrainer(model, TrainerConfig(replay_samples=0))
    text = "the quick brown fox jumps over the lazy dog"
    losses = []
    for _ in range(20):
        result = trainer.train_step(text, feedback="good")
        losses.append(result["loss"])
    # Loss should generally decrease
    assert losses[-1] < losses[0]


def test_state_roundtrip():
    model = TinyTransformer()
    trainer = OnlineTrainer(model)
    trainer.train_step("test text", feedback="good")
    state = trainer.get_state()

    model2 = TinyTransformer()
    trainer2 = OnlineTrainer(model2)
    trainer2.load_state(state)

    assert trainer2.step_count == trainer.step_count
    assert len(trainer2.replay_buffer) == len(trainer.replay_buffer)
