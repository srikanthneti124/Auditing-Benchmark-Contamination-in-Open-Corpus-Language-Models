import pytest

from evaluate import permute_choices

CHOICES = ["Paris", "London", "Berlin", "Madrid"]


@pytest.mark.parametrize("label", range(4))
@pytest.mark.parametrize("item_id", range(50))
def test_new_label_points_at_same_answer(label, item_id):
    choices, new_label = permute_choices(CHOICES, label, seed=0, item_id=item_id)
    assert choices[new_label] == CHOICES[label]
    assert sorted(choices) == sorted(CHOICES)


@pytest.mark.parametrize("item_id", range(200))
def test_never_identity(item_id):
    choices, _ = permute_choices(CHOICES, 0, seed=3, item_id=item_id)
    assert choices != CHOICES


def test_deterministic_and_seed_dependent():
    a = [permute_choices(CHOICES, 1, 0, i) for i in range(30)]
    assert a == [permute_choices(CHOICES, 1, 0, i) for i in range(30)]
    assert a != [permute_choices(CHOICES, 1, 1, i) for i in range(30)]


@pytest.mark.parametrize("bad", ["All of the above", "none of these", "Both A and B", "Neither"])
def test_unpermutable_items_skipped(bad):
    assert permute_choices(["x", "y", "z", bad], 0, 0, 0) is None
