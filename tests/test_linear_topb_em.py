import numpy as np

from dp_mechanisms.linear_topb_em import linear_topb_exponential_mechanism


def test_linear_topb_release_has_fixed_cardinality_and_is_seeded() -> None:
    ranking = np.arange(8)
    first = linear_topb_exponential_mechanism(ranking, epsilon=1.0, B=4, rng=42)
    second = linear_topb_exponential_mechanism(ranking, epsilon=1.0, B=4, rng=42)

    assert np.array_equal(first, second)
    assert first.shape == (8,)
    assert set(first.tolist()) <= {0, 1}
    assert int(first.sum()) == 4
