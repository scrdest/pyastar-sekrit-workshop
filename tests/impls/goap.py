import pytest
from pyastar.reasoning import reasoning_v3


@pytest.mark.parametrize(
    ("action", "blackboard", "expected"), (
        (
            "Eat",
            {
                'HasDirtyDishes': 1,
            },
            False
        ),
        (
            "Eat",
            {
                'HasFood': 1,
            },
            False
        ),
        (
            "Eat",
            {
                'HasFood': 1,
                'HasCleanDishes': 1,
            },
            True
        ),
    )
)
def test_preconds(action, blackboard, expected):
    map = reasoning_v3.custom_map()
    prec_checker = reasoning_v3.preconds_checker_for(map)
    result = prec_checker(action, blackboard)
    assert result is expected

