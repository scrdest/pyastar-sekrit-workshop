import pytest
from examples import reasoning


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
    test_map = reasoning.custom_map()
    prec_checker = reasoning.preconds_checker_for(test_map)
    result = prec_checker(action, blackboard)
    assert result is expected

