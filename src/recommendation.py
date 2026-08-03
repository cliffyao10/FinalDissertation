"""
Outfit recommendation module.

Phase 1 uses a simple rule-based recommendation method.
"""


def recommend_outfit(category, colour):
    """
    Generate outfit recommendations based on category and colour.
    """

    rules = {
        ("Jeans", "Blue"): {
            "items": [
                "White T-shirt",
                "Grey jacket",
                "White trainers"
            ],
            "explanation": (
                "Blue jeans are versatile and can be matched with neutral items "
                "to create a casual everyday outfit."
            )
        },
        ("Jeans", "Black"): {
            "items": [
                "White shirt",
                "Black blazer",
                "Black loafers"
            ],
            "explanation": (
                "Black jeans can be styled with formal neutral items "
                "to create a smart casual outfit."
            )
        },
        ("T-shirt", "White"): {
            "items": [
                "Blue jeans",
                "Denim jacket",
                "White trainers"
            ],
            "explanation": (
                "A white T-shirt works well with denim and simple trainers "
                "because it creates a clean casual look."
            )
        },
        ("Dress", "Black"): {
            "items": [
                "Silver accessories",
                "Black heels",
                "Small handbag"
            ],
            "explanation": (
                "A black dress can be paired with simple accessories "
                "for a balanced evening outfit."
            )
        }
    }

    default_recommendation = {
        "items": [
            "Neutral top",
            "Simple jacket",
            "Classic trainers"
        ],
        "explanation": (
            "The system uses a default recommendation because this category "
            "and colour combination is not included in the current rule set."
        )
    }

    return rules.get((category, colour), default_recommendation)