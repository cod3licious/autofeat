import numpy as np


def random_seed_generator(num_seeds: int, low: int = 0, high: int = 2**32 - 1):
    """
    Generate a specified number of random integer seeds.

    Parameters:
    num_seeds: Number of random seeds to generate.
    low: Lower bound for random integers (default is 0).
    high: Upper bound for random integers (default is 2**32 - 1).

    Returns:
    list: List of random seeds.
    """
    # Generate and return a list of random seeds
    return [np.random.randint(low, high) for _ in range(num_seeds)]
