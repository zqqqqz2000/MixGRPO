import random


def compute_reward(images, input_prompts, reward_function, reward_weights):
    assert len(images) == len(
        input_prompts
    ), f"length of `images` ({len(images)}) must be equal to length of `input_prompts` ({len(input_prompts)})"

    # Initialize results
    rewards_dict = {}
    successes_dict = {}

    rewrads = reward_function(images, input_prompts["prompts"], input_prompts)

    return rewards, [1], rewards_dict, successes_dict


def balance_pos_neg(samples, use_random=False):
    """Balance positive and negative samples distribution in the samples list."""
    if use_random:
        return random.sample(samples, len(samples))
    else:
        positive_samples = [sample for sample in samples if sample["advantages"].item() > 0]
        negative_samples = [sample for sample in samples if sample["advantages"].item() < 0]

        positive_samples = random.sample(positive_samples, len(positive_samples))
        negative_samples = random.sample(negative_samples, len(negative_samples))

        num_positive = len(positive_samples)
        num_negative = len(negative_samples)

        balanced_samples = []

        if num_positive < num_negative:
            smaller_group = positive_samples
            larger_group = negative_samples
        else:
            smaller_group = negative_samples
            larger_group = positive_samples

        for i in range(len(smaller_group)):
            balanced_samples.append(smaller_group[i])
            balanced_samples.append(larger_group[i])

        # If there are remaining samples in the larger group, add them
        remaining_samples = larger_group[len(smaller_group) :]
        balanced_samples.extend(remaining_samples)
        return balanced_samples
