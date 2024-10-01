import numpy as np
import random
import csv

NUM_CARDS = 52
NUM_HAND_CARDS = 11
NUM_ACTIONS = 11
NUM_METRICS = 12

suit_symbols_list = ['♣', '♦', '♥', '♠']
rank_symbols_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

rank_point_values = {
    'A': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'T': 10,
    'J': 10,
    'Q': 10,
    'K': 10
}


def get_card_rank_suit(card_number):
    if not (1 <= card_number <= 52):
        raise ValueError(f"Invalid card number {card_number}. Must be between 1 and 52.")

    suit_index = (card_number - 1) // 13
    rank_index = (card_number - 1) % 13

    rank = rank_symbols_list[rank_index]
    suit = suit_symbols_list[suit_index]

    return rank, suit


def get_card_str(card_number):
    if not (1 <= card_number <= 52):
        raise ValueError(f"Invalid card number {card_number}. Must be between 1 and 52.")

    suit_index = (card_number - 1) // 13
    rank_index = (card_number - 1) % 13

    rank = rank_symbols_list[rank_index]
    suit = suit_symbols_list[suit_index]

    return f"{rank}{suit}"


def get_card_number_from_string(card_str):
    rank = card_str[:-1]
    suit = card_str[-1]

    rank_index = rank_symbols_list.index(rank)
    suit_index = suit_symbols_list.index(suit)

    card_number = suit_index * 13 + rank_index + 1
    return card_number


def get_card_number_from_tuple(card_tuple):
    suit_index, rank_index = card_tuple

    card_number = suit_index * 13 + rank_index + 1
    return card_number


def find_least_useful_card(full_hand):
    useful_cards = set()
    semi_useful_cards = set()
    sets = []
    runs = []

    rank_counts = {}
    for card in full_hand:
        rank, _ = get_card_rank_suit(card)
        rank_counts[rank] = rank_counts.get(rank, 0) + 1

    for rank, count in rank_counts.items():
        if count >= 3:
            cards_in_set = [card for card in full_hand if get_card_rank_suit(card)[0] == rank]
            sets.append(cards_in_set)
            useful_cards.update(cards_in_set)
        elif count == 2:
            semi_useful_cards.update([card for card in full_hand if get_card_rank_suit(card)[0] == rank])

    suit_groups = {}
    rank_mapping = {rank: i + 1 for i, rank in enumerate(rank_symbols_list)}

    for card in full_hand:
        rank, suit = get_card_rank_suit(card)
        rank_num = rank_mapping[rank]
        suit_groups.setdefault(suit, []).append(rank_num)

    for suit, ranks in suit_groups.items():
        ranks.sort()
        current_run = []
        for i in range(len(ranks)):
            if i == 0 or ranks[i] == ranks[i - 1] + 1:
                current_run.append(ranks[i])
            else:
                if len(current_run) >= 3:
                    run_cards = [card for card in full_hand if get_card_rank_suit(card)[1] == suit and rank_mapping[
                        get_card_rank_suit(card)[0]] in current_run]
                    runs.append(run_cards)
                    useful_cards.update(run_cards)
                current_run = [ranks[i]]
        if len(current_run) >= 3:
            run_cards = [card for card in full_hand if get_card_rank_suit(card)[1] == suit and rank_mapping[
                get_card_rank_suit(card)[0]] in current_run]
            runs.append(run_cards)
            useful_cards.update(run_cards)

    for suit, ranks in suit_groups.items():
        ranks.sort()
        for i in range(len(ranks) - 1):
            if ranks[i + 1] == ranks[i] + 1:
                run_cards = [card for card in full_hand if
                             get_card_rank_suit(card)[1] == suit and rank_mapping[get_card_rank_suit(card)[0]] in [
                                 ranks[i], ranks[i + 1]]]
                if len(run_cards) == 2 and all(card not in useful_cards for card in run_cards):
                    semi_useful_cards.update(run_cards)

    semi_useful_cards -= useful_cards

    non_useful_cards = []
    for card in full_hand:
        if card not in useful_cards and card not in semi_useful_cards:
            non_useful_cards.append(card)

    if non_useful_cards:
        return max(non_useful_cards, key=lambda x: rank_mapping[get_card_rank_suit(x)[0]])
    elif semi_useful_cards:
        return max(semi_useful_cards, key=lambda x: rank_mapping[get_card_rank_suit(x)[0]])
    else:
        long_set = [_set for _set in sets if len(_set) > 3]
        long_run = [run for run in runs if len(run) > 3]
        if long_set:
            return max(long_set[0], key=lambda x: rank_mapping[get_card_rank_suit(x)[0]])
        elif long_run:
            return max(long_run[0], key=lambda x: rank_mapping[get_card_rank_suit(x)[0]])
        else:
            return max(useful_cards, key=lambda x: rank_mapping[get_card_rank_suit(x)[0]])


def generate_metrics(sorted_hand):
    """
    Generates a list of metrics based on the hand, top_discard, and action.

    Args:
        sorted_hand (list): List of card numbers in the hand.
        top_discard (int): The top discard card number.

    Returns:
        list: List of metric values.
    """
    ranks = [get_card_rank_suit(card)[0] for card in sorted_hand]
    rank_mapping_inv = {rank: i + 1 for i, rank in enumerate(rank_symbols_list)}

    try:
        rank_nums = [rank_mapping_inv[rank] for rank in ranks]
    except KeyError as e:
        print(f"KeyError: {e} in ranks {ranks}")
        raise

    average_rank = np.mean(rank_nums)
    normalized_average_rank = average_rank / 13

    median_rank = np.median(rank_nums)
    normalized_median_rank = median_rank / 13

    variance = np.var(rank_nums)
    normalized_variance = variance / 169

    rank_range = max(rank_nums) - min(rank_nums)
    normalized_range = rank_range / 12

    high_cards = sum(1 for rank in rank_nums if rank >= 10)
    proportion_high = high_cards / 10

    low_cards = sum(1 for rank in rank_nums if rank <= 5)
    proportion_low = low_cards / 10

    sets = 0
    rank_counts = {}
    for card in sorted_hand:
        rank, _ = get_card_rank_suit(card)
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    for count in rank_counts.values():
        if count >= 3:
            sets += 1
    normalized_sets = sets / 3

    sequences = 0
    suits_dict = {}
    for card in sorted_hand:
        rank, suit = get_card_rank_suit(card)
        rank_num = rank_mapping_inv[rank]
        if suit not in suits_dict:
            suits_dict[suit] = []
        suits_dict[suit].append(rank_num)
    for suit, ranks_in_suit in suits_dict.items():
        sorted_ranks = sorted(ranks_in_suit)
        consecutive = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i - 1] + 1:
                consecutive += 1
                if consecutive >= 3:
                    sequences += 1
            else:
                consecutive = 1
    normalized_sequences = sequences / 3

    total_set_cards = 0
    for rank, count in rank_counts.items():
        if count >= 3:
            total_set_cards += count
    proportion_in_sets = total_set_cards / 10

    total_sequence_cards = 0
    for suit, ranks_in_suit in suits_dict.items():
        sorted_ranks = sorted(ranks_in_suit)
        consecutive = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i - 1] + 1:
                consecutive += 1
                if consecutive >= 3:
                    total_sequence_cards += 2
            else:
                consecutive = 1
    proportion_in_sequences = total_sequence_cards / 10

    if sets > 0:
        average_set_size = total_set_cards / sets
    else:
        average_set_size = 0
    normalized_avg_set_size = average_set_size / 4

    if sequences > 0:
        average_sequence_length = total_sequence_cards / sequences
    else:
        average_sequence_length = 0
    normalized_avg_seq_length = average_sequence_length / 13

    metrics = [
        normalized_average_rank,
        normalized_median_rank,
        normalized_variance,
        normalized_range,
        proportion_high,
        proportion_low,
        normalized_sets,
        normalized_sequences,
        proportion_in_sets,
        proportion_in_sequences,
        normalized_avg_set_size,
        normalized_avg_seq_length,
    ]

    return metrics


def generate_dataset(num_samples):
    dataset = []

    for _ in range(num_samples):
        hand = random.sample(range(1, 53), 11)

        sorted_hand = sorted(hand)
        metrics = generate_metrics(sorted_hand.copy())

        discard_card = find_least_useful_card(sorted_hand)
        discard_index = sorted_hand.index(discard_card)

        dataset.append((sorted_hand, discard_card, discard_index, metrics))

    return dataset


def display_metrics(metrics):
    print("Metrics:")
    metrics_names = [
        'normalized_average_rank',
        'normalized_median_rank',
        'normalized_variance',
        'normalized_range',
        'proportion_low',
        'proportion_high',
        'normalized_sets',
        'normalized_sequences',
        'proportion_in_sets',
        'proportion_in_sequences',
        'normalized_avg_set_size',
        'normalized_avg_seq_length',
    ]
    for name, value in zip(metrics_names, metrics):
        print(f"  {name}: {value:.4f}")


if __name__ == '__main__':
    dataset = generate_dataset(20000)

    for i, sample in enumerate(dataset[:5], start=1):
        sorted_hand, discard_card, discard_index, metrics = sample

        print(f"Sample {i}:")
        print(f"Hand: {[get_card_str(card) for card in sorted_hand]}")
        print(f"Discarded Card: {get_card_str(discard_card)}")
        print(f"Discarded Card Index: {discard_index}")
        print(f"Metrics: {metrics}\n")

    # with open('rummy_dataset.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Sorted Hand', 'Discard Card', 'Discard Index', 'Metrics'])
    #     for sample in dataset:
    #         writer.writerow(sample)

    with open('rummy_test_dataset.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sorted Hand', 'Discard Card', 'Discard Index', 'Metrics'])
        for sample in dataset:
            writer.writerow(sample)
