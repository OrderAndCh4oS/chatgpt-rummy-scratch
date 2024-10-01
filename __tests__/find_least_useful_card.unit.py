import unittest

from generate_data import find_least_useful_card, get_card_rank_suit, get_card_number_from_string


class TestFindLeastUsefulCard(unittest.TestCase):
    def test_discard_least_useful_card_when_no_sets_or_runs_present(self):
        """
        Test discarding the least useful card when no sets or runs are present.
        """
        hand = [1, 15, 28]
        new_card = 43
        expected = 43
        result = find_least_useful_card(hand, new_card)

        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_full_set_is_present_with_extra_card(self):
        """
        Test discarding the least useful card when a full set is present with an extra card.
        """
        hand = [1, 2, 3]
        new_card = 4
        expected = 4
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_run_present_and_non_useful_card_discarded(self):
        """
        Test discarding the least useful card when a run is present and some cards are non-useful.
        """
        hand = [1, 2, 3, 15]
        new_card = 4
        expected = 15
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_pairs_are_present_and_non_useful_card(self):
        """
        Test discarding the least useful card when pairs are present and a non-useful card is also in hand.
        """
        hand = [1, 2, 15]
        new_card = 29
        expected = 29
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_two_card_sequences_exist_with_multiple_discard_options(self):
        """
        Test discarding the least useful card when two-card sequences are present, with multiple discard options.
        """
        hand = [1, 2, 15, 16]
        new_card = 3
        expected = [16, 3]
        result = find_least_useful_card(hand, new_card)
        print(f"Result: {result}")
        self.assertIn(result, expected)

    def test_discard_least_useful_card_with_multiple_overlapping_runs(self):
        """
        Test discarding the least useful card when multiple overlapping runs are present.
        """
        hand = [1, 2, 3, 4, 18, 19]
        new_card = 20
        expected = 4
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_all_cards_are_part_of_sets_or_runs(self):
        """
        Test discarding the least useful card when all cards are part of sets or runs.
        """
        hand = [1, 2, 3, 4]
        new_card = 5
        expected = 5
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_with_two_card_sequences_and_pairs_present(self):
        """
        Test discarding the least useful card when multiple two-card sequences and pairs are present.
        """
        hand = [1, 2, 3, 14, 15]
        new_card = 27
        expected = 15
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_multiple_non_useful_cards_are_present(self):
        """
        Test discarding the least useful card when multiple non-useful cards are present.
        """
        hand = [1, 15, 28, 43]
        new_card = 5
        expected = 5
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_hand_has_full_set_and_non_useful_card(self):
        """
        Test discarding the least useful card when a full set is present and some cards are non-useful.
        """
        hand = [1, 14, 27, 40]
        new_card = 50
        expected = 50
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_melds_overlap_with_runs_and_sets(self):
        """
        Test discarding the least useful card when there is an overlap between a set and a run.
        """
        hand = [1, 2, 3, 14, 15, 16]
        new_card = 4
        expected = 4
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_hand_has_two_runs(self):
        """
        Test discarding the least useful card when the hand has two separate runs.
        """
        hand = [1, 2, 3, 28, 29, 30]
        new_card = 15
        expected = 30
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_hand_has_only_one_non_useful_card(self):
        """
        Test discarding the least useful card when the hand has only one non-useful card.
        """
        hand = [1, 2, 3, 4, 28, 29]
        new_card = 30
        expected = 4
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_hand_has_multiple_full_sets(self):
        """
        Test discarding the least useful card when there are multiple full sets.
        """
        hand = [1, 14, 27, 40, 2, 15, 28]
        new_card = 3
        expected = [1, 14, 27, 40]
        result = find_least_useful_card(hand, new_card)
        self.assertIn(result, expected)

    def test_discard_least_useful_card_when_new_card_completes_a_pair(self):
        """
        Test discarding the least useful card when the new card completes a pair.
        """
        hand = [1, 14, 2, 15]
        new_card = 28
        expected = 1
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_new_card_is_part_of_a_run(self):
        """
        Test discarding the least useful card when the new card is part of a run.
        """
        hand = [1, 2, 3]
        new_card = 4
        expected = 4
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)

    def test_discard_least_useful_card_when_several_disjoint_cards_present(self):
        """
        Test discarding the least useful card when the hand has several disjoint cards and the discard should be 7♠.
        """
        card_strings = ["2♦", "3♠", "5♠", "7♠", "8♣", "9♣", "J♥", "J♦", "K♣", "K♦"]
        hand = [get_card_number_from_string(card) for card in card_strings]
        new_card = get_card_number_from_string('K♠')
        expected = 46  # 7♠ should be discarded
        result = find_least_useful_card(hand, new_card)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
