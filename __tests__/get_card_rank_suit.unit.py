import unittest

from generate_data import get_card_rank_suit


class TestCardRankSuitConversion(unittest.TestCase):

    def setUp(self):
        global rank_symbols_list, suit_symbols_list
        # Assuming rank_symbols_list and suit_symbols_list are defined globally
        rank_symbols_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        suit_symbols_list = ['♣', '♦', '♥', '♠']

    def test_valid_card_number_1(self):
        # Ace of Spades
        self.assertEqual(get_card_rank_suit(1), ('A', '♣'))

    def test_valid_card_number_13(self):
        # King of Spades
        self.assertEqual(get_card_rank_suit(13), ('K', '♣'))

    def test_valid_card_number_14(self):
        # Ace of Hearts
        self.assertEqual(get_card_rank_suit(14), ('A', '♦'))

    def test_valid_card_number_26(self):
        # King of Hearts
        self.assertEqual(get_card_rank_suit(26), ('K', '♦'))

    def test_valid_card_number_27(self):
        # Ace of Diamonds
        self.assertEqual(get_card_rank_suit(27), ('A', '♥'))

    def test_valid_card_number_39(self):
        # King of Diamonds
        self.assertEqual(get_card_rank_suit(39), ('K', '♥'))

    def test_valid_card_number_40(self):
        # Ace of Clubs
        self.assertEqual(get_card_rank_suit(40), ('A', '♠'))

    def test_valid_card_number_52(self):
        # King of Clubs
        self.assertEqual(get_card_rank_suit(52), ('K', '♠'))

    def test_invalid_card_number_0(self):
        with self.assertRaises(ValueError):
            get_card_rank_suit(0)

    def test_invalid_card_number_53(self):
        with self.assertRaises(ValueError):
            get_card_rank_suit(53)

if __name__ == '__main__':
    unittest.main()
