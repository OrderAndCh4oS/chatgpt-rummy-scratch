import ast

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from generate_data import get_card_str
from main import load_and_preprocess_data


def evaluate_model_with_tally_and_print(model, test_input, test_output, test_hands):
    """Evaluate the model, calculate accuracy, and print hand and discard details."""
    predictions = model.predict(test_input)

    predicted_indices = np.argmax(predictions, axis=1)
    actual_indices = np.argmax(test_output, axis=1)

    accuracy = np.mean(predicted_indices == actual_indices)
    tally = predicted_indices == actual_indices

    correct_count = np.sum(tally)
    incorrect_count = len(tally) - correct_count

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Correct Predictions: {correct_count}")
    print(f"Incorrect Predictions: {incorrect_count}")

    # Print hand, discarded card, and discard index for the first 10 predictions
    for i in range(10):
        sorted_hand = ast.literal_eval(test_hands[i])  # Parse the string representation of the hand
        predicted_card = predicted_indices[i] + 1  # Convert back to 1-indexed card number
        actual_card = actual_indices[i] + 1
        correct = "Correct" if tally[i] else "Incorrect"

        print(f"\nPrediction {i + 1} ({correct}):")
        print(f"Hand: {[get_card_str(card) for card in sorted_hand]}")
        print(f"Predicted Discard: {get_card_str(predicted_card)}")
        print(f"Actual Discard: {get_card_str(actual_card)}")
        print(f"Predicted Discard Index: {predicted_indices[i] + 1}, Actual Discard Index: {actual_indices[i] + 1}")

    return accuracy, predicted_indices, actual_indices, correct_count, incorrect_count, tally


if __name__ == '__main__':
    model = tf.keras.models.load_model('discard_model.keras')

    model.summary()

    test_input_tensor, test_output_tensor = load_and_preprocess_data('rummy_test_dataset.csv')

    df_test = pd.read_csv('rummy_test_dataset.csv')
    test_hands = df_test['Sorted Hand'].tolist()

    accuracy, predicted_indices, actual_indices, correct_count, incorrect_count, tally = evaluate_model_with_tally_and_print(
        model, test_input_tensor, test_output_tensor, test_hands
    )
