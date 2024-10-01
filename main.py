import pandas as pd
import numpy as np
import tensorflow as tf
import ast

# from discard_test_set import evaluate_model_with_tally_and_print
# from generate_data import get_card_str


def process_hand(hand_string):
    hand = ast.literal_eval(hand_string)
    one_hot_hand = np.zeros((52,))
    for card in hand:
        one_hot_hand[card - 1] = 1  # Adjust for 1-indexed card numbers
    return one_hot_hand


def process_metrics(metrics_string):
    return ast.literal_eval(metrics_string)  # Convert string to list of floats


def process_discard_index(index, num_classes=52):
    one_hot = np.zeros((num_classes,))
    one_hot[index - 1] = 1  # Adjust for 1-indexed discard numbers
    return one_hot


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    input_tensors = []
    output_tensors = []

    for idx, row in df.iterrows():
        one_hot_hand = process_hand(row['Sorted Hand'])
        metrics = process_metrics(row['Metrics'])
        combined_input = np.concatenate([one_hot_hand, metrics])
        input_tensors.append(combined_input)

        one_hot_discard = process_discard_index(row['Discard Card'])
        output_tensors.append(one_hot_discard)

    input_tensors = np.array(input_tensors)
    input_tensor = tf.convert_to_tensor(input_tensors, dtype=tf.float32)

    output_tensors = np.array(output_tensors)
    output_tensor = tf.convert_to_tensor(output_tensors, dtype=tf.float32)

    return input_tensor, output_tensor

if __name__ == "__main__":
    input_tensor, output_tensor = load_and_preprocess_data('rummy_dataset.csv')

    input_shape = input_tensor.shape[1]

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(52, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(input_tensor, output_tensor, epochs=10, batch_size=32)

    model.save('discard_model.keras')

    # test_input_tensor, test_output_tensor = load_and_preprocess_data('rummy_test_dataset.csv')
    #
    # df_test = pd.read_csv('rummy_test_dataset.csv')
    # test_hands = df_test['Sorted Hand'].tolist()
    #
    # accuracy, predicted_indices, actual_indices, correct_count, incorrect_count, tally = evaluate_model_with_tally_and_print(
    #     model, test_input_tensor, test_output_tensor, test_hands
    # )