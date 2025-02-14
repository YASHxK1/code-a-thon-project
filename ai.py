# Import necessary libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Import Hugging Face datasets library
from datasets import load_dataset

# Load the dataset from Hugging Face
def load_huggingface_dataset():
    dataset = load_dataset("ShivomH/Mental-Health-Conversations")
    return dataset['train']

# Create a dataset from the Hugging Face dataset
def create_dataset_from_huggingface():
    dataset = load_huggingface_dataset()
    conversations = []
    for example in dataset:
        conversations.append((example['input'], example['response']))
    return conversations

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\\w\\s\\?]', '', text)
    text = re.sub(r'\\s+', ' ', text)
    return text.strip()

# Data preparation class
class DataPreparator:
    def __init__(self, max_sequence_length=50):
        self.tokenizer = None
        self.max_sequence_length = max_sequence_length

    def prepare_data(self, conversations):
        # Split conversations into questions and answers
        questions, answers = zip(*conversations)

        # Preprocess texts
        questions = [preprocess_text(q) for q in questions]
        answers = [preprocess_text(a) for a in answers]

        # Add start and end tokens to answers
        answers = [f'<start> {a} <end>' for a in answers]

        # Create and fit tokenizer
        self.tokenizer = Tokenizer(filters='')
        self.tokenizer.fit_on_texts(questions + answers)

        # Convert texts to sequences
        encoder_input_data = self.tokenizer.texts_to_sequences(questions)
        decoder_input_data = self.tokenizer.texts_to_sequences(answers)

        # Pad sequences
        encoder_input_data = pad_sequences(encoder_input_data, maxlen=self.max_sequence_length, padding='post')
        decoder_input_data = pad_sequences(decoder_input_data, maxlen=self.max_sequence_length, padding='post')

        # Create decoder output data (shifted by one position)
        decoder_output_data = np.zeros_like(decoder_input_data)
        decoder_output_data[:, :-1] = decoder_input_data[:, 1:]

        return encoder_input_data, decoder_input_data, decoder_output_data

    @property
    def vocab_size(self):
        return len(self.tokenizer.word_index) + 1

# Mental Health Consultation Model
class MentalHealthLLM:
    def __init__(self, vocab_size, max_sequence_length, embedding_dim=256, lstm_units=256):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.encoder_model = None
        self.decoder_model = None
        self.model = self.build_model()

    def build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(self.max_sequence_length,))
        encoder_embedding = Embedding(self.vocab_size, self.embedding_dim)(encoder_inputs)
        encoder_lstm = LSTM(self.lstm_units, return_state=True, return_sequences=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Add self-attention to encoder
        attention = MultiHeadAttention(num_heads=8, key_dim=32)
        attention_output = attention(encoder_outputs, encoder_outputs)
        encoder_outputs = LayerNormalization()(attention_output + encoder_outputs)

        # Decoder
        decoder_inputs = Input(shape=(self.max_sequence_length,))
        decoder_embedding = Embedding(self.vocab_size, self.embedding_dim)
        decoder_lstm = LSTM(self.lstm_units, return_sequences=True, return_state=True)

        # Connect decoder to encoder
        decoder_embedded = decoder_embedding(decoder_inputs)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)

        # Add attention between encoder and decoder
        attention = MultiHeadAttention(num_heads=8, key_dim=32)
        attention_output = attention(decoder_outputs, encoder_outputs)
        decoder_outputs = LayerNormalization()(attention_output + decoder_outputs)

        # Add dropout for regularization
        decoder_outputs = Dropout(0.5)(decoder_outputs)
        decoder_dense = Dense(self.vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Create the full model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create inference models
        self.encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

        decoder_state_input_h = Input(shape=(self.lstm_units,))
        decoder_state_input_c = Input(shape=(self.lstm_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedded,
            initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )

        return model

    def train(self, encoder_input_data, decoder_input_data, decoder_output_data,
              batch_size=32, epochs=100, validation_split=0.2):

        # Implement early stopping and model checkpointing
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]

        # Train the model
        history = self.model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_output_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks
        )

        return history

# Main execution
def main():
    print("Loading dataset from Hugging Face...")
    conversations = create_dataset_from_huggingface()

    print("Preparing data...")
    data_preparator = DataPreparator(max_sequence_length=50)
    encoder_input_data, decoder_input_data, decoder_output_data = data_preparator.prepare_data(conversations)

    print("Building and training model...")
    model = MentalHealthLLM(
        vocab_size=data_preparator.vocab_size,
        max_sequence_length=50,
        embedding_dim=256,
        lstm_units=256
    )

    history = model.train(
        encoder_input_data,
        decoder_input_data,
        decoder_output_data,
        batch_size=32,
        epochs=50,
        validation_split=0.2
    )

    print("\nTesting the model with sample inputs...")
    test_inputs = [
        "I've been feeling really sad lately",
        "I'm having trouble sleeping",
        "I feel anxious all the time"
    ]

    for test_input in test_inputs:
        response = model.generate_response(test_input, data_preparator.tokenizer)
        print(f"\nInput: {test_input}")
        print(f"Response: {response}")

if __name__ == "__main__":
    print("Starting Mental Health LLM training...")
    main()