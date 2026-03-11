# Global configuration file

DATA_PATH = "data/raw_midi"
PROCESSED_PATH = "data/processed"
TRAIN_SPLIT_PATH = "data/train_test_split"

SEQUENCE_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

INPUT_DIM = 128
HIDDEN_DIM = 256
LATENT_DIM = 128

DEVICE = "cuda"