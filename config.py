# Number of epochs to train the model
EPOCHS = 10

# Assign positive and negative classes
CLASSES = {
    "POS": "NIO",
    "NEG": "IO"
}

# Define the size of the trainig and testing datasets
TRAIN_SIZE_RATIO = 0.8

# Define the batch size for the DataLoader
BATCH_SIZE = 32

# Shuffle the dataset
SHUFFLE = True