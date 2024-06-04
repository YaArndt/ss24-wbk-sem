# Number of epochs to train the model
EPOCHS = 150

# Assign positive and negative classes
CLASSES = {
    "POS": "NIO",
    "NEG": "IO"
}

# Define the size of the trainig and testing datasets
TRAIN_SIZE_RATIO = 0.75

# Define the batch size for the DataLoader
BATCH_SIZE = 16

# Number of batches to visualize before training
VIZ_BATCHES = 10

# Shuffle the dataset
SHUFFLE = True