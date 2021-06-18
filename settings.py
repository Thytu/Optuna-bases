"""Import global variables"""

from torch import device, cuda


def init(epochs, batch_size, train_exemples, test_exemples, nb_trials, timeout) -> None:
    """Init all global variables according to provided optins"""

    global CLASSES
    global DEVICE
    global EPOCHS
    global BATCH_SIZE
    global N_TRAIN_EXAMPLES
    global N_VALID_EXAMPLES
    global N_TRIALS
    global TIMEOUT

    CLASSES = 10
    DEVICE = device('cuda') if cuda.is_available() else device('cpu')
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    N_TRAIN_EXAMPLES = train_exemples
    N_VALID_EXAMPLES = test_exemples


def verify() -> None:
    """Verify that all global variables are rightly assigned"""

    print("CLASSES:", CLASSES)
    print("DEVICE:", DEVICE)
    print("EPOCHS:", EPOCHS)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("N_TRAIN_EXAMPLES:", N_TRAIN_EXAMPLES)
    print("N_VALID_EXAMPLES:", N_VALID_EXAMPLES)
