import ast
import datetime
import os
import torch
import torch.nn as nn
from dataset import BengaliDatasetTrain
from model_dispatcher import MODEL_DISPATCHER
from pytorchtools import EarlyStopping
from sklearn.metrics import recall_score
from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAINING_FOLDS_CSV = os.environ.get('TRAINING_FOLDS_CSV')
IMG_HEIGHT = int(os.environ.get('IMG_HEIGHT'))
IMG_WIDTH = int(os.environ.get('IMG_WIDTH'))
EPOCHS = int(os.environ.get('EPOCHS'))

TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE'))
TEST_BATCH_SIZE = int(os.environ.get('TEST_BATCH_SIZE'))

MODEL_MEAN = ast.literal_eval(os.environ.get('MODEL_MEAN'))
MODEL_STD = ast.literal_eval(os.environ.get('MODEL_STD'))

TRAINING_FOLDS = ast.literal_eval(os.environ.get('TRAINING_FOLDS'))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get('VALIDATION_FOLDS'))
BASE_MODEL = os.environ.get('BASE_MODEL')


def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    return (l1 + l2 + l3) / 3


def train(model, optimizer, loss_fn, data_loader):
    model.train()

    for sample in tqdm(data_loader, total=len(data_loader)):
        image = sample['image']
        grapheme_root = sample['grapheme_root']
        vowel_diacritic = sample['vowel_diacritic']
        consonant_diacritic = sample['consonant_diacritic']

        # Load the tensors to the GPU
        image = image.to(device=DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(device=DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(device=DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(
            device=DEVICE, dtype=torch.long)

        # Forward pass the image through the model
        outputs = model(image)

        # Compute the loss
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        # Back-propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(model, data_loader):
    model.eval()
    total_loss = 0.
    correct, total = 0, 0
    for sample in tqdm(data_loader, total=len(data_loader)):
        image = sample['image']
        grapheme_root = sample['grapheme_root']
        vowel_diacritic = sample['vowel_diacritic']
        consonant_diacritic = sample['consonant_diacritic']

        # Load the tensors to the GPU
        image = image.to(device=DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(device=DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(device=DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(
            device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            outputs = model(image)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            # Compute accuracy
            for output, target in zip(outputs, targets):
                total += len(output)
                correct += int((output.argmax(1) == target).sum())

    return total_loss, correct / total


def get_dataloaders():
    # Training data
    train_dataset = BengaliDatasetTrain(
        folds=TRAINING_FOLDS,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    # Validation data
    val_dataset = BengaliDatasetTrain(
        folds=VALIDATION_FOLDS,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # load validation data in batches
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=TEST_BATCH_SIZE,
        num_workers=4
    )

    return train_loader, val_loader


def main():
    # Get data
    train_loader, val_loader = get_dataloaders()

    # Get model from dispatcher
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)

    # Train on multiple GPU's if possible
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE)

    # Specify optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

    # Callbacks
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           patience=5, factor=0.3, verbose=True)

    for epoch in range(1, EPOCHS + 1):
        train(model, optimizer, loss_fn, train_loader)
        loss, acc = validate(model, val_loader)
        scheduler.step(acc)

        if epoch == 1 or epoch % 10 == 0:
            print(f'{datetime.datetime} Epoch {epoch}, Validation loss {loss}, Validation acc {acc}')

    # Save weights
    torch.save(model.state_dict(),
               f'weights/{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.pth')


if __name__ == '__main__':
    main()
