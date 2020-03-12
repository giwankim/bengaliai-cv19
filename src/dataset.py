import albumentations
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class BengaliDatasetTrain:
    def __init__(self, folds, height, width, mean, std):
        # path object to the data
        self.data_path = Path(__file__).parent.resolve().parent / 'input'

        # read in the .csv file containing training data and kfold info
        df = pd.read_csv(self.data_path / 'train_folds.csv')

        # keep the relevant columns
        df = df.drop('grapheme', axis=1)
        # df = df[['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'kfold']]

        # select the rows in folds (list)
        df = df[df.kfold.isin(folds)].reset_index(drop=True)

        self.image_ids = df.image_id.values
        self.grapheme_roots = df.grapheme_root.values
        self.vowel_diacritics = df.vowel_diacritic.values
        self.consonant_diacritics = df.consonant_diacritic.values

        # data augmentation
        if len(folds) == 1:  # validation
            self.aug = albumentations.Compose([
                albumentations.Resize(height, width, always_apply=True),
                albumentations.Normalize(mean, std, always_apply=True)
            ])
        else:  # train
            self.aug = albumentations.Compose([
                albumentations.Resize(height, width, always_apply=True),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=5,
                    p=0.9
                ),
                albumentations.Normalize(mean, std, always_apply=True)
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # load the pickled image file and convert into RGB image
        image = joblib.load(
            self.data_path / f'image_pickles/{self.image_ids[idx]}.pkl')
        image = image.reshape(137, 236).astype(float)
        image = Image.fromarray(image).convert('RGB')
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'grapheme_root': torch.tensor(self.grapheme_roots[idx], dtype=torch.long),
            'vowel_diacritic': torch.tensor(self.vowel_diacritics[idx], dtype=torch.long),
            'consonant_diacritic': torch.tensor(self.consonant_diacritics[idx], dtype=torch.long),
        }
