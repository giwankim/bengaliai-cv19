import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pathlib import Path


if __name__ == '__main__':
    data_path = Path(__file__).parent.parent / 'input'
    df = pd.read_csv(data_path / 'train.csv')
    print(df.head())

    # Shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    x = df['image_id'].values
    y = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    # Column to indicate validation fold number
    df['kfold'] = -1

    mskf = MultilabelStratifiedKFold(n_splits=5)

    for fold, (trn_idx, val_idx) in enumerate(mskf.split(x, y)):
        print("TRAIN:", trn_idx)
        print('VAL:', val_idx)
        df.loc[val_idx, 'kfold'] = fold

    print(df['kfold'].value_counts())
    df.to_csv(data_path / 'train_folds.csv', index=False)
