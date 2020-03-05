import pandas as pd
import numpy as np
import joblib
import glob
from tqdm import tqdm

if __name__ == '__main__':
    files = glob.glob('../input/train_*.parquet')
    for f in files:
        df = pd.read_parquet(f)
        image_idx = df['image_id'].values
        image_arr = df.drop('image_id', axis=1).values
        for j, img_id in tqdm(enumerate(image_idx), total=len(image_idx)):
            joblib.dump(image_arr[j], f'../input/image_pickles/{img_id}.pkl')
