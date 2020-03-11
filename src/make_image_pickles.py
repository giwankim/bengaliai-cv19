import joblib
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm


def save_fn(image_id, image, path):
    joblib.dump(image, path / f'image_pickles/{image_id}.pkl')


if __name__ == '__main__':
    data_path = Path(__file__).parent.resolve().parent / 'input'
    files = data_path.glob('train_*.feather')

    for f in files:
        df = pd.read_feather(f)

        image_ids = df['image_id'].values
        image_array = df.drop('image_id', axis=1).values

        Parallel(n_jobs=1, backend='multiprocessing')(
            delayed(save_fn)(image_id, image, data_path)
            for image_id, image in tqdm(zip(image_ids, image_array), total=len(image_ids))
        )
