from pathlib import Path
import numpy as np
from loguru import logger


def save_model(funk_svd, factor_idx, finished, final_path=None):
    """
    Save model to disk, unless save_path is None
    """
    if funk_svd.save_path is None:
        return
    save_path = Path(funk_svd.save_path) / 'model'
    if finished:
        save_path = save_path / 'final'
    else:
        save_path = save_path / str(factor_idx)
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {save_path}")

    np.save(save_path / 'user_factors.npy', funk_svd.user_factors)
    np.save(save_path / 'item_factors.npy', funk_svd.item_factors)
    np.save(save_path / 'user_bias.npy', funk_svd.user_bias)
    np.save(save_path / 'item_bias.npy', funk_svd.item_bias)

    metadata = {
        'user_id_to_idx': funk_svd.user_id_to_idx,
        'item_id_to_idx': funk_svd.item_id_to_idx,
        'global_mean': funk_svd.global_mean,
        'n_factors': funk_svd.n_factors,
        'current_factor': factor_idx,
        'user_ids': funk_svd.user_ids,
        'item_ids': funk_svd.item_ids
    }
    np.save(save_path / 'metadata.npy', metadata)


def load_model(funk_svd, model_path):
    """
    Load model data into an existing FunkSVD instance
    """
    model_path = Path(model_path)
    logger.info(f"Loading model from {model_path}")

    funk_svd.user_factors = np.load(model_path / 'user_factors.npy')
    funk_svd.item_factors = np.load(model_path / 'item_factors.npy')
    funk_svd.user_bias = np.load(model_path / 'user_bias.npy')
    funk_svd.item_bias = np.load(model_path / 'item_bias.npy')
    metadata = np.load(model_path / 'metadata.npy', allow_pickle=True).item()

    funk_svd.user_id_to_idx = metadata['user_id_to_idx']
    funk_svd.item_id_to_idx = metadata['item_id_to_idx']
    funk_svd.global_mean = metadata['global_mean']
    funk_svd.n_factors = metadata['n_factors']
    funk_svd.user_ids = metadata['user_ids']
    funk_svd.item_ids = metadata['item_ids']
    return funk_svd
