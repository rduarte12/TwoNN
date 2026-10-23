import numpy as np
from sklearn.neighbors import BallTree
from sklearn.model_selection import StratifiedKFold

def create_folds(features, labels, n_folds=10):
    kf= StratifiedKFold(n_splits=n_folds, shuffle=False)
    return list(kf.split(features, labels))


def run_ball_tree(features, k):
    if not isinstance(features, np.ndarray):
        raise ValueError('As features devem estar em formato numpy')
    if features.ndim != 2:
        raise ValueError('As features devem ser um vetor 2D (n_samples, n_features)')
    
    tree= BallTree(features, leaf_size=500)
    _, ranked_lists= tree.query(features, k=k)
    return ranked_lists