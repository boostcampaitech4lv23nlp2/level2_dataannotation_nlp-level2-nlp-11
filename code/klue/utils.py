import math
import os
import pickle as pickle
import random

import numpy as np
import pandas as pd
import torch


def set_seed(random_seed: int) -> None:
    print(f"Set global seed {random_seed}")
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


def label_to_num(label: np.ndarray) -> list:
    num_label = []
    dict_label_to_num = {
        'country:year_of_victory': 0,
        'no_relation': 1,
        'country:faced_country': 2,
        'competition:year': 3,
        'country:population': 4,
        'country:number_of_wins': 5,
        'country:soccer_player': 6,
        'org:alternative_name': 7,
        'parent relationship:sub relationship': 8,
        'country:game_results': 9,
        'org:country_of_headquarters': 10,
        'competition:place': 11
    }
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def num_to_label(label: np.ndarray) -> list:
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open("/opt/ml/code/dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label
