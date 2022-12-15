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
        'competition:year': 0,
        'competition:place': 1,
        'org:country_of_headquarters': 2,
        'country:number_of_wins': 3,
        'parent relationship:sub relationship': 4,
        'country:population': 5,
        'country:faced_country': 6,
        'org:alternative_name': 7,
        'no_relation': 8,
        'country:soccer_player': 9,
        'country:game_results': 10,
        'country:FIFA_ranking': 11,
        'country:year_of_victory': 12
        }
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def num_to_label(label: np.ndarray) -> list:
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    dict_num_to_label = {
        0: 'competition:year',
        1: 'competition:place',
        2: 'org:country_of_headquarters',
        3: 'country:number_of_wins',
        4: 'parent relationship:sub relationship',
        5: 'country:population',
        6: 'country:faced_country',
        7: 'org:alternative_name',
        8: 'no_relation',
        9: 'country:soccer_player',
        10: 'country:game_results',
        11: 'country:FIFA_ranking',
        12: 'country:year_of_victory'
        }
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label
