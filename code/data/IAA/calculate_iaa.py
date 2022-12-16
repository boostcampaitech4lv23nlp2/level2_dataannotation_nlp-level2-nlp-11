import pandas as pd
import numpy as np
from fleiss import fleissKappa

result = pd.read_excel('./Flue_IAA.xlsx',engine='openpyxl')
print(result.keys())

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

for idx, row in enumerate(result.iterrows()):
    new_row = []
    for tag in row[1]:
        if tag in dict_label_to_num:
            new_row.append(dict_label_to_num[tag])
        else:
            new_row.append(None)
    # print(new_row)
    result.iloc[idx] = new_row

result = result.to_numpy()
num_classes = len(dict_label_to_num)

transformed_result = []
for i in range(len(result)):
    temp = np.zeros(num_classes)
    for j in range(len(result[i])):
        temp[int(result[i][j])] += 1
    transformed_result.append(temp.astype(int).tolist())

kappa = fleissKappa(transformed_result,len(result[0]))
print(kappa)