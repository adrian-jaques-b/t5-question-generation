import json
from glob import glob

for i in glob('t5qg_output/grid_search/mt5_small/model_*/trainer_config.json'):
    with open(i) as f:
        tmp = json.load(f)
    tmp['batch'] = 16
    tmp['gradient_accumulation_steps'] = 32
    with open(i, 'w') as f:
        json.dump(tmp, f)

with open('t5qg_output/grid_search/mt5_small/config_dynamic.json') as f:
    tmp = json.load(f)
    tmp['batch'] = [16]

with open('t5qg_output/grid_search/mt5_small/config_dynamic.json', 'w') as f:
    json.dump(tmp, f)

with open('t5qg_output/grid_search/mt5_small/config_static.json') as f:
    tmp = json.load(f)
    tmp['gradient_accumulation_steps'] = 32

with open('t5qg_output/grid_search/mt5_small/config_static.json', 'w') as f:
    json.dump(tmp, f)
