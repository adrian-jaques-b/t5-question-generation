from t5qg.data import get_dataset

# data = 'squad'
data = 'tydiqa'
for i in ['train', 'test', 'dev']:
    get_dataset(data, split=i)
    # get_dataset('squad', split=i, no_prefix=True)
