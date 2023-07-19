from data_loaders.base_loader import DataLoader
import pyarrow.parquet as pq
from tqdm import tqdm

class MTDataLoader(DataLoader):
    def __init__(self):
        pass

    def load(self, path: str, verbose = True):
        dataset = pq.read_table(path)
        dataset = dataset.to_pandas()
        dataset = dataset["translation"].to_list()
        eng = []
        rus = []
        for row in tqdm(dataset):
            eng.append(row['en'])
            rus.append('[start]'+row['ru']+'[end]')

        return eng, rus