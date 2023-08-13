import obonet
from torch.utils.data import Dataset


class GoDataloader(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = obonet.read_obo(path)
        self.nodes = self.data.nodes(data=True)
        self.edges = self.data.edges(data=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        term_idx = list(self.nodes)[idx]
        term_id = term_idx[0]
        term_data = term_idx[1]
        return term_id, term_data
