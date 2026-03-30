
from typing import List, Union
from pytorch_lightning import LightningModule, Trainer
import lightning as L
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MyDataModule(L.LightningDataModule):
    def __init__(self, train_data, batch_size=32):
        super().__init__()
        self.train_data = train_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Create your dataset here
        self.train_dataset = MyDataset(self.train_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)



class Job:
    def __init__( data_loader, model, job: str = None ):
        self._inits = []
        self._sorts = []  
        
        if job:
            with open(job, 'r') as f:
                job = json.load(f)
                
                
                
                
    def run(self):
        
        for isort, sort in enumerate( self.sorts ):
            
            
            for iinit, init in enumerate(self.inits):

            

    @property
    def sorts(self) -> List[int]:
        return self._sorts
    
    @sorts.setter
    def sorts(self, value: Union[int, List[int]]):
        self._sorts = list(range(value)) if type(value) == int else value

    @property
    def inits(self) -> List[int]:
        return self._inits
    
    @inits.setter
    def inits(self, value: Union[int, List[int]]):
        self._inits = list(range(value)) if type(value) == int else value
            