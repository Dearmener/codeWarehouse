import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import numpy as np
class DiabetesDataset(Dataset):
    def __init__(self,file_path) -> None:
        super().__init__()
        self.xy =  np.loadtxt(file_path,delimiter=",",dtype=np.float32)
        self.len = self.xy.shape[0]
        self.x_data = torch.from_numpy(self.xy[:,:-1])
        self.y_data = torch.from_numpy(self.xy[:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.len 

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = torch.nn.Linear(8,6) 
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x)) 
        x = self.sigmoid(self.linear3(x))
        return x

dataset = DiabetesDataset('diabetes.csv.gz')
data_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)

model = Model()
cret = torch.nn.BCELoss(size_average=True)
opti = torch.optim.SGD(model.parameters(),lr = 0.01)

for epoch in range(100): 
   for i,datas in enumerate(data_loader,0):
       xs,ys = datas
       y_h = model(xs)
       loss = cret(y_h,ys)
       opti.zero_grad()
       loss.backward()
       opti.step()
       print(f"epoch is {epoch},loss is {loss.item()}")
