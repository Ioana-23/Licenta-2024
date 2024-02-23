import torch.utils.data
from torch import nn
from torchvision.transforms import v2
from tqdm import tqdm

from model import GoogLeNet
from readImages import MyDataset

trainDataset = MyDataset(base_folder="D:/Licenta/Proiect/data/",
                         transforms=v2.Compose([v2.Resize(1996), v2.CenterCrop(1890), v2.ToImage(),
                                                v2.ToDtype(torch.float32, scale=True)]),
                         threshold=2, split_name="train")
testDataset = MyDataset(base_folder="D:/Licenta/Proiect/data/",
                        transforms=v2.Compose([v2.Resize(1996), v2.CenterCrop(1890), v2.ToImage(),
                                               v2.ToDtype(torch.float32, scale=True)]),
                        threshold=2, split_name="test")
bs = 1
trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=bs, shuffle=True, num_workers=0)
testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=bs, shuffle=True, num_workers=0)

model = GoogLeNet()


# model = torchvision.models.GoogLeNet(num_classes=4)

def train_one_epoch(loader, googlenet_model, googlenet_optimizer, googlenet_loss_function, googlenet_scheduler):
    for data, targets in tqdm(loader):
        predictions = googlenet_model(data)
        loss = googlenet_loss_function(predictions[0], targets)

        googlenet_optimizer.zero_grad()
        loss.backward()
        googlenet_optimizer.step()

        # wandb.log({"loss": loss.item()})
    googlenet_scheduler.step()


loss_function = nn.CrossEntropyLoss()
learning_rate = 0.0001
number_of_epochs = 2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
for epoch in range(number_of_epochs):
    train_one_epoch(trainDataloader, model, optimizer, loss_function, scheduler)
    # accuracy, iou, _ = check_metrics(test_dataloader, model)
