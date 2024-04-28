import os
import torch
import wandb
from tqdm import tqdm
from AverageMeter import AverageMeter


class Fitter:

    def __init__(self, model, config, sweep_config=None, device="cpu"):
        self.epoch = 0
        self.config = config

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.best_summary_loss = 10 ** 5

        self.model = model.to(device)
        self.device = device

        self.sweep_config = sweep_config
        if sweep_config is not None:
            run = wandb.init(config=sweep_config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):

            summary_loss = self.train_one_epoch(train_loader)

            if summary_loss<self.best_summary_loss:
                self.save(f'{self.base_dir}/loss_{summary_loss.avg}_epoch_{self.epoch}.pth')
                self.best_summary_loss = summary_loss

            summary_loss = self.validation(validation_loader)
            if self.config.validation_scheduler:
                self.scheduler.step()

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        for data, targets in tqdm(val_loader):
            with torch.no_grad():
                images = torch.stack(data).cpu().to(self.device)
                batch_size = images.shape[0]

                transformed_dict = {'bbox': [], 'cls': [],
                                    'img_size': torch.tensor((512, 512)).unsqueeze(dim=0).cpu().to(self.device),
                                    'img_scale': torch.tensor(1).unsqueeze(dim=0).cpu().to(self.device)}
                for element in targets:
                    for key, value in element.items():
                        transformed_dict[key].append(value)

                transformed_dict['bbox'] = torch.cat(transformed_dict['bbox'], dim=0).cpu().to(self.device)
                transformed_dict['cls'] = torch.cat(transformed_dict['cls'], dim=0).cpu().to(self.device)

                loss = self.model(images, transformed_dict)["loss"]

                if self.sweep_config is not None:
                    wandb.log({"loss_test": loss.item()})

                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        for data, targets in tqdm(train_loader):
            images = torch.stack(data).cpu().to(self.device)
            batch_size = images.shape[0]

            self.optimizer.zero_grad()

            transformed_dict = {'bbox': [], 'cls': []}

            for element in targets:
                for key, value in element.items():
                    transformed_dict[key].append(value)

            transformed_dict['bbox'] = torch.cat(transformed_dict['bbox'], dim=0).cpu().to(self.device)
            transformed_dict['cls'] = torch.cat(transformed_dict['cls'], dim=0).cpu().to(self.device)

            loss = self.model(images, transformed_dict)["loss"]

            loss.backward()
            self.optimizer.step()

            summary_loss.update(loss.detach().item(), batch_size)

            if self.sweep_config is not None:
                wandb.log({"loss": loss.item()})

        self.scheduler.step(metrics=summary_loss)
        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
