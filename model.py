import time
from typing import List
import torch
import copy

from torch.utils.data import dataloader
from .data_objects import Stage, DataHolder
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def holder_priority(data_holder: DataHolder):
    return data_holder.stage.value

class Model:
    def __init__(self, model, criterion=None, optimizer=None, scheduler=None, data_holders:List[DataHolder] = None) -> None:
        self.model = model
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_model = model
        self.best_acc = 0.0
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        if data_holders:
            self.data_holders = sorted(data_holders, key=holder_priority)
        self.current_epoch = 0

    def fit(self, num_epochs):
        since = time.time()
        max_epoch = self.current_epoch + num_epochs
        for _ in range(num_epochs):
            self.run_epoch(max_epoch, self.current_epoch)
            self.current_epoch += 1

        time_elapsed = time.time() - since


        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print('Best val Acc: {:4f}'.format(self.best_acc))

        self.best_model.load_state_dict(self.best_model_wts)

    def run_epoch(self, num_epochs, epoch):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for data_holder in self.data_holders:
            self._run_on_holder(data_holder)

        print()

    def _run_on_holder(self, data_holder: DataHolder):
        return self._step(
            data_holder = data_holder,
            train_mode = data_holder.stage is Stage.TRAIN,
            val_mode = data_holder.stage is Stage.VAL
        )

    def _step(self, data_holder: DataHolder, train_mode = False, val_mode = False):
        if train_mode:
            self.model.train()  
        else:
            self.model.eval()   

        running_loss = 0.0
        running_corrects = 0

        all_predictions = []

        dl = data_holder.dataloader

        loop = tqdm(enumerate(dl), total=len(dl), leave=False)
        for batch_idx, (inputs, labels) in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if self.optimizer:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train_mode):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                if train_mode:
                    loss.backward()
                    self.optimizer.step()

                    # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_predictions += preds
            loop.set_postfix(loss = f'{running_loss/len(all_predictions):3f}', acc=f'{(running_corrects/len(all_predictions)).item():3f}')

        epoch_loss = running_loss / data_holder.size
        epoch_acc = running_corrects.double() / data_holder.size

        print(f'{data_holder.name} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')  
        
        if train_mode:
            self.scheduler.step()
        if val_mode and epoch_acc > self.best_acc:
            self.best_acc = epoch_acc
            self.best_model_wts = copy.deepcopy(self.model.state_dict())

        return all_predictions

    def __call__(self, data_holder):
        return self._step(data_holder)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
