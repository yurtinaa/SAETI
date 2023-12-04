
from .nan_func import *
from .log_func import *

@dataclass
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss: torch.nn.functional
    dataset: MISSING
    score_func: MISSING
    scheduler: MISSING = None
    check_out: MISSING = None
    add_nan: str = 'random'
    __func_nan: MISSING = None
    device: torch.device = torch.device('cpu')
    epochs_count: int = 1000
    bar: bool = False
    log: bool = True
    early_stop: int = 50
    wandb_log: bool = False
    name_log: str = ''

    func_log: MISSING = print_log

    def __post_init__(self):
        self.best_val_loss = float('inf')
        self.best_epoch_i = 0
        self.epoch = 0
        self.best_model = copy.deepcopy(self.model)

        if not self.log and self.bar:
            self.epoch_bar = tqdm(np.arange(self.epochs_count))
        else:
            self.epoch_bar = np.arange(self.epochs_count)

        if self.add_nan != 'None':
            self.__func_nan = dict_func_nan[self.add_nan]
        # print(self.__func_nan)

        if wandb.run is not None:
            self.func_log = wandb_log

    def __get_train_batch_bar(self, type_):
        loader = self.dataset.get_loader(type_)
        if self.log and self.bar:
            return tqdm(loader,
                        desc=f"{self.epoch + 1}/{self.epochs_count}")
        else:
            return loader

    def __one_epoch(self, type_):
        batch_bar = self.__get_train_batch_bar(type_)
        loss = 0
        score = 0
        i = 0
        for i, (x, y) in enumerate(batch_bar):
            self.optimizer.zero_grad()
            
            x = x.to(self.device)
            if self.__func_nan is not None:
                x = self.__func_nan(x)
            y = y.to(self.device)

            if type_ == 'train':
                grad = True
            else:
                grad = False

            with torch.set_grad_enabled(grad):
                y_pred = self.model.forward(x)
                # print(y_pred[0].grad)
                loss_value = self.loss(x, y, y_pred)
                loss += loss_value.detach().cpu().item()
                score += self.score_func(x.cpu().detach(),
                                         y.cpu().detach(),
                                         y_pred.cpu().detach(),
                                         )
                if type_ == 'train':
                    # loss_value = self.loss(y_pred, y_true)
                    loss_value.backward()
                    self.optimizer.step()
            # print(i)
        #  del x, y, y_pred
        # torch.cuda.empty_cache()
        # gc.collect()
        loss /= (i + 1)
        score /= (i + 1)
        return loss, score

    def __print(self, score_train, score_val):
        dict_log = {
            f'{self.name_log}:epoch': self.epoch + 1,
            f'{self.name_log}:train': float(score_train),
            f'{self.name_log}:val': float(score_val),
        }
        self.func_log(dict_log)

    def __update_model(self,
                       val_loss):
        if val_loss < self.best_val_loss:
            self.best_epoch_i = self.epoch
            self.best_val_loss = val_loss
            self.best_model = copy.deepcopy(self.model)
            if self.log:
                log_func('new best model')

    def __check_scheduler(self,
                          val_loss,
                          score_train,
                          score_val):
        if self.scheduler is not None:
            self.scheduler.step(val_loss)
        if self.check_out is not None:
            if self.check_out(score_val) and self.check_out(score_train):
                return True
        return False

    def train(self):
        history = {"train": [],
                   "val": []}
        for self.epoch in self.epoch_bar:
            self.model.train()
            _, score_train = self.__one_epoch('train')
            self.model.eval()
            with torch.no_grad():
                loss, score_val = self.__one_epoch('val')
            self.__print(score_train, score_val)
            self.__update_model(loss)

            history["train"].append(score_train)
            history["val"].append(score_val)

            if self.__check_scheduler(loss,
                                      score_val,
                                      score_train) or (self.epoch - self.best_epoch_i) > self.early_stop:
                break
        return self.best_model, self.best_epoch_i
