import numpy as np
import multiprocessing as mp
import torch
import time
from helpers import im2arr


class Trainer(object):

    def __init__(
        self, model, loss_fn, optimizer, cpu_count, mean_global, px,
        model_name, torch_dtype
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.cpu_count = cpu_count
        self.mean_global = mean_global
        self.px = px
        self.model_name = model_name
        self.dtype = torch_dtype

        # Will be set in fit method.
        self.batch_size = None
        self.epoch_partion = None
        self.train = None
        self.ytrain = None
        self.valid = None
        self.yvalid = None

        self.loss_hist = {'train_loss': [], 'train_acc': [],
                          'valid_loss': [], 'valid_acc': []}
        self.pool = None

    def prepare_batch(self, x, y):
        batch_ids = np.random.choice(x.shape[0], self.batch_size)
        X_batch = self.pool.map(im2arr, x[batch_ids])

        X_batch = np.array(X_batch).reshape(-1, 1, self.px, self.px)
        X_batch -= self.mean_global
        X_batch = torch.tensor(X_batch).type(self.dtype)
        y_batch = torch.tensor(y[batch_ids]).type(self.dtype).long()
        return X_batch, y_batch

    def make_step(self, x, y):
        preds = self.model(x)
        loss = self.loss_fn(preds, y).type(self.dtype)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return preds, loss.item()

    def metric(self, preds, y):
        # Accuracy
        return np.array(preds.argmax(dim=1) == y).mean()

    def evaluate(self):
        self.pool = mp.Pool(self.cpu_count)
        eval_results = []
        t_start = time.time()
        batches_to_eval = int(self.valid.shape[0] / self.batch_size * self.epoch_partion)
        self.model.eval()
        for i in range(1, batches_to_eval+1):
            X_batch, y_batch = self.prepare_batch(self.valid, self.yvalid)
            preds = self.model(X_batch)
            eval_results.append(self.metric(preds, y_batch))

            if i % 100 == 0:
                self.pool.close()
                self.pool.join()
                self.pool = mp.Pool(self.cpu_count)

        eval_results = np.mean(eval_results)
        print('Eval accuracy {}%. Done in {} seconds on {} batches.'.format(
            round(eval_results*100, 2), int(time.time() - t_start), batches_to_eval)
        )
        self.pool.close()
        self.pool.join()
        return eval_results

    def fit_epoch(self):
        batches_per_epoch = int(self.train.shape[0] / self.batch_size * self.epoch_partion)

        self.pool = mp.Pool(self.cpu_count)
        t_start = time.time()
        self.model.train()
        for i in range(1, batches_per_epoch + 1):

            X_batch, y_batch = self.prepare_batch(self.train, self.ytrain)
            preds, batch_loss = self.make_step(X_batch, y_batch)
            self.loss_hist['train_loss'].append(batch_loss)
            batch_acc = self.metric(preds, y_batch)
            self.loss_hist['train_acc'].append(batch_acc)

            if i % 100 == 0:
                self.pool.close()
                self.pool.join()
                self.pool = mp.Pool(self.cpu_count)
            print("Batch {}/{}. Batch accuracy {} Seconds elapsed {} ".format(
                i, batches_per_epoch, round(np.mean(self.loss_hist['train_acc'][-1000:]), 3),
                int(time.time() - t_start)), end='\r')
        self.pool.close()
        self.pool.join()
        print()

    def fit(self, train, ytrain, valid, yvalid, batch_size, epoch_partion, epochs_num, patience):
        self.train = train
        self.ytrain = ytrain
        self.valid = valid
        self.yvalid = yvalid

        self.batch_size = batch_size
        self.epoch_partion = epoch_partion

        no_imprv_epochs = 0
        epochs_since_lr_lowering = 0

        for epoch in range(1, epochs_num+1):
            print('Started epoch', epoch)
            self.fit_epoch()
            eval_metric = self.evaluate()
            self.loss_hist['valid_acc'].append(eval_metric)

            epochs_since_lr_lowering += 1
            if round(eval_metric, 4) < round(np.max(self.loss_hist['valid_acc']), 4):
                no_imprv_epochs += 1
            else:
                no_imprv_epochs = 0
                torch.save(self.model.state_dict(), self.model_name)

            # reduce lr if on plateau.
            if no_imprv_epochs >= patience/2 and epochs_since_lr_lowering >= patience/2:
                self.model.load_state_dict(torch.load(self.model_name))
                current_lr = self.optimizer.param_groups[0]['lr']
                for g in self.optimizer.param_groups:
                    g['lr'] = current_lr / 5
                print('Learning rate reduced to', self.optimizer.param_groups[0]['lr'])
                epochs_since_lr_lowering = 0

            if no_imprv_epochs >= patience:
                self.model.load_state_dict(torch.load(self.model_name))
                break

            print()

    def predict_proba(self, x):
        preds = []
        self.pool = mp.Pool(self.cpu_count)
        for i in range(0, x.shape[0], self.batch_size):
            X_batch = self.pool.map(im2arr, x[i: i+self.batch_size])
            X_batch = np.array(X_batch).reshape(-1, 1, self.px, self.px)
            X_batch = torch.tensor(X_batch).float()

            preds_tmp = self.model(X_batch)
            preds_tmp = preds_tmp.detach().numpy()
            preds.append(preds_tmp)
            if i % 100 == 0:
                self.pool.close()
                self.pool.join()
                self.pool = mp.Pool(self.cpu_count)

        self.pool.close()
        self.pool.join()
        return np.concatenate(preds, axis=0)