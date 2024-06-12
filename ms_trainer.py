from __future__ import division
from __future__ import print_function

import datetime
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from ms_utils import get_optimizer, get_loss
from grda_mindspore import GRDA
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore.train.callback import LossMonitor

class Trainer:
    def __init__(self, model=None, train_gen=None, test_gen=None, valid_gen=None,
                 opt1='adam', opt2='grda', epsilon=1e-8, initial_accumulator_value=1e-8, momentum=0.95,
                 loss_fn=None, pos_weight=1.0,
                 n_epoch=1, train_per_epoch=10000, test_per_epoch=10000, early_stop_epoch=5,
                 batch_size=2000, learning_rate=1e-2, decay_rate=0.95, learning_rate2=1e-2, decay_rate2=1,
                 logdir=None, load_ckpt=False, ckpt_time=10, grda_c=0.005, grda_mu=0.51,
                 test_every_epoch=1, retrain_stage=0):
        self.model = model
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.valid_gen = valid_gen
        optimizer1 = get_optimizer(opt1)
        optimizer2 = get_optimizer(opt2)
        self.loss_fn = loss_fn
        self.pos_weight = pos_weight
        self.n_epoch = n_epoch
        self.train_per_epoch = train_per_epoch + 1
        self.early_stop_epoch = early_stop_epoch
        self.test_per_epoch = test_per_epoch
        self.batch_size = batch_size
        self._learning_rate = learning_rate
        self.decay_rate = decay_rate
        self._learning_rate2 = learning_rate2
        self.decay_rate2 = decay_rate2
        self.logdir = logdir
        self.ckpt_time = ckpt_time
        self.epsilon = epsilon
        self.test_every_epoch = test_every_epoch
        self.retrain_stage = retrain_stage

        self.call_auc = roc_auc_score
        self.call_loss = log_loss

        self.learning_rate = Parameter(Tensor(learning_rate, ms.float32), name='learning_rate')
        self.learning_rate2 = Parameter(Tensor(learning_rate2, ms.float32), name='learning_rate2')
        self.global_step = Parameter(Tensor(0, ms.int32), name='global_step')

        if opt1 == 'adam':
            opt1 = optimizer1(params=self.model.trainable_params(), learning_rate=self.learning_rate, eps=self.epsilon)
        elif opt1 == 'adagrad':
            opt1 = optimizer1(params=self.model.trainable_params(), learning_rate=self.learning_rate, initial_accumulator_value=initial_accumulator_value)
        elif opt1 == 'momentum':
            opt1 = optimizer1(params=self.model.trainable_params(), learning_rate=self.learning_rate, momentum=momentum)
        elif opt1 == 'grda':
            opt1 = GRDA(params=self.model.trainable_params(), learning_rate=self.learning_rate, c=grda_c, mu=grda_mu)
        else:
            opt1 = optimizer1(params=self.model.trainable_params(), learning_rate=self.learning_rate)

        if opt2 == 'grda':
            opt2 = GRDA(params=self.model.trainable_params(), learning_rate=self.learning_rate2, c=grda_c, mu=grda_mu)

        self.model.compile(loss_fn=self.loss_fn, optimizer1=opt1, optimizer2=opt2, global_step=self.global_step)

    def _train_step(self, X, y):
        print("Training step...")  # 调试信息
        self.model.set_train(True)
        if isinstance(self.model.inputs, list):
            for i in range(len(self.model.inputs)):
                self.model.inputs[i].set_data(Tensor(X[i], ms.float32))
        else:
            self.model.inputs.set_data(Tensor(X, ms.float32))
        self.model.labels.set_data(Tensor(y, ms.float32))

        loss = self.model.loss(self.model.logits, self.model.labels)
        grads = self.model.optimizer1.parameters.gradients(loss)
        self.model.optimizer1.apply_gradients(grads)
        return loss

    def _eval_step(self, X, y):
        print("Evaluation step...")  # 调试信息
        self.model.set_train(False)
        if isinstance(self.model.inputs, list):
            for i in range(len(self.model.inputs)):
                self.model.inputs[i].set_data(Tensor(X[i], ms.float32))
        else:
            self.model.inputs.set_data(Tensor(X, ms.float32))
        self.model.labels.set_data(Tensor(y, ms.float32))

        logits = self.model.logits(self.model.inputs)
        loss = self.model.loss(logits, self.model.labels)
        preds = ops.sigmoid(logits)
        return loss, preds

    def fit(self):
        print("Starting training...")
        for epoch in range(self.n_epoch):
            print(f"Epoch {epoch + 1}/{self.n_epoch}")
            for step, (X, y) in enumerate(self.train_gen):
                loss = self._train_step(X, y)
                if step % 100 == 0:
                    print(f"Step {step}, Loss: {loss.asnumpy()}")
                if step >= self.train_per_epoch // self.batch_size:
                    break

            if (epoch + 1) % self.test_every_epoch == 0:
                self.evaluate()

    def evaluate(self):
        print("Evaluating...")
        preds = []
        labels = []
        for step, (X, y) in enumerate(self.test_gen):
            loss, pred = self._eval_step(X, y)
            preds.append(pred.asnumpy())
            labels.append(y)
            if step >= self.test_per_epoch // self.batch_size:
                break
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        auc = self.call_auc(y_true=labels, y_score=preds)
        loss = self.call_loss(y_true=labels, y_pred=preds)
        print(f"Evaluation loss: {loss}, AUC: {auc}")

    def save_checkpoint(self, path):
        ms.save_checkpoint(self.model, path)

    def load_checkpoint(self, path):
        ms.load_checkpoint(path, self.model)



