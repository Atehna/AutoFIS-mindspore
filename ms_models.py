from __future__ import print_function
from abc import abstractmethod
from itertools import combinations
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore.nn import TrainOneStepCell, WithLossCell
import mindspore.numpy as mnp

import __init__
from ms_utils import row_col_fetch, row_col_expand, batch_kernel_product, batch_mlp, create_placeholder, drop_out, embedding_lookup, linear, output, bin_mlp, get_variable, layer_normalization, batch_normalization, get_l2_loss, split_data_mask

dtype = __init__.config['dtype']

if dtype.lower() == 'float32' or dtype.lower() == 'float':
    dtype = ms.float32
elif dtype.lower() == 'float64':
    dtype = ms.float64

class Model(nn.Cell):
    def __init__(self):
        super(Model, self).__init__()
        self.inputs = None
        self.outputs = None
        self.logits = None
        self.labels = None
        self.learning_rate = None
        self.loss = None
        self.l2_loss = None
        self.optimizer = None
        self.grad = None

    @abstractmethod
    def compile(self, **kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__

def generate_pairs(ranges=range(1, 100), mask=None, order=2):
    res = [[] for _ in range(order)]
    for i, pair in enumerate(list(combinations(ranges, order))):
        if mask is None or mask[i] == 1:
            for j in range(order):
                res[j].append(pair[j])
    print("generated pairs", len(res[0]))
    return res

class AutoFM(Model):
    def __init__(self, init='xavier', num_inputs=None, input_dim=None, embed_size=None, l2_w=None, l2_v=None,
                 norm=False, real_inputs=None, comb_mask=None, weight_base=0.6, third_prune=False,
                 comb_mask_third=None, weight_base_third=0.6, retrain_stage=0):
        super(AutoFM, self).__init__()
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.l2_ps = l2_v
        self.third_prune = third_prune
        self.retrain_stage = retrain_stage

        self.inputs, self.labels, self.training = create_placeholder(num_inputs, ms, True)

        inputs, mask, flag, num_inputs = split_data_mask(self.inputs, num_inputs, norm=norm, real_inputs=real_inputs)

        self.xw, self.xv, b, self.xps = embedding_lookup(init=init, input_dim=input_dim, factor=embed_size, inputs=inputs,
                                                         apply_mask=flag, mask=mask, third_order=third_prune)

        l = linear(self.xw)
        self.cols, self.rows = generate_pairs(range(self.xv.shape[1]), mask=comb_mask)
        t_embedding_matrix = ops.transpose(self.xv, (1, 0, 2))
        left = ops.transpose(ops.gather(t_embedding_matrix, self.rows, 0), (1, 0, 2))
        right = ops.transpose(ops.gather(t_embedding_matrix, self.cols, 0), (1, 0, 2))
        level_2_matrix = ops.reduce_sum(left * right, -1)

        self.edge_weights = Parameter(Tensor(np.random.uniform(weight_base - 0.001, weight_base + 0.001, size=(len(self.cols),)), dtype=dtype), name='weights')
        mask = ops.expand_dims(self.edge_weights, 0)

        level_2_matrix = nn.BatchNorm1d(num_features=level_2_matrix.shape[-1], use_batch_statistics=not self.training)(level_2_matrix)
        level_2_matrix *= mask

        if third_prune:
            self.first, self.second, self.third = generate_pairs(range(self.xps.shape[1]), mask=comb_mask_third, order=3)
            t_embedding_matrix = ops.transpose(self.xps, (1, 0, 2))
            first_embed = ops.transpose(ops.gather(t_embedding_matrix, self.first, 0), (1, 0, 2))
            second_embed = ops.transpose(ops.gather(t_embedding_matrix, self.second, 0), (1, 0, 2))
            third_embed = ops.transpose(ops.gather(t_embedding_matrix, self.third, 0), (1, 0, 2))
            level_3_matrix = ops.reduce_sum(first_embed * second_embed * third_embed, -1)
            self.third_edge_weights = Parameter(Tensor(np.random.uniform(weight_base_third - 0.001, weight_base_third + 0.001, size=(len(self.first),)), dtype=dtype), name='third_weights')
            third_mask = ops.expand_dims(self.third_edge_weights, 0)
            level_3_matrix = nn.BatchNorm1d(num_features=level_3_matrix.shape[-1], use_batch_statistics=not self.training)(level_3_matrix)
            level_3_matrix *= third_mask

        fm_out = ops.reduce_sum(level_2_matrix, -1)
        if third_prune:
            fm_out2 = ops.reduce_sum(level_3_matrix, -1)
            self.logits, self.outputs = output([l, fm_out, fm_out2, b])
        else:
            self.logits, self.outputs = output([l, fm_out, b])

    def analyse_structure(self, print_full_weight=False, epoch=None):
        wts = self.edge_weights.asnumpy()
        mask = self.edge_weights.asnumpy()
        if print_full_weight:
            outline = ",".join(map(str, wts)) + "\n"
            print(f"log avg auc all weights for(epoch:{epoch})", outline)
        print("wts", wts[:10])
        print("mask", mask[:10])
        zeros_ = np.zeros_like(mask, dtype=np.float32)
        zeros_[mask == 0] = 1
        print("masked edge_num", sum(zeros_))
        if self.third_prune:
            wts = self.third_edge_weights.asnumpy()
            mask = self.third_edge_weights.asnumpy()
            if print_full_weight:
                outline = ",".join(map(str, wts)) + "\n"
                print(f"third log avg auc all third weights for(epoch:{epoch})", outline)
            print("third wts", wts[:10])
            print("third mask", mask[:10])
            zeros_ = np.zeros_like(mask, dtype=np.float32)
            zeros_[mask == 0] = 1
            print("third masked edge_num", sum(zeros_))

    def compile(self, loss=None, optimizer1=None, optimizer2=None, global_step=None, pos_weight=1.0):
        self.loss = loss(logits=self.logits, labels=self.labels, pos_weight=pos_weight)
        _loss_ = self.loss
        if self.third_prune:
            self.l2_loss = get_l2_loss([self.l2_w, self.l2_v, self.l2_ps], [self.xw, self.xv, self.xps])
        else:
            self.l2_loss = get_l2_loss([self.l2_w, self.l2_v], [self.xw, self.xv])
        if self.l2_loss is not None:
            _loss_ += self.l2_loss

        if self.retrain_stage:
            self.optimizer1 = optimizer1.minimize(_loss_, self.trainable_params())
        else:
            weight_var = [self.edge_weights]
            if self.third_prune:
                weight_var += [self.third_edge_weights]
            other_var = [param for param in self.trainable_params() if param not in weight_var]
            self.optimizer1 = optimizer1.minimize(_loss_, other_var)
            self.optimizer2 = optimizer2.minimize(_loss_, weight_var)


from mindspore import nn, Tensor, Parameter, ops
import numpy as np
import mindspore as ms

class AutoDeepFM(Model):
    def __init__(self, init='xavier', num_inputs=None, input_dim=None, embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, norm=False, real_inputs=None,
                 batch_norm=False, layer_norm=False, comb_mask=None, weight_base=0.6, third_prune=False,
                 comb_mask_third=None, weight_base_third=0.6, retrain_stage=0):
        super(AutoDeepFM, self).__init__()
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.l2_ps = l2_v
        self.layer_l2 = layer_l2
        self.retrain_stage = retrain_stage
        self.inputs, self.labels, self.training = create_placeholder(num_inputs, ms, True)
        layer_keeps = drop_out(self.training, layer_keeps)
        inputs, mask, flag, num_inputs = split_data_mask(self.inputs, num_inputs, norm=norm, real_inputs=real_inputs)

        self.xw, xv, _, self.xps = embedding_lookup(init=init, input_dim=input_dim, factor=embed_size, inputs=inputs,
                                                    apply_mask=flag, mask=mask, use_b=False, third_order=third_prune)
        self.third_prune = third_prune
        self.xv = xv
        h = ops.reshape(xv, (-1, num_inputs * embed_size))
        h, self.layer_kernels, _ = bin_mlp(init, layer_sizes, layer_acts, layer_keeps, h, num_inputs * embed_size,
                                           batch_norm=batch_norm, layer_norm=layer_norm, training=self.training)
        h = ops.squeeze(h)

        l = linear(self.xw)
        self.cols, self.rows = generate_pairs(range(self.xv.shape[1]), mask=comb_mask)
        self.cols = Tensor(self.cols, dtype=ms.int32)
        self.rows = Tensor(self.rows, dtype=ms.int32)
        t_embedding_matrix = ops.transpose(self.xv, (1, 0, 2))
        left = ops.transpose(ops.gather(t_embedding_matrix, self.rows, 0), (1, 0, 2))
        right = ops.transpose(ops.gather(t_embedding_matrix, self.cols, 0), (1, 0, 2))
        level_2_matrix = ops.reduce_sum(left * right, -1)

        self.edge_weights = Parameter(
            Tensor(np.random.uniform(weight_base - 0.001, weight_base + 0.001, size=(len(self.cols),)), dtype=ms.float32),
            name='weights')
        mask = ops.expand_dims(self.edge_weights, 0)
        level_2_matrix = nn.BatchNorm1d(num_features=level_2_matrix.shape[-1], use_batch_statistics=not self.training)(
            level_2_matrix)
        level_2_matrix *= mask

        if third_prune:
            self.first, self.second, self.third = generate_pairs(range(self.xps.shape[1]), mask=comb_mask_third,
                                                                 order=3)
            self.first = Tensor(self.first, dtype=ms.int32)
            self.second = Tensor(self.second, dtype=ms.int32)
            self.third = Tensor(self.third, dtype=ms.int32)
            t_embedding_matrix = ops.transpose(self.xps, (1, 0, 2))
            first_embed = ops.transpose(ops.gather(t_embedding_matrix, self.first, 0), (1, 0, 2))
            second_embed = ops.transpose(ops.gather(t_embedding_matrix, self.second, 0), (1, 0, 2))
            third_embed = ops.transpose(ops.gather(t_embedding_matrix, self.third, 0), (1, 0, 2))
            level_3_matrix = ops.reduce_sum(first_embed * second_embed * third_embed, -1)
            self.third_edge_weights = Parameter(
                Tensor(np.random.uniform(weight_base_third - 0.001, weight_base_third + 0.001, size=(len(self.first),)),
                       dtype=ms.float32), name='third_weights')
            third_mask = ops.expand_dims(self.third_edge_weights, 0)
            level_3_matrix = nn.BatchNorm1d(num_features=level_3_matrix.shape[-1],
                                            use_batch_statistics=not self.training)(level_3_matrix)
            level_3_matrix *= third_mask

        fm_out = ops.reduce_sum(level_2_matrix, -1)
        if third_prune:
            fm_out2 = ops.reduce_sum(level_3_matrix, -1)
            self.logits, self.outputs = output([l, fm_out, fm_out2, h])
        else:
            self.logits, self.outputs = output([l, fm_out, h])

    def analyse_structure(self, print_full_weight=False, epoch=None):
        wts = self.edge_weights.asnumpy()
        mask = self.edge_weights.asnumpy()
        if print_full_weight:
            outline = ",".join(map(str, wts)) + "\n"
            print(f"log avg auc all weights for(epoch:{epoch})", outline)
        print("wts", wts[:10])
        print("mask", mask[:10])
        zeros_ = np.zeros_like(mask, dtype=np.float32)
        zeros_[mask == 0] = 1
        print("masked edge_num", sum(zeros_))
        if self.third_prune:
            wts = self.third_edge_weights.asnumpy()
            mask = self.third_edge_weights.asnumpy()
            if print_full_weight:
                outline = ",".join(map(str, wts)) + "\n"
                print(f"third log avg auc all third weights for(epoch:{epoch})", outline)
            print("third wts", wts[:10])
            print("third mask", mask[:10])
            zeros_ = np.zeros_like(mask, dtype=np.float32)
            zeros_[mask == 0] = 1
            print("third masked edge_num", sum(zeros_))

    def compile(self, loss_fn, optimizer1, optimizer2, global_step):
        print("Compiling model...")  # 调试信息
        self.loss = loss_fn(logits=self.logits, labels=self.labels)
        _loss_ = self.loss
        if self.third_prune:
            self.l2_loss = get_l2_loss([self.l2_w, self.l2_v, self.l2_ps, self.layer_l2], [self.xw, self.xv, self.xps, self.layer_kernels])
        else:
            self.l2_loss = get_l2_loss([self.l2_w, self.l2_v, self.layer_l2], [self.xw, self.xv, self.layer_kernels])
        if self.l2_loss is not None:
            _loss_ += self.l2_loss

        loss_net = WithLossCell(self, _loss_)
        if self.retrain_stage:
            self.optimizer1 = TrainOneStepCell(loss_net, optimizer1)
        else:
            weight_var = [self.edge_weights]
            if self.third_prune:
                weight_var += [self.third_edge_weights]
            other_var = [param for param in self.trainable_params() if param not in weight_var]
            self.optimizer1 = TrainOneStepCell(loss_net, optimizer1)
            self.optimizer2 = TrainOneStepCell(loss_net, optimizer2)
        print("Model compiled.")  # 调试信息



