import os
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops
import mindspore.numpy as mnp
from mindspore.common.initializer import XavierUniform, Normal, TruncatedNormal, Uniform, Zero, One

import __init__

dtype = ms.float32 if __init__.config['dtype'] == 'float32' else ms.float64
minval = __init__.config['minval']
maxval = __init__.config['maxval']
mean = __init__.config['mean']
stddev = __init__.config['stddev']


def get_variable(init_type='xavier', shape=None, name=None, minval=minval, maxval=maxval, mean=mean,
                 stddev=stddev, dtype=dtype):
    shape = tuple(shape)  # Convert shape to tuple
    dtype_map = {
        ms.float32: np.float32,
        ms.float64: np.float64,
        ms.int32: np.int32,
        ms.int64: np.int64,
        ms.bool_: np.bool_,
    }
    np_dtype = dtype_map.get(dtype, np.float32)  # Default to np.float32 if dtype not found

    if isinstance(init_type, str):
        init_type = init_type.lower()
    if init_type == 'tnormal':
        data = np.random.normal(loc=mean, scale=stddev, size=shape).astype(np_dtype)
    elif init_type == 'uniform':
        data = np.random.uniform(low=minval, high=maxval, size=shape).astype(np_dtype)
    elif init_type == 'normal':
        data = np.random.normal(loc=mean, scale=stddev, size=shape).astype(np_dtype)
    elif init_type in ['xavier', 'xavier_out', 'xavier_in']:
        if len(shape) < 2:
            data = np.random.normal(loc=mean, scale=stddev, size=shape).astype(np_dtype)
        else:
            fan_in, fan_out = shape[0], shape[1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            data = np.random.uniform(-limit, limit, size=shape).astype(np_dtype)
    elif init_type == 'zero':
        data = np.zeros(shape, dtype=np_dtype)
    elif init_type == 'one':
        data = np.ones(shape, dtype=np_dtype)
    elif init_type == 'identity' and len(shape) == 2 and shape[0] == shape[1]:
        data = np.eye(shape[0], dtype=np_dtype)
    elif isinstance(init_type, (int, float)):
        data = np.full(shape, init_type, dtype=np_dtype)
    else:
        raise ValueError(f"Unsupported init_type: {init_type}")

    return ms.Parameter(Tensor(data), name=name)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * ops.where(x >= 0.0, x, alpha * ops.elu(x))

def activate(weights, act_type):
    if isinstance(act_type, str):
        act_type = act_type.lower()
    if act_type == 'sigmoid':
        return ops.sigmoid(weights)
    elif act_type == 'softmax':
        return ops.softmax(weights)
    elif act_type == 'relu':
        return ops.relu(weights)
    elif act_type == 'tanh':
        return ops.tanh(weights)
    elif act_type == 'elu':
        return ops.elu(weights)
    elif act_type == 'selu':
        return selu(weights)
    elif act_type == 'none':
        return weights
    else:
        return weights

def get_optimizer(opt_algo):
    opt_algo = opt_algo.lower()
    if opt_algo == 'adadelta':
        return nn.Adadelta
    elif opt_algo == 'adagrad':
        return nn.Adagrad
    elif opt_algo == 'adam':
        return nn.Adam
    elif opt_algo == 'momentum':
        return nn.Momentum
    elif opt_algo == 'ftrl':
        return nn.FTRL
    elif opt_algo == 'sgd':
        return nn.SGD
    elif opt_algo == 'rmsprop':
        return nn.RMSProp
    else:
        return nn.SGD

def get_loss(loss_type):
    if loss_type == 'weight':
        return WeightedSoftmaxCrossEntropyWithLogits
    elif loss_type == 'mse':
        return nn.MSELoss
    elif loss_type == 'mae':
        return nn.L1Loss
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


class WeightedSoftmaxCrossEntropyWithLogits(nn.Cell):
    def __init__(self, pos_weight=1.0):
        super(WeightedSoftmaxCrossEntropyWithLogits, self).__init__()
        self.pos_weight = pos_weight
        self.loss = nn.SoftmaxCrossEntropyWithLogits()

    def construct(self, logits, labels):
        logits = ops.reshape(logits, (-1, logits.shape[-1]))  # 确保logits的形状为2
        labels = ops.cast(labels, ms.int32)
        labels_one_hot = ops.one_hot(labels, logits.shape[-1], ops.scalar_to_tensor(1.0), ops.scalar_to_tensor(0.0))
        weights = (labels_one_hot * (self.pos_weight - 1.0) + 1.0)
        unweighted_losses = self.loss(logits, labels_one_hot)
        weighted_losses = unweighted_losses * weights
        return weighted_losses.mean()





def check(x):
    try:
        return x is not None and x is not False and float(x) > 0
    except TypeError:
        return True

def get_l2_loss(params, variables):
    _loss = None
    for p, v in zip(params, variables):
        if not isinstance(p, list):
            if check(p):
                if isinstance(v, list):
                    for _v in v:
                        if _loss is None:
                            _loss = p * ops.reduce_sum(ops.square(_v))
                        else:
                            _loss += p * ops.reduce_sum(ops.square(_v))
                else:
                    if _loss is None:
                        _loss = p * ops.reduce_sum(ops.square(v))
                    else:
                        _loss += p * ops.reduce_sum(ops.square(v))
        else:
            for _lp, _lv in zip(p, v):
                if _loss is None:
                    _loss = _lp * ops.reduce_sum(ops.square(_lv))
                else:
                    _loss += _lp * ops.reduce_sum(ops.square(_lv))
    return _loss

def normalize(norm, x, scale):
    if norm:
        return x * scale
    else:
        return x

def mul_noise(noisy, x, training=None):
    if check(noisy) and training is not None:
        noise = Tensor(np.random.normal(loc=1.0, scale=noisy, size=x.shape), dtype=x.dtype)
        return ops.select(training, x * noise, x)
    else:
        return x

def add_noise(noisy, x, training):
    if check(noisy):
        noise = Tensor(np.random.normal(loc=0, scale=noisy, size=x.shape), dtype=x.dtype)
        return ops.select(training, x + noise, x)
    else:
        return x

def create_placeholder(num_inputs, dtype=dtype, training=False):
    inputs = Tensor(np.zeros((1, num_inputs)), dtype=ms.int32)
    labels = Tensor(np.zeros((1,)), dtype=ms.float32)
    if check(training):
        training = Tensor(False, dtype=ms.bool_)
    return inputs, labels, training




def split_data_mask(inputs, num_inputs, norm=False, real_inputs=None, num_cat=None):
    if not check(real_inputs):
        mask = np.sqrt(1. / num_inputs) if check(norm) else 1
        flag = norm
    else:
        inputs, mask = inputs[:, :real_inputs], inputs[:, real_inputs:]
        mask = mnp.array(mask, dtype=ms.float32)
        if check(norm):
            mask /= np.sqrt(num_cat + 1)
            mask_cat, mask_mul = mask[:, :num_cat], mask[:, num_cat:]
            sum_mul = ops.reduce_sum(mask_mul, 1, keepdims=True)
            sum_mul = ops.maximum(sum_mul, ops.ones_like(sum_mul))
            mask_mul /= ops.sqrt(sum_mul)
            mask = ops.concat([mask_cat, mask_mul], 1)
        flag = True
        num_inputs = real_inputs
    return inputs, mask, flag, num_inputs

def drop_out(training, keep_probs):
    keep_probs = [float(kp) for kp in keep_probs]  # Ensure keep_probs are floats
    keep_probs = Tensor(keep_probs, dtype=dtype)
    keep_probs = ops.select(training, keep_probs, ops.ones_like(keep_probs))
    return keep_probs



def embedding_lookup(init, input_dim, factor, inputs, apply_mask=False, mask=None,
                     use_w=True, use_v=True, use_b=True, fm_path=None, fm_step=None, third_order=False, order=None,
                     embedsize=None):
    xw, xv, b, xps = None, None, None, None
    if fm_path is not None and fm_step is not None:
        fm_dict = load_fm(fm_path, fm_step)
        if use_w:
            w = ms.Parameter(Tensor(fm_dict['w'], dtype=dtype), name='w')
            xw = ops.gather(w, inputs, axis=0)
            if apply_mask:
                xw = xw * mask
        if use_v:
            v = ms.Parameter(Tensor(fm_dict['v'], dtype=dtype), name='v')
            xv = ops.gather(v, inputs, axis=0)
            if apply_mask:
                xv = xv * mnp.expand_dims(mask, 2)
        if use_b:
            b = ms.Parameter(Tensor(fm_dict['b'], dtype=dtype), name='b')
    else:
        if use_w:
            w = get_variable(init, name='w', shape=[input_dim])
            xw = ops.gather(w, inputs, axis=0)
            if apply_mask:
                xw = xw * mask
        if use_v:
            v = get_variable(init_type=init, name='v', shape=[input_dim, factor])
            xv = ops.gather(v, inputs, axis=0)
            if apply_mask:
                xv = xv * mnp.expand_dims(mask, 2)
        if third_order:
            third_v = get_variable(init_type=init, name='third_v', shape=[input_dim, factor])
            xps = ops.gather(third_v, inputs, axis=0)
            if apply_mask:
                xps = xps * mnp.expand_dims(mask, 2)
        if use_b:
            b = get_variable('zero', name='b', shape=[1])
    return xw, xv, b, xps

def linear(xw):
    return ops.reduce_sum(xw, 1)


def output(x):
    logits = sum(x) if isinstance(x, list) else x
    outputs = ops.sigmoid(logits)
    return logits, outputs

def row_col_fetch(xv_embed, num_inputs):
    rows = []
    cols = []
    for i in range(num_inputs - 1):
        for j in range(i + 1, num_inputs):
            rows.append([i, j - 1])
            cols.append([j, i])
    xv_p = mnp.transpose(ops.gather_nd(mnp.transpose(xv_embed, [1, 2, 0, 3]), rows), [1, 0, 2])
    xv_q = mnp.transpose(ops.gather_nd(mnp.transpose(xv_embed, [1, 2, 0, 3]), cols), [1, 0, 2])
    return xv_p, xv_q

def row_col_expand(xv_embed, num_inputs):
    rows = []
    cols = []
    for i in range(num_inputs - 1):
        for j in range(i + 1, num_inputs):
            rows.append(i)
            cols.append(j)
    xv_p = mnp.transpose(ops.gather(mnp.transpose(xv_embed, [1, 0, 2]), rows, axis=0), [1, 0, 2])
    xv_q = mnp.transpose(ops.gather(mnp.transpose(xv_embed, [1, 0, 2]), cols, axis=0), [1, 0, 2])
    return xv_p, xv_q

def batch_kernel_product(xv_p, xv_q, kernel=None, add_bias=True, factor=None, num_pairs=None, reduce_sum=True, mask=None):
    if kernel is None:
        maxval = np.sqrt(3. / factor)
        minval = -maxval
        kernel = get_variable('uniform', name='kernel', shape=[factor, num_pairs, factor], minval=minval, maxval=maxval)
    if add_bias:
        bias = get_variable(0, name='bias', shape=[num_pairs])
    else:
        bias = None
    xv_p = mnp.expand_dims(xv_p, 1)
    prods = ops.reduce_sum(ops.multiply(mnp.transpose(ops.reduce_sum(ops.multiply(xv_p, kernel), -1), [0, 2, 1]), xv_q), -1)
    if add_bias:
        prods += bias
    if reduce_sum:
        prods = ops.reduce_sum(prods, 1)
    return prods, kernel, bias

def batch_mlp(h, node_in, num_pairs, init, net_sizes, net_acts, net_keeps, add_bias=True,
              reduce_sum=True, layer_norm=False, batch_norm=False, apply_mask=False, mask=None):
    h = mnp.transpose(h, [1, 0, 2])
    if apply_mask and not isinstance(mask, np.float64):
        mask = mnp.expand_dims(mnp.transpose(mask), 2)
    net_kernels = []
    net_biases = []
    for i in range(len(net_sizes)):
        _w = get_variable(init, name='w_%d' % i, shape=[num_pairs, node_in, net_sizes[i]])
        _wx = ops.matmul(h, _w)
        net_kernels.append(_w)
        if layer_norm:
            _wx = layer_normalization(_wx, reduce_dim=[0, 2], out_dim=[num_pairs, 1, net_sizes[i]], bias=False)
        elif batch_norm:
            _wx = batch_normalization(_wx, reduce_dim=[0, 1], out_dim=[num_pairs, 1, net_sizes[i]], bias=False)
        if add_bias:
            _b = get_variable(0, name='b_%d' % i, shape=[num_pairs, 1, net_sizes[i]])
            _wx += _b
            net_biases.append(_b)
        h = nn.Dropout(1 - net_keeps[i])(activate(_wx, net_acts[i]))
        node_in = net_sizes[i]
        if apply_mask and not isinstance(mask, np.float64):
            h = h * mask
    h = mnp.transpose(h, [1, 0, 2])
    if reduce_sum:
        h = ops.reduce_sum(h, 1, keepdims=False)
    return h, net_kernels, net_biases

def batch_normalization(x, reduce_dim=0, out_dim=None, scale=None, bias=None):
    reduce_dim = [reduce_dim] if isinstance(reduce_dim, int) else reduce_dim
    out_dim = [out_dim] if isinstance(out_dim, int) else out_dim
    batch_mean, batch_var = ops.moments(x, reduce_dim, keepdims=True)
    x = (x - batch_mean) / ops.sqrt(batch_var)
    if scale is not False:
        scale = scale if scale is not None else ms.Parameter(Tensor(np.ones(out_dim), dtype=dtype), name='g')
    if bias is not False:
        bias = bias if bias is not None else ms.Parameter(Tensor(np.zeros(out_dim), dtype=dtype), name='b')
    if scale is not False and bias is not False:
        return x * scale + bias
    elif scale is not False:
        return x * scale
    elif bias is not False:
        return x + bias
    else:
        return x

def layer_normalization(x, reduce_dim=1, out_dim=None, scale=None, bias=None):
    reduce_dim = [reduce_dim] if isinstance(reduce_dim, int) else reduce_dim
    out_dim = [out_dim] if isinstance(out_dim, int) else out_dim
    layer_mean, layer_var = ops.moments(x, reduce_dim, keepdims=True)
    x = (x - layer_mean) / ops.sqrt(layer_var)
    if scale is not False:
        scale = scale if scale is not None else ms.Parameter(Tensor(np.ones(out_dim), dtype=dtype), name='g')
    if bias is not False:
        bias = bias if bias is not None else ms.Parameter(Tensor(np.zeros(out_dim), dtype=dtype), name='b')
    if scale is not False and bias is not False:
        return x * scale + bias
    elif scale is not False:
        return x * scale
    elif bias is not False:
        return x + bias
    else:
        return x

def bin_mlp(init, layer_sizes, layer_acts, layer_keeps, h, node_in, batch_norm=False, layer_norm=False, training=True,
            res_conn=False):
    layer_kernels = []
    layer_biases = []
    x_prev = None
    for i in range(len(layer_sizes)):
        wi = get_variable(init, name='w_%d' % i, shape=[node_in, layer_sizes[i]])
        bi = get_variable(0, name='b_%d' % i, shape=[layer_sizes[i]])
        h = ops.matmul(h, wi)
        if i < len(layer_sizes) - 1:
            if batch_norm:
                h = nn.BatchNorm1d(layer_sizes[i])(h)
            elif layer_norm:
                h = layer_normalization(h, out_dim=layer_sizes[i], bias=False)
        h = h + bi
        if res_conn and x_prev is not None and layer_sizes[i-1] == layer_sizes[i]:
            h += x_prev
            x_prev = h
        h = nn.Dropout(p=float(1 - layer_keeps[i]))(activate(h, layer_acts[i]))
        node_in = layer_sizes[i]
        layer_kernels.append(wi)
        layer_biases.append(bi)
    return h, layer_kernels, layer_biases



def load_fm(fm_path, fm_step, fm_data):
    fm_abs_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'log', fm_data, 'FM', fm_path, 'checkpoints', 'model.ckpt-%d' % fm_step)
    fm_dict = ms.load_checkpoint(fm_abs_path)
    return {'w': fm_dict['embedding/w'], 'v': fm_dict['embedding/v'], 'b': fm_dict['embedding/b']}
