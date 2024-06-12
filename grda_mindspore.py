import numpy as np
from mindspore import nn, ops, Tensor, Parameter
import mindspore.numpy as mnp

class GRDA(nn.Optimizer):
    """Optimizer that implements the GRDA algorithm."""

    def __init__(self, params, learning_rate=0.005, c=0.005, mu=0.7):
        """Construct a new GRDA optimizer.
        Args:
            learning_rate: A Tensor or a floating point value. The learning rate.
            c: A float value or a constant float tensor. Turn on/off the l1 penalty and initial penalty.
            mu: A float value or a constant float tensor. Time expansion of l1 penalty.
        """
        super(GRDA, self).__init__(learning_rate, params)
        self.learning_rate = learning_rate
        self.c = c
        self.mu = mu
        self.l1_accum = Parameter(Tensor(0.0, dtype=mnp.float32), name="l1_accum")
        self.iter = Parameter(Tensor(0.0, dtype=mnp.float32), name="iter")
        self._accumulators = {}
        for param in self.parameters:
            self._accumulators[param.name] = Parameter(Tensor(np.random.uniform(-0.1, 0.1, param.shape), dtype=param.dtype), name=param.name + "_accumulator")

    def construct(self, grads):
        lr = Tensor(self.learning_rate, dtype=mnp.float32)
        c = Tensor(self.c, dtype=mnp.float32)
        mu = Tensor(self.mu, dtype=mnp.float32)
        iter_ = self.iter
        l1_accum = self.l1_accum

        l1_diff = c * ops.pow(lr, (0.5 + mu)) * ops.pow(iter_ + 1.0, mu) - c * ops.pow(lr, (0.5 + mu)) * ops.pow(iter_ + 0.0, mu)
        l1_accum = l1_diff + l1_accum
        first_iter = ops.maximum(1 - iter_, 0)

        updates = []
        for param, grad in zip(self.parameters, grads):
            v = self._accumulators[param.name]
            v_t = v + first_iter * param - lr * grad
            updates.append(ops.assign(param, ops.sign(v_t) * ops.maximum(ops.abs(v_t) - l1_accum, 0)))
            updates.append(ops.assign(v, v_t))

        iter_update = ops.assign_add(self.iter, 1.0)
        l1_accum_update = ops.assign(self.l1_accum, l1_accum)

        return ops.depend(True, [iter_update, l1_accum_update] + updates)

    def apply_gradients(self, grads):
        return self.construct(grads)
