# AutoFIS

Automatic Feature Interaction Selection in Factorization Models for Click-Through Rate Prediction（点击率预测的分解模型中的自动特征交互选择）

## 论文背景

点击率（CTR）预测在推荐系统中至关重要，其任务是预测用户点击推荐项目的概率。为了提高CTR模型的性能，显式的特征交互被认为是非常重要的 。传统的协同过滤推荐算法，如矩阵分解（MF）和因子分解机（FM），通过双线性学习模型来提取二阶信息 。然而，并非所有交互都有助于性能，一些基于树的方法被提出来自动寻找有用的交互 。但树模型只能探索推荐系统中多字段分类数据的所有可能特征交互的一小部分 。

## 模型介绍

AutoFIS模型的设计目标是自动选择因子分解模型中的重要特征交互。

**因子分解模型（基础模型）**：FM,DeepFM,IPNN。通过内积或神经网络等运算将不同特征的多个嵌入的相互作用建模为实数的模型。在这个模型中，所有的特征交互都以相同的贡献度传递给下一层。
**AutoFIS**：可以应用于任何分解模型的特征交互层，从而有AutoFM，AutoDeepFM，AutoIPNN...

AutoFIS通过两阶段的方法来选择重要的特征交互：

**搜索阶段**：在这一阶段，AutoFIS不再在离散的候选特征交互集合上搜索，而是通过引入架构参数将选择放宽为连续的。通过实现一个正则化的优化器，模型可以在训练过程中自动识别并移除冗余的特征交互。

**重新训练阶段**：在这一阶段，选择具有非零值的架构参数（具有特定特征交互的模型）进行重新训练，保持架构参数作为注意力单元以进一步提升模型的准确率。

[![71819063519](C:/Users/十二/AppData/Local/Temp/1718190635197.png)](https://github.com/Atehna/AutoFIS-mindspore/blob/main/imgs/AutoFIS%20structure.png)

## 开发环境

##### tensorflow版本

python 3.7，tensorflow 1.15.0

GPU: 1*Tnt004(16GB)|CPU: 8核 32GB

##### mindspore版本

使用华为ModelArts平台上的公共镜像：mindspore1.7.0-cuda10.1-py3.7-ubuntu18.04

GPU: 1*Tnt004(16GB)|CPU: 8核 32GB

## 数据集

论文中使用了三个大规模数据集来评估AutoFIS方法的效果：Avazu、Criteo这两个公共数据集和一个私有数据集。

**Avazu数据集**：包含了点击率预测任务中的广告点击日志数据。训练集:测试集=8：2

**Criteo数据集**：包含一个月的点击日志数据。训练和验证集——day 6-12，评估集——day-13

**私有数据集**：该数据集来自华为应用商店的一个游戏推荐场景，包含应用特征（如ID、类别）、用户特征（如用户的行为历史）和上下文特征。

**数据集下载链接**：https://github.com/Atomu2014/Ads-RecSys-Datasets。注意事项：数据集必须要通过tar命令解压，不要用解压工具解压。

## mindspore实现版本

### mindspore框架介绍

MindSpore 是华为公司推出的一个全栈深度学习框架，旨在支持从端到端的全场景 AI 开发。它覆盖了从芯片、芯片操作系统、到框架和应用的整个 AI 生态系统。MindSpore 拥有丰富的开源生态，支持与 TensorFlow、PyTorch 等主流深度学习框架进行模型转换和兼容，方便开发者迁移和复用现有模型。其开源社区活跃，提供了大量的示例代码和文档，帮助开发者快速上手。

**框架特点**

- **全栈优化**：MindSpore 支持从端到端（Edge-to-Cloud）的 AI 部署，包括端侧设备、边缘设备和云端数据中心，提供一体化的开发体验。
- **高性能**：MindSpore 通过软硬协同优化，充分利用硬件能力，实现高效的计算性能。它支持多种硬件平台，如华为昇腾（Ascend）AI 处理器、GPU 和 CPU。
- **自动并行**：框架内置了自动并行技术，可以根据硬件资源和模型规模自动选择最佳并行策略，提高训练和推理效率。
- **隐私保护**：MindSpore 通过创新的数据流图架构，可以在不依赖原始数据的情况下进行训练，提升数据隐私保护能力。
- **开发友好**：MindSpore 提供了简单易用的接口和丰富的库，支持灵活的网络结构定义和调试，降低开发者的学习成本和使用难度。

**适用场景**

- **端侧设备**：如智能手机、智能家居设备等，通过轻量化设计和硬件加速支持，提升端侧 AI 应用的性能和能效。
- **边缘计算**：如工业物联网、智能交通等场景，通过边缘侧推理和训练，降低数据传输延迟和带宽消耗。
- **云端数据中心**：如大规模数据训练和推理任务，通过分布式训练和大规模并行计算，实现高效的云端 AI 部署。

### 环境准备

使用华为ModelArts平台上的公共镜像：mindspore1.7.0-cuda10.1-py3.7-ubuntu18.04

GPU: 1*Tnt004(16GB)|CPU: 8核 32GB

### 模型迁移

使用工具**X2MindSpore**将基于Tensorflow的模型转成Mindspore版本。

**迁移过程**：

- 安装方法：

先装MindStudio和CANN，然后在MindStudio中配置CANN，即可使用X2MindSpore工具。

![71818048493](C:/Users/十二/AppData/Local/Temp/1718180484934.png)

![71818050752](C:/Users/十二/AppData/Local/Temp/1718180507529.png)

- 将Tensorflow的Api替换为MindSpore的API，下面是本次模型迁移过程中替换的API和Class

| TensorFlow API/Class                   | MindSporeAPI/Class             | 说明                                                         |
| -------------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| tensorflow.float32                     | mindspore.common.dtype.float32 |                                                              |
| tensorflow.python.ops.math_ops.pow     | mindspore.ops.maximum          | 对两个张量进行逐元素求幂运算。                               |
| tensorflow.python.ops.math_ops.maximum | mindspore.ops.maximum          | 计算两个张量的逐元素最大值。                                 |
| tensorflow.train.AdagradOptimizer      | mindspore.nn. optim.Adagrad    | Adagrad优化器是一种梯度优化算法。它根据历史梯度的平方和的平方根来自适应地调整每个参数的学习率。 |
| tensorflow.train.AdamOptimizer         | mindspore.nn.optim.Adam        | Adam优化器是随机梯度下降的扩展，它为每个参数计算自适应学习率，并使用梯度的一阶和二阶矩来调整学习率。 |
| tensorflow.train.MomentumOptimizer     | mindspore.nn.optim.Momentum    | Adam优化器是随机梯度下降的扩展，它为每个参数计算自适应学习率，并使用梯度的一阶和二阶矩来调整学习率。 |
| tensorflow.train.FtrlOptimizer         | mindspore.nn.optim.FTRL        | FTRL-Proximal优化器是用于大规模数据集的在线学习算法。它结合了Adagrad和FTRL的优点。 |
| tensorflow.nn.l2_loss                  | mindspore.nn.L2Loss            | 计算张量的一半L2范数，不包括平方根。                         |
| tensorflow.shape                       | mindspore.ops.Shape            | 返回张量的形状。                                             |
| tensorflow.placeholder                 | mindspore .nn.Cell             | 返回张量的形状。                                             |
| ...                                    | ...                            | ...                                                          |

### 部分代码微调

- grad_mindspore_.py中的 _create_slots函数  tensorflow.python.ops.random_ops.random_uniform

```python
import mindspore.ops as ops
import mindspore.ops.functional as F
import mindspore.nn as nn
import numpy as np
def _create_slots(self, var_list):
        seed = 123
        for v in var_list:
        # 使用mindspore中的random_uniform
            v_ini = ops.random_uniform(
                v.get_shape(), minval=-0.1, maxval=0.1, dtype=v.dtype, seed=seed) * 0
            self.accumulator = self._get_or_make_slot(v, v_ini, "accumulator", self.name)

        first_var = min(var_list, key=lambda x: x.name)
        self.l1_accum = self._create_non_slot_variable(initial_value=0., name="l1_accum", colocate_with=first_var)
        self.iter = self._create_non_slot_variable(initial_value=0., name="iter", colocate_with=first_var)
```

- ms_utils.py中的embedding_lookup函数

```python
import mindspore as ms
from mindspore import Tensor, nn, ops
import mindspore.numpy as mnp
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
```

- ms_utils.py中的get_loss函数  tensorflow.nn.weighted_cross_entropy_with_logits

```python
import mindspore.nn as nn
import mindspore.ops.operations as P
def get_loss(loss_func, pos_weight=1.0):
    loss_func = loss_func.lower()
    if loss_func == 'weight' or loss_func == 'weighted':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_func == 'sigmoid':
        return nn.BCEWithLogitsLoss()
    elif loss_func == 'softmax':
        return nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")
```

### 训练结果

以下是Avazu数据集在mindspore版本下autodeepfm模型的训练结果:

**搜索阶段**

![516d131bd0e9038918301cc93ab27](../../Temp/9516d131bd0e9038918301cc93ab27b.png)![46859bacfcf94e24b2fd325a0b2e2](../../Temp/546859bacfcf94e24b2fd325a0b2e2e.png)![3173fd45938e2aa0d5d2bec824906](../../Temp/f3173fd45938e2aa0d5d2bec8249069.png)

**重训练阶段**:将comb_mask设为[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1]得到最终训练结果

![ab1fa20719ca7f3d6491bf6f80424](../../Temp/5ab1fa20719ca7f3d6491bf6f804245.png)![e8bb2c892d1dc087f1da60e656697](../../Temp/0e8bb2c892d1dc087f1da60e656697b.png)![742a2419173c74f40075cc6a4952f](../../Temp/a742a2419173c74f40075cc6a4952f5.png)

