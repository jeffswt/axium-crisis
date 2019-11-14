
# Axium Crisis

Save entire experiment process while training deep learning models with
pytorch. This includes:

  * random number generator states (random, numpy, pytorch)
  * how is the dataset loaded
  * model parameters at significant epochs
  * data written to tensorboard

You may replay the experiment and fast-forward to an epoch at any time.

The project is composed mainly of 6 units:

  * InoRI: random number generator manager
  * HikarI: experiment history writer
  * TairitsU: history reader and replay manager
  * ShirabE: cache-based high performance dataset
  * KuroyuKI: retroactive model trainer
  * train: a useful function that connects them all

By default, this project is set to optimize with L1 loss, evaluate PSNR and
gradient descent with Adam (b1 = 0.999, b2 = 0.9).

## Usage

When you used to write the code like this:

```python
model = WaifuModel((48, 48), channels=16, upsample=4)
```

In Axium Crisis you write the training code like this:

```python
import axiumcrisis

model_args = (WaifuModel, [(48, 48)], dict(channels=16, upsample=4))
axiumcrisis.train(model_args, './dataset/', (48, 48), 4,
                  100, history='./run-10/', force_seed='pure-memory',
                  tb_log_path='../tensorboard/logs/10/')
```

Or recalling history:

```python
axiumcrisis.train(model_args, './dataset/', (48, 48), 4,
                  200, history='./run-11/', recall='./run-10/',
                  start_from_epoch=101, tb_log_path='../tensorboard/logs/11/')
```

Consult the code documentation for details on further usage.

## Installation

Copy the code to your working directory and it should work.

Remember to install requirements with:

```sh
pip install -r requirements.txt
```

You might have to install pytorch manually from the official website.

## Contribution

We are open to issues and pull requests for new features or bug fixes.
