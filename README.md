# Learning Generalizable Dexterous Manipulation from Human Grasp Affordance

[[Project Page]](https://kristery.github.io/ILAD/) [[Paper]](https://arxiv.org/abs/2204.02320)
-----

[Learning Generalizable Dexterous Manipulation from Human Grasp Affordance](https://kristery.github.io/ILAD/),Yueh-Hua Wu*, 
Jiashun Wang*, Xiaolong Wang, ECCV 2022.

ILAD is a novel pipeline for learning dexterous manipulation from human affordance.
This repo contains the **imitation learning** code for ILAD.

![DexMV Teaser](docs/teaser.png)

## Bibtex

```
@misc{wu2022ILAD,
  title={Learning Generalizable Dexterous Manipulation from Human Grasp Affordance},
  author={Wu, Yueh-Hua and Wang, Jiashun and Wang, Xiaolong},
  year={2022},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
  }
```

## Installation

1. Install the MuJoCo. It is recommended to use conda to manage python package:

Install MuJoCo from: http://www.mujoco.org/ and put your MuJoCo licence to your install directory. If you already have
MuJoCo on your computer, please skip this step.

2. Install Python dependencies Create a conda env with all the Python dependencies.

```bash
# Download the code from this repo for the simulated environment, retargeting and examples
git clone https://github.com/kristery/Learning-Generalizable-Dexterous-Manipulation-from-Human-Grasp-Affordance.git
cd Learning-Generalizable-Dexterous-Manipulation-from-Human-Grasp-Affordance

# The provoided package version in the yml is our testing environment, you do not need to follow the version of each python package precisely to run this code.
conda env create -f environment.yml 
conda activate ilad
pip install -e mj_envs/
pip install -e mjrl/
python setup.py develop

cd hand_imitation
python setup.py develop
cd ..
```

3. The file structure is listed as follows:

`exp_cfg/`: training configurations for ILAD, DAPG, and RL

`mjrl/`: Implementation of algorithms

`dapg/`: Demonstrations for training ILAD and DAPG

`hand_imitation`: training environments

`tools/main.py`: main script for training



## Training

### Download processed demonstrations

### Training config

We use a config system to specify the training parameters, including task, object, imitation learning algorithm or RL
only. For convenience, we have provided several config files for you in `exp_cfg/`.

### Training

For example, if you want to train relocate mug task using ILAD with demonstrations:

```bash
python tools/main.py --cfg exp_cfg/shapenet_relocate_mug_ilad_cfg.yaml
```

Similarly, there are several config files for other tasks and algorithms in the `exp_cfg` directory you can use
for training.

## Acknowledge

Some file in this repository is modified based on the code
from [soil](https://people.eecs.berkeley.edu/~ilija/soil/),
[dapg](https://github.com/aravindr93/hand_dapg), [dexmv-sim](https://github.com/yzqin/dexmv-sim). We gratefully thank the authors of these amazing projects.

