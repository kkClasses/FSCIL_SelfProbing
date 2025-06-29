        title={Few-Shot Class-Incremental Learning via Training-Free Prototype Calibration},
# Few Shot Class Incremental Learning with Self Probing (SP-FSCIL)

##Citation

## Abstract
Few-Shot Class-Incremental Learning (FSCIL) presents sig-
nificant challenges, notably catastrophic forgetting and overfitting, due
to the absence of prior training data and the scarcity of samples for
newly introduced classes. Although various approaches have been pro-
posed to tackle these challenges, this study highlights a key limitation
in terms of the final-layer embeddings of a pre-trained feature extractor
being overly task-specific, leading to sparse and less expressive represen-
tations for new classes. However, some intermediate layers encode more
transferable features, which can help reduce this sparsity.
Building on this insight, the paper introduces a novel self-probing (SP)
mechanism that employs meta-training to leverage transferable features
from an inner layer. This approach aims to enrich the final-layer embed-
dings for new classes. The meta-training process incorporates a sparse
loss function designed to align the feature distributions between the self-
probed layer and the final layer, thereby enhancing the representational
quality for novel categories.
Evaluations of the proposed approach on benchmark datasets; miniIma-
geNet, CIFAR100, and CUB200 demonstrate that the proposed method
consistently outperforms existing state-of-the-art (SOTA) approaches.
Keywords: Few shot class incremental learning · Self Probing · Sparsity
in embedding.

## Results
<img src='imgs/results.png' width='900' height='700'>

Please refer to the paper...

## Prerequisites

The following packages are required to run the scripts:

- [PyTorch-1.4 and torchvision](https://pytorch.org)

- tqdm

## Dataset
We provide the source code on three benchmark datasets, i.e., CIFAR100, CUB200 and miniImageNet. 
Please follow the guidelines in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare them.


## Code Structures and details
There are four parts in the code.
 - `models`: It contains the backbone network and training protocols for the experiment.
 - `data`: The splits for the data sets.
 - `dataSet`: The data sets.
 - `dataloader`: Dataloader of different datasets.

## Training scripts

Please see `scripts` folder.


## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

- [CEC](https://github.com/icoz69/CEC-CVPR2021)
- [TEEN](https://github.com/wangkiw/TEEN)



## Contact 
If there are any questions, please feel free to contact with the author: K K Singh (krishnasingh@rguktn.ac.in)