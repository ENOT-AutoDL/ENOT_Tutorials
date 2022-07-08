# ENOT_Tutorials
## Basic examples
Here you can find tutorials for ENOT framework.

To get started with ENOT, you first need to install it. You can find the
installation instructions here:
https://enot-autodl.rtd.enot.ai/en/latest/installation_guide.html#installation

### 1. Tutorial - automatic quantization for enot-lite
This notebook shows how to apply enot-autodl framework for
automatic quantization to create quantized model for enot-lite framework.

### 2. Tutorial - automatic pruning
This notebook shows how to apply enot-autodl framework for
automatic network pruning and fine-tuning.

### 3. Tutorial - automatic pruning (manual)
This notebook shows how to apply enot-autodl framework for
automatic network pruning and fine-tuning. Gradient accumulation is performed
manually.

### 4. Tutorial - Ultralytics YOLO-v5 quantization
This notebook shows how to apply enot-autodl framework for
automatic network quantization of Ultralytics YOLO-v5.


### Multigpu pretrain example
In this folder you can find
[.sh script](multigpu_pretrain/run_multigpu_pretrain.sh) for running multi-gpu
pretrain. You can change its configuration to run on single GPU, on multiple
GPU within a single node, or on multiple compute nodes with multiple GPUs.

The second file in this folder is a
[.py script](multigpu_pretrain/multigpu_pretrain.py) which is launched by .sh
script. This script uses functions from other tutorials, but it is adapted to run
in distributed manner. This script should be viewed as a reference point for
user-defined distributed pretrain scripts.

Distributed search is not recommended as it is under development. Moreover, the
search procedure is usually relatively fast. At the tuning stage, you will have
a regular model without any of the ENOT specifics, so it is your responsibility
to write correct distributed code (probably by wrapping found model with
[DistributedDataParallel module](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)).


## Neural Architecture Search examples
In the "advanced" folder you can find tutorials for NAS procedure.
 
### 1. Tutorial - getting started
Describes the basic steps you need to optimize an architecture:
* Create your model and move it into SearchSpace;
* Pretrain search space on target task (Imagenette image classification in this
  example);
* Search the best architecture for your task in search space;
* Tune this model.

### 2. Tutorial - search space autogeneration
This notebook describes how to automatically generate search space from your
model.

### 3. Tutorial - custom model
This notebook describes the ways to implement your own model.

Typical use cases:
* Creating models which can not be built by "block models builders" from ENOT
  framework;
* You already have your model, and you don't want to rewrite your code.

### 4. Tutorial - using latency optimization
This notebook describes the additional steps required to enable latency
optimization for custom models.

### 5. Tutorial - latency calculation
This notebook describes how to calculate latency using ENOT framework.

Main chapters of this notebook:
* Initialize latency of search space (SearchSpaceModel);
* Calculate latency of an arbitrary model/module.

### 6. Tutorial - search with the specified latency
This notebook describes how to search only architectures with latency strictly
lower than the specified value.

### 7. Tutorial - resolution search for image classification
This notebook shows how to search optimal image resolution for computer vision
task. In this tutorial, we search model and resolution with the best accuracy
and fixed maximum latency.

### 8. Tutorial - search space autogeneration (EfficientNet-V2 S)
This notebook describes how to generate search space automatically from
EfficientNet-V2 Small model. This example uses
[pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
EfficientNet-V2 implementation.

### 9. Tutorial - metric learning
This notebook shows an advanced example usage of ENOT framework. We will show
how to use angular margin loss for metric learning task.

