# ENOT_Tutorials
Here you can find tutorials for ENOT framework.

### Tutorial - getting started
Describes the basic steps you need to optimize an architecture:
* Create your model with some searchable operations (we will use MobileNetV2-like architecture in this tutorial) and move this model to SearchSpace
* Pretrain search space on target task (Imagenette image classification task in this example)
* Search best architecture in search_space for your task
* Tune searched model

### Tutorial - adding custom ops
Describes how you can implement your own operation to use with ENOT Neural Architecture Search

### Tutorial - custom model
In this notebook we demonstrate how you can implement your own model. This is requred when your model can not be built by "block model builders" from ENOT framework.

### Tutorial - using latency optimization
Here we describe the additional steps required to enable latency optimization during search phase:
* Add a custom non-searchable operation/module with latency to use with search space
* Add a custom searchable operation with latency to use with search space

### Tutorial - metric learning
This notebook shows application of neural architecture search for metric learning task using Arcface loss.
