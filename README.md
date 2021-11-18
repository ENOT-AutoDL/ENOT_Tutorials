# ENOT_Tutorials
Here you can find tutorials for ENOT framework.  

To get started with ENOT, you first need to install it. You can find the installation instructions here:
https://enot-autodl.rtd.enot.ai/en/latest/installation_guide.html#installation

### 1. Tutorial - getting started
Describes the basic steps you need to optimize an architecture:
* Create your model with some searchable operations (we will use MobileNetV2-like architecture in this tutorial) and move this model to SearchSpace
* Pretrain search space on target task (Imagenette image classification task in this example)
* Search best architecture in search_space for your task
* Tune searched model

### 2. Tutorial - search space autogeneration
This notebook describes how to automatically generate search space from your model.

### 3. Tutorial - custom model
In this notebook we demonstrate how you can implement your own model. This is requred when your model can not be built by "block model builders" from ENOT framework.

### 4. Tutorial - using latency optimization
Here we describe the additional steps required to enable latency optimization during search phase:
* Add a custom non-searchable operation/module with latency to use with search space
* Add a custom searchable operation with latency to use with search space

### 5. Tutorial - latency calculation
This notebook describes how to calculate latency using ENOT framework.
Main chapters of this notebook:
* Initialize latency of search space (SearchSpaceModel)
* Calculate latency of arbitrary model/module

### 6. Tutorial - search with the specified latency
This notebook describes how to search only architectures with latency strictly lower than the specified value.

### 7. Tutorial - resolution search for image classification
This notebook shows how to search optimal image resolution for computer vision task.
In this tutorial we search model and resolution with best accuracy with fixed maximum latency.

### 8. Tutorial - search space autogeneration (EfficientNet-V2 S)
This notebook describes how to generate search space automatically from EfficientNet-V2 Small model.
This example uses pytorch-image-models EfficientNet-V2 implementation.

### 9. Tutorial - metric learning
This notebook shows application of neural architecture search for metric learning task using Arcface loss.


### 10. Tutorial - adding custom operations for model builder
Describes how you can implement your own operation to use with ENOT Neural Architecture Search

### Multigpu pretrain example
In this folder you can find .sh script to run multigpu pretrain with the correct torch distributed initialiization and .py script with pretrain part from getting started tutorial, with some multigpu related changes. Distributed search is not recommended as it is under development. Moreover, the search procedure is usually fast. At the tuning stage, you have a simple model without any of the ENOT specifics, so it is your responsibility to write correct distributed code. 
