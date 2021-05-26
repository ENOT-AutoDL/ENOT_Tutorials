# ENOT_Tutorials
Here you can find tutorials for ENOT framework.

### Tutorial - getting started
Describes the basic steps you need to optimize an architecture:
* Create your model with some searchable operations (we will use MobileNetV2-like architecture in this tutorial) and move this model to SearchSpace
* Pretrain search space on target task (Imagenette image classification task in this example)
* Search best architecture in search_space for your task
* Tune searched model

### Tutorial - adding custom operations for model builder
Describes how you can implement your own operation to use with ENOT Neural Architecture Search

### Tutorial - custom model
In this notebook we demonstrate how you can implement your own model. This is requred when your model can not be built by "block model builders" from ENOT framework.

### Tutorial - using latency optimization
Here we describe the additional steps required to enable latency optimization during search phase:
* Add a custom non-searchable operation/module with latency to use with search space
* Add a custom searchable operation with latency to use with search space

### Tutorial - latency calculation
This notebook describes how to calculate latency using ENOT framework.
Main chapters of this notebook:
* Initialize latency of search space (SearchSpaceModel)
* Calculate latency of arbitrary model/module

### Tutorial - metric learning
This notebook shows application of neural architecture search for metric learning task using Arcface loss.

### Tutorial - resolution search for image classification
This notebook shows how to search optimal image resolution for computer vision task.
In this tutorial we search model and resolution with best accuracy with fixed maximum latency.

### Tutorial - search space autogeneration
This notebook describes how to generate search space automaticaly from your model.

### Multigpu pretrain
In this folder you can find .sh script to run multigpu pretrain with the correct torch distributed initialiization and .py script with pretrain part from getting started tutorial, with some multigpu related changes. Distributed search is not recommended as it is under development. Moreover, the search procedure is usually fast. At the tuning stage, you have a simple model without any of the ENOT specifics, so it is your responsibility to write correct distributed code. 
