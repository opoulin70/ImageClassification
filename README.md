# **Image Classification Pipeline**
*A modular and extensible image classification pipeline built using PyTorch, designed for training, evaluating, and optimizing deep learning models.*


## **Project Overview**
This project is a deep learning pipeline for image classification, currently configured for *CIFAR-10* but extendable to other datasets. It follows modern industry practices offerering support for, among other things: 
- Stratified data splitting to maintain class distributions
- Configurable training, validation, and testing stages
- A modular architecture for extensibility and scalability

It is designed to be easily extendable, allowing integration of custom datasets, model architectures and training strategies.

The goal is to progressively extend the pipeline so that it supports various machine learning tasks beyond image classification, making it a versatile framework for experimentation and deployment.

---

## **Current Features**

### **Key Features**
- **End-to-End ML Pipeline**: Automates the entire training workflow from data loading to evaluation.
- **Flexible Configuration**: Adjustable parameters for model selection, batch size, learning rate, optimizers and more.
- **Modular Stage System**: Easily add or remove stages to customize the pipeline.
- **Scalable Design**: Supports integrating new datasets and architectures with minimal changes.

### **Technical Features**
- **Stratified Data Splitting**: Ensures class distribution remains balanced across train and validation sets.
- **Custom DataLoader**: Configurable batch sizes, shuffle options, and multi-threaded loading.
- **Logging & Metrics Tracking**: Records loss, accuracy, and processing time per epoch.
- **Visualization Support**: Generates loss/accuracy curves for better insights.
- **Device Compatibility**: Seamlessly runs on CPU or GPU (supports CUDA-enabled GPUs).

---

## **Pipeline Overview**

The pipeline is built using the `PipelineOrchestrator`, which manages all ML workflow stages.

### 1 - Configuration : `config.yaml` 

All hyperparameters and settings are defined, by default, in `config/config.yaml`. Users can modify this file to change, among other things, dataset sources, model architectures and training parameters.

Exemple :
```Yaml
# General Settings
batch_size: 128
device: "cuda"
num_epochs: 10
use_validation: True

# DataLoader Settings
data_augmentation: False
dataset: "cifar10"
dataset_directory: "./data"
num_workers: 2
shuffle_train: True

# Model Settings
model_name: "resnet18"
pretrained: true

# Optimizer Settings
learning_rate: 0.001
momentum: 0.9
optimizer: "sgd"

# Training Stage Settings
early_stopping: false
early_stopping_patience: 5
loss_function: "cross_entropy"
```

### 2 - Pipeline Execution

The `PipelineOrchestrator` executes all pipeline stages in a sequential order. Users can modify the order or add new stages based on their needs.

Exemple usage :
```Python
def main():
    # Load the configuration from a YAML file.
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    config = Config.from_yaml(config_path)

    # Create the orchestrator and add the stages.
    orchestrator = PipelineOrchestrator()

    orchestrator.add_stage(DataLoaderStage())
    orchestrator.add_stage(ModelLoaderStage())
    orchestrator.add_stage(TrainingStage())
    orchestrator.add_stage(EvaluationStage())
    orchestrator.add_stage(VisualizationStage())
    
    # Execute the pipeline.
    orchestrator.run(config)
```

---

## **Pipeline Stages**

Each stage is implemented as a separate module, making it easy to extend or modify.

- **1 - DataLoaderStage**: Load the dataset and split into train/test loaders.
- *2 - AugmentationStage (planned)*: Apply augmentations to training data.
- **3 - ModelLoaderStage**: Loads and initializes the selected model.
- **4 - TrainingStage**: Train the model while saving training metrics.
- **5 - EvaluationStage**: Compute validation/test metrics.
- **6 - VisualizationStage**: Plot loss/accuracy curves.
- *7 - ModelStorageStage (planned)*: Save the trained model for future use.
- *8 - PredictionStage  (planned)*: Runs inference on new/unseen data.

---

## Installation

### **Prerequisites**  
- Python 3.8+  
- PyTorch and torchvision
- scikit-learn
