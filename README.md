---
license: apache-2.0
metrics:
- accuracy
---

# image_classification
(this model was not trained using Trainer API)
This model is a fine-tuned version of [EfficientNetB7](https://github.com/lukemelas/EfficientNet-PyTorch) on the [Tyre-Quality-Classification](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification/code) dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2341
- Accuracy: 91.9355%

## Intended uses & limitations

Can be used for quality control to identify the condition of tyres

## Training and evaluation data

Data can be seen at [Weights and Biases](https://wandb.ai/faldeus0092/efficientnetb7_tyrequality_classifier/runs/1z5mnxps/overview?workspace=user-faldeus0092)

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 16
- eval_batch_size: 16
- train_set: 1434
- test_set: 372
- optimizer: SGD with momentum = 0.9
- num_epochs: 5

### Training results

See at [Weights and Biases](https://wandb.ai/faldeus0092/efficientnetb7_tyrequality_classifier/runs/1z5mnxps/overview?workspace=user-faldeus0092)
