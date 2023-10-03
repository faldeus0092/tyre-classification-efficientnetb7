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

### Example usage
```py
from efficientnet_pytorch import EfficientNet
import torch
import torchvision.transforms as transforms

model = EfficientNet.from_name('efficientnet-b7')
model._fc= torch.nn.Linear(in_features=model._fc.in_features, out_features=len(annotations_map), bias=True)
model.load_state_dict(torch.load('/content/efficientnetb7_tyrequality_classifier.pth'))

model.eval()
img = Image.open('/content/defective-tires-cause-accidents-min.jpg')
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
])
input_data = test_transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_data)

_, predicted_class = torch.max(output, 1)

probs = torch.nn.functional.softmax(output, dim=1)
conf, _ = torch.max(probs, 1)

print('Predicted Class:', predicted_class.item())
print('Predicted Label:', id2label[predicted_class.item()])
print(f'Confidence: {conf.item()*100}%')

plt.title(id2label[predicted_class.item()])
plt.axis("off")
plt.imshow(img)
plt.show()
```