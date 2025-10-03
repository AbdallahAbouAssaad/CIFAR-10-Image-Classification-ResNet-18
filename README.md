CIFAR-10 Image Classification using Transfer Learning on ResNet-18

Project Overview

This project implements transfer learning using ResNet-18 on the CIFAR-10 dataset. The goal is to classify images into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using a pre-trained convolutional neural network.

The model underwent fine-tuning progressively:
1.Started with only the final fully-connected layer.
2.Unfroze blocks 3 and 4, applying different learning rates per block.
3.Finally, unfreezing block 2 with a smaller learning rate for better feature adaptation.
-------------------------------------------

Dataset

CIFAR-10:
50,000 training images
10,000 test images
Image size: 32x32 RGB (Channels x Height x Width = 3 x 32 x 32)
-------------------------------------------

Data augmentation used:
-Random horizontal flip
-Random crop with padding
-Color jitter (brightness, contrast, saturation, hue)
-Random erasing
-------------------------------------------

Training Setup

Optimizer: Adam with different learning rates per block: (The deeper the less lr, since base layers shouldn't change much as they are already good at what they do, capturing edges..)
Block 2: 0.0001
Block 3: 0.0005
Block 4: 0.001
FC Layer: 0.001

Early stopping: stops if validation loss does not improve for 5 epochs

Epochs: up to 100, typically stopping earlier due to early stopping

Batch size: 128
-------------------------------------------

Results

Final Accuracy on Test Set: 85.78% (good for small models like Resnet-18)

-------------------------------------------

Limitations and Future Work

ResNet-18 architecture is the limiting factor:
1.Small depth and fewer parameters compared to ResNet-34, ResNet-50, or EfficientNet.
2.May not fully capture complex patterns in CIFAR-10 images.

Next steps to improve accuracy:
1.Use larger ResNet variants (ResNet-34, ResNet-50) or other architectures (DenseNet, EfficientNet).
2.Train with more epochs and fine-tuned learning rate schedules.
3.Apply mixup, CutMix, or AutoAugment for stronger data augmentation. (Although not advised as a first step, because I already applied enough augmentation and changing the model is better)
