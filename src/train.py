"""Run the training here."""
from pytorch_accelerated import Trainer
from torch import nn, optim
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import ToTensor

training_data = ImageFolder(
    "/workspaces/real-time-hand-gesture-detection/data/raw/hand_gesture_recognition_dataset/train",
    transform=ToTensor(),
)

train_dataset, validation_dataset = random_split(
    training_data, [int(len(training_data) * 0.8), int(len(training_data) * 0.2)]
)


model = resnet18(weights=ResNet18_Weights.DEFAULT)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loss_fn = nn.CrossEntropyLoss()


# TODO: Check if using accelerated or lightning makes sense here
trainer = Trainer(
    model=model,
    loss_func=train_loss_fn,
    optimizer=optimizer,
)


if __name__ == "__main__":
    trainer.train(
        train_dataset=train_dataset, eval_dataset=validation_dataset, num_epochs=2, per_device_batch_size=32
    )
