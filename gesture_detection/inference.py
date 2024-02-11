import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from gesture_detection.models.lstm import LSTM

@torch.no_grad()
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ckpt", type=Path)
    args = argparser.parse_args()

    state_dict = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    model = LSTM(26)
    model.load_state_dict(state_dict)
    cam = cv2.VideoCapture(0)

    input_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    hx = None
    fig = plt.figure()
    bar_container = plt.bar(np.arange(26), np.zeros(26))
    ax = plt.gca()
    ax.set_xlim([0, 26])
    ax.set_ylim([0, 1])

    try:
        frames = -1
        while True:
            result, image = cam.read()
            cv2.imshow("cam", image)

            image = cv2.resize(image, (100, 100))
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            frames += 1

            if frames % 3 == 0:
                input_tensor = input_transform(image)
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

                preds, hx = model.forward(input_tensor, hx_init=hx, return_hx=True)
                probs = torch.nn.functional.softmax(preds.flatten(), dim=0)

                for idx, rect in enumerate(bar_container):
                    rect.set_height(probs[idx])
                fig.canvas.draw()
                bar_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
                bar_plot = bar_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                bar_plot = cv2.cvtColor(bar_plot, cv2.COLOR_RGB2BGR)

                cv2.imshow("bar", bar_plot)

            cv2.waitKey(5)

    except KeyboardInterrupt:
        print(f"Got KeyboardInterrupt. Stopping inference.")


if __name__ == "__main__":
    main()