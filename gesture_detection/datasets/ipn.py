import os

import torchvision.transforms.functional
from PIL import Image
import torch.utils.data as data
import csv
import numpy as np
import torch
import torchvision.transforms as transforms
import lightning as L
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from gesture_detection.config.base_config import get_base_config
from gesture_detection.utility.find_num_worker_default import find_num_worker_default


def load_csv(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)

        return list(reader)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            img = np.array(img).astype(np.float32) / 255
            return img


def get_video_loader():
    return video_loader


def video_loader(video_path, frame_indices):
    image_loader = pil_loader

    video = []
    for i in frame_indices:
        image_path = os.path.join(video_path, '{:s}_{:06d}.jpg'.format(video_path.split('/')[-1], i))

        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print(f'{image_path} does not exist')
            return video

    return video


def sample_indices(start, end, sample_length):
    return np.linspace(start, end, sample_length).astype(int)


def all_indices(start, end):
    return np.range(start, end + 1).astype(int)


def create_dataset(data_path, annotation_path, sample_length):
    annotations = load_csv(annotation_path)

    dataset = []

    for i, row in enumerate(annotations):
        video_path = os.path.join(data_path, row['video'])

        if not os.path.exists(video_path):
            continue

        start = int(row['t_start'])
        end = int(row['t_end'])

        if sample_length != None:
            frame_indices = sample_indices(start, end, sample_length)
        else:
            frame_indices = all_indices(start, end)

        frame_number = len(frame_indices)

        label = row['id']

        sample = {
            'id': i,
            'video': video_path,
            'frame_indices': frame_indices,
            'frame_number': frame_number,
            'label': label
        }

        dataset.append(sample)

    return dataset


class IPN(data.Dataset):
    """
    Args:
        data_path: 
        path to folder of video data 

        annotation_path: 
        path to file that contains annotations to each clip [video, label, id, t_start, t_end, frames] 

        sample_lenght: 
        number of frames that each sample clip should have. 
        If None the clip from t_start to t_end is taken, which results in uneven length of clips.

    Parameters:
        dataset:
        list sample information [id, video_path, frame_indices, frame_number, label], needed to return the correct frames for a clip.

        spacial_transforms:
        spacial transformation of the image. Needed to convert PIL Image to torch tensor.

        video_loader:
        method load and return the frames of one video.

    Returns:
        clip:
        Tensor[num_frames, channels, height, width] containing the images of one clip

        target:
        target label of the clip
    """

    def __init__(self, data_path, annotation_path, sample_length, transform=None,
                 video_loader=get_video_loader):
        self.video_loader = video_loader()
        self.dataset = create_dataset(data_path, annotation_path, sample_length)
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.dataset[index]['video']
        frame_indices = self.dataset[index]['frame_indices']
        frame_number = self.dataset[index]['frame_number']

        clip = self.video_loader(image_path, frame_indices)

        if self.transform is not None:
            clip = [self.transform(img) for img in clip]

        image_dimension = clip[0].shape[-2:]
        clip = torch.cat(clip, 0).view((frame_number, -1) + image_dimension)

        # subtract 1 so that labels are in range [0, 13]
        target = int(self.dataset[index]['label']) - 1

        return clip, target

    def __len__(self):
        return len(self.dataset)


class IPNDataModule(L.LightningDataModule):
    name = "IPNHand"

    def __init__(
            self,
            data_dir: str,
            batch_size: int = 32,
            num_workers: int | None = None,
            pin_memory: bool = False,
            train_ratio: float = 0.9,
            seed: int = 42,
            sample_length: int = 32,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = find_num_worker_default(num_workers)
        self.pin_memory = pin_memory
        self.seed = seed
        self.train_ratio = train_ratio
        self.sample_length = sample_length
        self.save_hyperparameters()

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        # transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                transforms.Resize((120, 120))
            ]
        )

        dataset_path = Path(self.data_dir)

        # split dataset
        if stage == "fit" or stage is None:
            dataset = IPN(
                dataset_path / "frames",
                dataset_path / "Annot_TrainList.txt",
                self.sample_length,
                transform=transform
            )
            num_train = int(len(dataset) * self.train_ratio)
            self.train_dataset, self.valid_dataset = random_split(
                dataset=dataset,
                lengths=[num_train, len(dataset) - num_train],
                generator=torch.Generator().manual_seed(self.seed)
            )

        if stage == "test" or stage is None:
            self.test_dataset = IPN(
                dataset_path / "frames",
                dataset_path / "Annot_TestList.txt",
                self.sample_length,
                transform=transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def main():
    dataset = IPN(
        "data/ipnhand/frames",
        "data/ipnhand/Annot_List.txt",
        64)
    print(dataset.__getitem__(0))


if __name__ == "__main__":
    main()
