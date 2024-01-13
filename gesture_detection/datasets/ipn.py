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
import albumentations as A

from gesture_detection.config.base_config import get_base_config
from gesture_detection.utility.find_num_worker_default import find_num_worker_default


def load_csv(path):
    with open(os.path.abspath(path), 'r') as f:
        reader = csv.DictReader(f)

        return list(reader)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            img = np.array(img)
            return img


def get_video_loader():
    return video_loader


def video_loader(video_path, frame_indices):
    image_loader = pil_loader

    video = []
    for i in frame_indices:
        image_path = os.path.join(video_path, '{:s}_{:06d}.jpg'.format(video_path.split('/')[-1], i+1))

        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            raise FileNotFoundError(f'{image_path} does not exist')

    return video


def sample_indices(start, end, sample_length):
    return np.linspace(start, end, sample_length).astype(int)


def all_indices(start, end):
    return np.range(start, end + 1).astype(int)


def create_dataset(data_path, annotation_path, sample_length):
    annotations = load_csv(annotation_path)

    # group annotations by their containing video
    ann_by_vid_dict = {}
    for ann in annotations:
        if ann["video"] in ann_by_vid_dict.keys():
            ann_by_vid_dict[ann["video"]].append(ann)
        else:
            ann_by_vid_dict[ann["video"]] = [ann]

    # create a list of annotations for each video
    # break label list into sequences of length sample_length
    ann_by_vid_list = {}
    dataset = []
    for vid in ann_by_vid_dict.keys():
        video_path = os.path.join(data_path, vid)

        if not os.path.exists(video_path):
            continue

        anns = ann_by_vid_dict[vid]
        anns = sorted(anns, key=lambda ann: ann["t_start"])  # sort by t_start

        labels = []
        for ann in anns:
            labels.extend([int(ann["id"])] * int(ann["frames"]))
        ann_by_vid_list[vid] = labels

        N = len(labels)
        for idx in range(0, N, sample_length):
            if idx+sample_length >= N:
                dataset.append({
                    'video': video_path,
                    'frame_indices': np.linspace(idx, idx + sample_length, sample_length).astype(np.uint8),
                    'frame_number': sample_length,
                    'labels': labels[N-sample_length-1:]
                })
            else:
                dataset.append({
                    'video': video_path,
                    'frame_indices': np.linspace(idx, idx+sample_length, sample_length).astype(np.uint8),
                    'frame_number': sample_length,
                    'labels': labels[idx:idx+sample_length]
                })

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
            clip_dict = {f"image{i}": clip[i] for i in range(frame_number)}
            clip_dict["image"] = clip[0]
            clip_dict = self.transform(**clip_dict)
            clip = [clip_dict[f"image{i}"] for i in range(frame_number)]
        clip = [torchvision.transforms.functional.to_tensor(img) for img in clip]

        image_dimension = clip[0].shape[-2:]
        clip = torch.cat(clip, 0).view((frame_number, -1) + image_dimension)

        # subtract 1 so that labels are in range [0, 13]
        target = torch.tensor(self.dataset[index]['labels']) - 1

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
        transform = A.Compose([
            A.RGBShift(0.05, 0.05, 0.05),
            A.Rotate(30, crop_border=True),
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            #A.Resize(224, 224)
        ], additional_targets={f"image{i}": "image" for i in range(self.sample_length)})

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
    sample_length = 64
    transform = A.Compose([
        A.RGBShift(0.078, 0.078, 0.078),
        A.Rotate(30, crop_border=True),
        A.RandomResizedCrop(120, 120)
    ], additional_targets={f"image{i}": "image" for i in range(sample_length)})
    dataset = IPN(
        "data/ipnhand/frames",
        "data/ipnhand/Annot_List.txt",
        8,
        transform
    )
    for i in range(10):
        dataset.__getitem__(i)


if __name__ == "__main__":
    main()
