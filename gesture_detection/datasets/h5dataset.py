import csv
import albumentations as A
from typing import Union, List, Tuple

import h5py
import argparse
from pathlib import Path

import numpy as np
import torch.utils.data
import torchvision
import lightning as L
from PIL import Image
from io import BytesIO

from lightning_utilities.core.rank_zero import rank_zero_only

from gesture_detection.utility.find_num_worker_default import find_num_worker_default


def load_csv(path: Path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)

        return list(reader)


def bytesio_loader(path: Path):
    with open(path, "rb") as f:
        buffer = BytesIO()
        buffer.write(f.read())
        buffer.seek(0)
        return buffer


def pil_loader(buffer: BytesIO):
    with Image.open(buffer) as img:
        img = img.convert('RGB')
        img = np.array(img)
        return img


def load_video(h5video: Path, indices=Union[None, np.ndarray]):
    h5f = h5py.File(h5video, swmr=True)
    labels = np.array(h5f["label"])
    h5frames = h5f["frame"]

    N = len(labels)
    if indices is not None:
        assert np.all(indices < N)
    else:
        indices = np.arange(N)

    images = []
    for i in indices:
        blob = np.array(h5frames[str(i)])
        images.append(pil_loader(BytesIO(blob)))

    labels = labels[indices]

    return images, labels


def load_info(h5info: Path):
    h5info = h5py.File(h5info, swmr=True)

    num_classes = int(np.array(h5info["num_classes"]))
    train_videos = [(x[0].decode("utf-8"), int(x[1])) for x in np.array(h5info["train_videos"])]
    test_videos = [(x[0].decode("utf-8"), int(x[1])) for x in np.array(h5info["test_videos"])]
    return {
        "num_classes": num_classes,
        "train_videos": train_videos,
        "test_videos": test_videos
    }


def convert_ipnhand(dataset_root: Path, destination_root: Path):
    frames_root = dataset_root / "frames"
    annot_path = dataset_root / "Annot_List.txt"

    annotations = load_csv(annot_path)

    # group annotations by their containing video
    ann_by_vid_dict = {}
    for ann in annotations:
        if ann["video"] in ann_by_vid_dict.keys():
            ann_by_vid_dict[ann["video"]].append(ann)
        else:
            ann_by_vid_dict[ann["video"]] = [ann]

    h5info = h5py.File(destination_root / "ipnhand.hdf5", "w")
    h5info.create_dataset("num_classes", data=14)

    with open(dataset_root / "Video_TrainList.txt", "r") as f:
        reader = csv.DictReader(f, fieldnames=["video", "length"], delimiter="\t")
        data = [(x["video"], x["length"]) for x in list(reader)]
        h5info.create_dataset("train_videos", data=data)

    with open(dataset_root / "Video_TestList.txt", "r") as f:
        reader = csv.DictReader(f, fieldnames=["video", "length"], delimiter="\t")
        data = [(x["video"], x["length"]) for x in list(reader)]
        h5info.create_dataset("test_videos", data=data)

    # create a list of annotations for each video
    # break label list into sequences of length sample_length
    ann_by_vid_list = {}
    for vid in ann_by_vid_dict.keys():
        video_path = frames_root / vid

        if not video_path.exists():
            continue

        print(f"Processing video: {vid}", end="\r")

        anns = ann_by_vid_dict[vid]
        anns = sorted(anns, key=lambda ann: ann["t_start"])  # sort by t_start

        labels = []
        for ann in anns:
            labels.extend([int(ann["id"])] * int(ann["frames"]))
        ann_by_vid_list[vid] = labels

        h5f = h5py.File(destination_root / f"{vid}.hdf5", "w")
        h5f.create_dataset("label", (len(labels)), dtype=int, data=labels)

        h5_frames = h5f.create_group("frame")
        for i in range(len(labels)):
            buffer = bytesio_loader(video_path / f"{vid}_{i + 1:06d}.jpg")
            bbytes = np.void(buffer.getbuffer().tobytes())

            h5_frames.create_dataset(str(i), dtype=h5py.opaque_dtype(bbytes.dtype), data=np.void(bbytes))
    print("")


class H5Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_root: Path, videos: List[Tuple[str, int]], sample_length: int, transform=None):
        super().__init__()

        self.dataset_root = dataset_root
        self.videos = videos
        self.transform = transform
        self.sample_length = sample_length

        self.dataset = []
        for (video, length) in videos:
            for idy in range(0, length, sample_length):
                if idy + sample_length >= length:
                    self.dataset.append({
                        "video": video,
                        "indices": np.linspace(length - sample_length, length, sample_length).astype(np.uint8)
                    })
                else:
                    self.dataset.append({
                        "video": video,
                        "indices": np.linspace(idy, idy + sample_length, sample_length).astype(np.uint8)
                    })

    def __getitem__(self, index):
        video = self.dataset[index]['video']
        indices = self.dataset[index]['indices']

        clip, labels = load_video(self.dataset_root / f"{video}.hdf5", indices)

        if self.transform is not None:
            clip_dict = {f"image{i}": clip[i] for i in range(self.sample_length)}
            clip_dict["image"] = clip[0]
            clip_dict = self.transform(**clip_dict)
            clip = [clip_dict[f"image{i}"] for i in range(self.sample_length)]
        clip = [torchvision.transforms.functional.to_tensor(img) for img in clip]

        image_dimension = clip[0].shape[-2:]
        clip = torch.cat(clip, 0).view((self.sample_length, -1) + image_dimension)

        # subtract 1 so that labels are in range [0, 13]
        target = torch.tensor(labels) - 1

        return clip, target

    def __len__(self):
        return len(self.dataset)


class H5DataModule(L.LightningDataModule):
    name = "H5"

    def __init__(
            self,
            dataset_root: str,
            dataset_info_filename: str,
            batch_size: int = 32,
            num_workers: int | None = None,
            pin_memory: bool = False,
            train_ratio: float = 0.9,
            seed: int = 42,
            sample_length: int = 32,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_root = Path(dataset_root)
        self.dataset_info_filename = dataset_info_filename
        self.dataset_info = None
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
        self.dataset_info = load_info(Path(self.dataset_root / self.dataset_info_filename))

    def setup(self, stage: str = None):
        # transforms
        transform = A.Compose([
            #A.RGBShift(0.05, 0.05, 0.05),
            #A.Rotate(30, crop_border=True),
            #A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            A.Resize(224, 224)
        ], additional_targets={f"image{i}": "image" for i in range(self.sample_length)})

        # split dataset
        if stage == "fit" or stage is None:
            dataset = H5Dataset(
                self.dataset_root,
                self.dataset_info["train_videos"],
                self.sample_length,
                transform=transform
            )
            num_train = int(len(dataset) * self.train_ratio)
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(
                dataset=dataset,
                lengths=[num_train, len(dataset) - num_train],
                generator=torch.Generator().manual_seed(self.seed)
            )

        if stage == "test" or stage is None:
            self.test_dataset = H5Dataset(
                self.dataset_root,
                self.dataset_info["test_videos"],
                self.sample_length,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ipn", type=Path)
    argparser.add_argument("--h5", type=Path)
    argparser.add_argument("--convert", action="store_true")
    argparser.add_argument("--info", action="store_true")
    argparser.add_argument("--infofile", type=Path)
    args = argparser.parse_args()
    if args.convert:
        convert_ipnhand(args.ipn, args.h5)
    if args.info:
        info = load_info(args.infofile)
        print(info)

    info = load_info(args.infofile)
    dataset = H5Dataset(args.h5, info["train_videos"], 8)
    item = dataset.__getitem__(0)
    print("")


if __name__ == "__main__":
    main()
