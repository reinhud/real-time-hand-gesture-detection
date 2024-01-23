import argparse
import csv
from io import BytesIO
from pathlib import Path
from typing import Union, List, Tuple, Dict

import albumentations as A
import cv2
import h5py
import lightning as L
import numpy as np
import torch.utils.data
import torchvision
from PIL import Image

from gesture_detection.utility.find_num_worker_default import find_num_worker_default

MAP_NVGESTURE_LABELS = {
    0: 15,
    1: 16,
    2: 17,
    3: 18,
    4: 19,
    5: 20,
    6: 21,
    7: 22,
    8: 23,
    9: 24,
    10: 25,
    11: 26,
    12: 27,
    13: 28,
    14: 29,
    15: 30,
    16: 31,
    17: 32,
    18: 33,
    19: 34,
    20: 35,
    21: 36,
    22: 37,
    23: 38,
    24: 39,
}


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


def load_video(h5video: Path, indices: Union[None, np.ndarray] = None, labels_only: bool = False):
    h5f = h5py.File(h5video, swmr=True)
    labels = np.array(h5f["label"])
    h5frames = h5f["frame"]

    N = len(labels)
    if indices is not None:
        assert np.all(indices < N)
    else:
        indices = np.arange(N)

    if labels_only:
        return labels[indices]

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


def load_split_nvgesture(file_with_split: Path):
    list_split = []
    with open(file_with_split, 'rb') as f:
        dict_name = file_with_split.stem
        dict_name = dict_name[:dict_name.find('_')]

        for line in f:
            params = line.decode("utf-8").split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            path = params[0].split(':')[1]
            for param in params[1:]:
                parsed = param.split(':')
                key = parsed[0]
                if key == 'label':
                    # make label start from 0
                    label = int(parsed[1]) - 1
                    params_dictionary['label'] = label
                elif key in ('depth', 'color', 'duo_left'):
                    # othrwise only sensors format: <sensor name>:<folder>:<start frame>:<end frame>
                    sensor_name = key
                    # first store path
                    params_dictionary[key] = path + '/' + parsed[1]
                    # store start frame
                    params_dictionary[key + '_start'] = int(parsed[2])

                    params_dictionary[key + '_end'] = int(parsed[3])

            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']

            list_split.append(params_dictionary)

    return list_split


def extract_nvgesture_video(
        dataset_root: Path, destination_root: Path,
        config: Dict,
        sensor: str,
        image_width: int = 320,
        image_height: int = 240,
):
    path = dataset_root / f"{config[sensor]}.avi"
    start_frame = config[sensor + '_start']
    end_frame = config[sensor + '_end']
    label = config['label']

    frames_to_load = range(start_frame, end_frame)

    cap = cv2.VideoCapture(str(path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    labels = np.zeros((total_frames))
    labels[start_frame:end_frame] = MAP_NVGESTURE_LABELS[label]

    vid = "_".join(config[sensor].split("/")[2:4])
    h5f = h5py.File(destination_root / f"{vid}.hdf5", "w")
    h5f.create_dataset("label", dtype=int, data=labels)

    h5_frames = h5f.create_group("frame")

    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i, frameIndx in enumerate(frames_to_load):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (image_width, image_height))
            if sensor != "color":
                frame = frame[..., 0]
                frame = frame[..., np.newaxis]

            ret, buffer = cv2.imencode(".jpg", frame)
            buffer = BytesIO(buffer)
            bbytes = np.void(buffer.getbuffer().tobytes())

            h5_frames.create_dataset(str(i), dtype=h5py.opaque_dtype(bbytes.dtype), data=np.void(bbytes))
        else:
            raise ValueError("could not read frame")

    cap.release()

    return vid, total_frames


def convert_nvgesture(dataset_root: Path, destination_root: Path):
    train_list = load_split_nvgesture(dataset_root / "nvgesture_train_correct_cvpr2016_v2.lst")
    test_list = load_split_nvgesture(dataset_root / "nvgesture_test_correct_cvpr2016_v2.lst")
    sensors = ["color", "depth", "duo_left", "duo_right", "duo_disparity"]

    h5info = h5py.File(destination_root / "nvgesture.hdf5", "w")
    h5info.create_dataset("num_classes", data=25)

    sensor = sensors[0]
    train_videos = []
    for config in train_list:
        vid, length = extract_nvgesture_video(dataset_root, destination_root, config, sensor)
        train_videos.append((bytes(vid, "utf-8"), length))

    h5info.create_dataset("train_videos", data=train_videos)

    test_videos = []
    for config in train_list:
        vid, length = extract_nvgesture_video(dataset_root, destination_root, config, sensor)
        test_videos.append((bytes(vid, "utf-8"), length))

    h5info.create_dataset("test_videos", data=test_videos)


class H5Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_root: Path, videos: List[Tuple[str, int]], sample_length: int, transform=None, sequence_transform: bool = True):
        super().__init__()

        self.dataset_root = dataset_root
        self.videos = videos
        self.transform = transform
        self.sample_length = sample_length
        self.sequence_transform = sequence_transform

        self.dataset = []
        self.video_length = {}
        for (video, length) in videos:
            self.video_length[video] = length
            for idy in range(0, length, sample_length):
                if idy + sample_length >= length:
                    indices = np.linspace(length - sample_length, length, sample_length).astype(np.uint8)
                else:
                    indices = np.linspace(idy, idy + sample_length, sample_length).astype(np.uint8)
                labels = load_video(self.dataset_root / f"{video}.hdf5", indices, labels_only=True)

                # do not include examples of the majority classes
                # if np.all(labels <= 3):
                #    continue
                self.dataset.append({
                    "video": video,
                    "indices": indices
                })

    def __getitem__(self, index):
        video = self.dataset[index]['video']
        indices = self.dataset[index]['indices'].copy()
        length = self.video_length[video]

        if self.sequence_transform:
            if np.random.random() < 0.1:
                # randomly shift sequences to left or right
                offset = np.random.randint(0, self.sample_length // 2)
                if indices.max() + offset < length:
                    indices += offset
                else:
                    indices -= offset
            elif np.random.random() < 0.7:
                # enlarge sequence and drop random indices
                start = indices.min()
                stop = indices.max()

                if stop + self.sample_length // 4 < length:
                    indices = np.linspace(
                        start, stop + self.sample_length // 4, int(self.sample_length * 1.25), dtype=np.uint8
                    )
                else:
                    indices = np.linspace(
                        start - self.sample_length // 4, stop, int(self.sample_length * 1.25), dtype=np.uint8
                    )

                keep_indices = torch.randperm(int(self.sample_length * 1.25))[:self.sample_length]
                indices = indices[keep_indices]
                indices.sort()
            else:
                # draw random indices
                indices = np.random.choice(indices, size=self.sample_length)
                indices.sort()

        clip, labels = load_video(self.dataset_root / f"{video}.hdf5", indices)

        if self.transform is not None:
            clip_dict = {f"image{i}": clip[i] for i in range(self.sample_length)}
            clip_dict["image"] = clip[0]
            clip_dict = self.transform(**clip_dict)
            clip = [clip_dict[f"image{i}"] for i in range(self.sample_length)]
        clip = [torchvision.transforms.functional.to_tensor(img) for img in clip]
        clip = [torchvision.transforms.functional.normalize(
            img, torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]), False
        ) for img in clip]

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
            seed: int = 42,
            sample_length: int = 32,
            sample_size: int = 224
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_root = Path(dataset_root)
        self.dataset_info_filename = dataset_info_filename
        self.dataset_info = None
        self.num_workers = find_num_worker_default(num_workers)
        self.pin_memory = pin_memory
        self.seed = seed
        self.sample_length = sample_length
        self.sample_size = sample_size
        self.save_hyperparameters()

        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        self.dataset_info = load_info(Path(self.dataset_root / self.dataset_info_filename))

    def setup(self, stage: str = None):
        # transforms
        transform = A.Compose([
            A.OneOrOther(
                A.RGBShift(),
                A.ColorJitter()
            ),
            A.Blur(),
            A.GaussNoise(),
            A.OneOf([
                A.Compose([
                    A.Rotate(30, crop_border=False, p=1.0),
                    A.Resize(self.sample_size, self.sample_size)
                ]),
                A.RandomResizedCrop(self.sample_size, self.sample_size, scale=(0.8, 1.0))
            ], p=1.0)
        ], additional_targets={f"image{i}": "image" for i in range(self.sample_length)})

        # split dataset
        if stage == "fit" or stage is None:
            self.train_dataset = H5Dataset(
                self.dataset_root,
                self.dataset_info["train_videos"],
                self.sample_length,
                transform=transform
            )
            self.test_dataset = H5Dataset(
                self.dataset_root,
                self.dataset_info["test_videos"],
                self.sample_length,
                transform=A.Compose([
                    A.Resize(self.sample_size, self.sample_size)
                ], additional_targets={f"image{i}": "image" for i in range(self.sample_length)}),
                sequence_transform=False
            )

        if stage == "test" or stage is None:
            self.test_dataset = H5Dataset(
                self.dataset_root,
                self.dataset_info["test_videos"],
                self.sample_length,
                transform=A.Compose([
                    A.Resize(self.sample_size, self.sample_size)
                ], additional_targets={f"image{i}": "image" for i in range(self.sample_length)}),
                sequence_transform=False
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
            self.test_dataset,
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
    argparser.add_argument("--nvg", type=Path)
    argparser.add_argument("--h5", type=Path)
    argparser.add_argument("--convert", action="store_true")
    argparser.add_argument("--info", action="store_true")
    argparser.add_argument("--infofile", type=Path)
    args = argparser.parse_args()

    if args.convert:
        if args.ipn is not None and args.h5 is not None:
            convert_ipnhand(args.ipn, args.h5)
        if args.nvg is not None and args.h5 is not None:
            convert_nvgesture(args.nvg, args.h5)
    if args.info:
        info = load_info(args.infofile)
        print(info)

    info = load_info(args.infofile)
    dataset = H5Dataset(args.h5, info["train_videos"], 8)
    item = dataset.__getitem__(0)
    print("")


if __name__ == "__main__":
    main()
