import argparse
import csv
from io import BytesIO
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional

import albumentations as A
import cv2
import h5py
import lightning as L
import numpy as np
import torch.utils.data
import torchvision
from PIL import Image

from gesture_detection.utility.find_num_worker_default import find_num_worker_default

"""
IPNHand classes:

0: no gesture
1: pointing with one finger
2: pointing with two fingers
3: click with one finger
4: click with two fingers
5: throw up
6: throw down
7: throw left (from the subject's perspective)
8: throw right (from the subject's perspective)
9: open twice
10: double click with one finger
11: double click with two fingers
12: zoom in
13: zoom out

NVGesture classes:

1: move hand left (from subject's perspective)
2: move hand right (from subject's perspective)
3: move hand up
4: move hand down
5: move two fingers left
6: move two fingers right
7: move two fingers up
8: move two fingers down
9: click index finger (one finger)
10: call someone (close hands towards body)
11: open hand once
12: shaking hand
13: show index finger
14: show two fingers
15: show three fingers
16: push hand up
17: push hand down
18: push hand out
19: pull hand in
20: rotate fingers cw (subject's perspective)
21: rotate fingers ccw (subject's perspective)
22: push two fingers away
23: close hand two times
24: thumbs up
25: okay gesture

"""

MAP_JESTER_GESTURE_LABELS = {
    "Swiping Left": 0,
    "Swiping Right": 0,
    "Swiping Up": 0,
    "Swiping Down": 0,
    "Pushing Hand Away": 0,
    "Pulling Hand In": 0,
    "Sliding Two Fingers Left": 0,
    "Sliding Two Fingers Right": 0,
    "Sliding Two Fingers Down": 0,
    "Sliding Two Fingers Up": 0,
    "Pushing Two Fingers Away": 0,
    "Pulling Two Fingers In": 0,
    "Rolling Hand Forward": 0,
    "Rolling Hand Backward": 0,
    "Turning Hand Clockwise": 0,
    "Turning Hand Counterclockwise": 0,
    "Zooming In With Full Hand": 0,
    "Zooming Out With Full Hand": 0,
    "Zooming In With Two Fingers": 0,
    "Zooming Out With Two Fingers": 0,
    "Thumb Up": 0,
    "Thumb Down": 0,
    "Shaking Hand": 0,
    "Stop Sign": 0,
    "Drumming Fingers": 0,
    "No gesture": 0,
    "Doing other things": 0
}

ONLY_JESTER_GESTURE_LABELS = {} # {v: k+1 for k, v in MAP_JESTER_GESTURE_LABELS.items()}  # labels 1 - 25
ONLY_JESTER_GESTURE_LABELS[0] = 0  # add no gesture class

MAP_NVGESTURE_LABELS = {
    0:  14,
    1:  15,
    2:  16,
    3:  17,
    4:  18,
    5:  19,
    6:  20,
    7:  21,
    8:  22,
    9:  23,
    10: 24,
    11: 25,
    12: 26,
    13: 27,
    14: 28,
    15: 29,
    16: 30,
    17: 31,
    18: 32,
    19: 33,
    20: 34,
    21: 35,
    22: 36,
    23: 37,
    24: 38,
}

ONLY_NVGESTURE_LABELS = {v: k+1 for k, v in MAP_NVGESTURE_LABELS.items()}  # labels 1 - 25
ONLY_NVGESTURE_LABELS[0] = 0  # add no gesture class


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
    try:
        h5f = h5py.File(h5video, swmr=True)
    except FileNotFoundError:
        print(f"Couldn't find {h5video}. Trying again. ")
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

    num_classes = None
    if "num_classes" in h5info.keys():
        num_classes = int(np.array(h5info["num_classes"]))
    train_videos = [(x[0].decode("utf-8"), int(x[1])) for x in np.array(h5info["train_videos"])]
    test_videos = [(x[0].decode("utf-8"), int(x[1])) for x in np.array(h5info["test_videos"])]
    eval_videos = None
    if "eval_videos" in h5info.keys():
        eval_videos = [(x[0].decode("utf-8"), int(x[1])) for x in np.array(h5info["eval_videos"])]
    return {
        "num_classes": num_classes,
        "train_videos": train_videos,
        "test_videos": test_videos,
        "eval_videos": eval_videos
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
        labels = np.array(labels, np.uint8) - 1

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
    for i, frameIndx in enumerate(range(total_frames)):
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
    for config in test_list:
        vid, length = extract_nvgesture_video(dataset_root, destination_root, config, sensor)
        test_videos.append((bytes(vid, "utf-8"), length))

    h5info.create_dataset("test_videos", data=test_videos)


def convert_jester(dataset_root: Path, destination_root: Path):
    frames_root = dataset_root / "frames"
    labels_root = dataset_root / "labels"

    h5info = h5py.File(destination_root / "jester.hdf5", "w")

    for subset in ["validation.csv"]:  # ["train.csv", "validation.csv", "test-answers.csv"]:
        with open(labels_root / subset, "r") as f:
            reader = csv.DictReader(f, fieldnames=["video", "label"], delimiter=";")

            videos_labels = {
                x["video"]: MAP_JESTER_GESTURE_LABELS[x["label"]] for x in list(reader)
            }

            data = []

            for video in sorted(videos_labels.keys()):
                idx = 1

                h5f = h5py.File(destination_root / f"{video}.hdf5", "w")

                h5_frames = h5f.create_group("frame")

                while (img := frames_root / video / f"{idx:05d}.jpg").exists():
                    frame = cv2.imread(str(img))
                    frame = cv2.resize(frame, (320, 240))

                    ret, buffer = cv2.imencode(".jpg", frame)
                    buffer = BytesIO(buffer)
                    bbytes = np.void(buffer.getbuffer().tobytes())

                    h5_frames.create_dataset(str(idx - 1), dtype=h5py.opaque_dtype(bbytes.dtype), data=np.void(bbytes))

                    idx += 1

                if idx == 1:
                    continue

                labels = np.ones((idx - 1), dtype=np.uint8) * videos_labels[video]
                h5f.create_dataset("label", (len(labels)), dtype=int, data=labels)
                data.append((bytes(video, "utf-8"), len(labels)))
                break

            h5info.create_dataset("train_videos", data=data)  # data [(vid, length), (...)]



def merge_datasets(a: Path, b: Path, out: Path):
    info_a = load_info(a)
    info_b = load_info(b)

    num_classes = info_a["num_classes"] + info_b["num_classes"]
    train_videos = info_a["train_videos"] + info_b["train_videos"]
    test_videos = info_a["test_videos"] + info_b["test_videos"]
    h5info = h5py.File(out, "w")
    h5info.create_dataset("num_classes", data=num_classes)

    h5info.create_dataset("train_videos", data=[(bytes(vid, "utf-8"), length) for vid, length in train_videos])
    h5info.create_dataset("test_videos", data=[(bytes(vid, "utf-8"), length) for vid, length in test_videos])


class H5Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_root: Path, videos: List[Tuple[str, int]], sample_length: int, transform=None,
                 sequence_transform: bool = True, labels_transform: Optional[Dict] = None):
        super().__init__()

        self.dataset_root = dataset_root
        self.videos = videos
        self.transform = transform
        self.sample_length = sample_length
        self.sequence_transform = sequence_transform
        self.labels_transform = labels_transform

        self.dataset = []
        self.video_length = {}
        for (video, length) in videos:
            self.video_length[video] = length
            for idy in range(0, length, sample_length):
                if idy + sample_length >= length:
                    indices = np.linspace(length - sample_length, length - 1, sample_length).astype(np.uint8)
                else:
                    indices = np.linspace(idy, idy + sample_length - 1, sample_length).astype(np.uint8)

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

        if self.labels_transform is not None:
            labels = [self.labels_transform[label] for label in labels]

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

        return clip, torch.tensor(labels)

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

        self.labels_transform = None
        if self.dataset_info_filename in ["ipnhand.hdf5", "nvgesture.hdf5", "jestergesture.hdf5"]:
            if self.dataset_info_filename == "ipnhand.hdf5":
                print(f"Using IPNHand labels only!")
                # nothing to do since the IPNHand labels already occupy classes 0-13
            elif self.dataset_info_filename == "nvgesture.hdf5":
                print(f"Using NVGesture labels only!")
                self.labels_transform = ONLY_NVGESTURE_LABELS
            elif self.dataset_info_filename == "jestergesture.hdf5":
                print(f"Using JesterGesture labels only!")
                self.labels_transform = ONLY_JESTER_GESTURE_LABELS

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
                transform=transform, labels_transform=self.labels_transform
            )
            self.test_dataset = H5Dataset(
                self.dataset_root,
                self.dataset_info["test_videos"],
                self.sample_length,
                transform=A.Compose([
                    A.Resize(self.sample_size, self.sample_size)
                ], additional_targets={f"image{i}": "image" for i in range(self.sample_length)}),
                sequence_transform=False, labels_transform=self.labels_transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = H5Dataset(
                self.dataset_root,
                self.dataset_info["test_videos"],
                self.sample_length,
                transform=A.Compose([
                    A.Resize(self.sample_size, self.sample_size)
                ], additional_targets={f"image{i}": "image" for i in range(self.sample_length)}),
                sequence_transform=False, labels_transform=self.labels_transform
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
    argparser.add_argument("--jg", type=Path)
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
        if args.jg is not None and args.h5 is not None:
            convert_jester(args.jg, args.h5)
    if args.info:
        info = load_info(args.infofile)
        print(info)

    info = load_info(args.infofile)
    dataset = H5Dataset(args.h5, info["train_videos"], 8)
    item = dataset.__getitem__(0)
    print("")


if __name__ == "__main__":
    main()
