import os
from PIL import Image
import torch.utils.data as data
import csv
import numpy as np
import torch
import torchvision.transforms as transforms


def load_csv(path): 
    with open(path, 'r') as f:
        reader = csv.DictReader(f)

        return list(reader)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:

            return img.convert('RGB')
        
        
def get_video_loader():
    return video_loader


def get_spacial_transforms():
    return transforms.Compose([
        transforms.PILToTensor()
    ])


def video_loader(video_path, frame_indices):
    image_loader = pil_loader

    video = []
    for i in frame_indices:
        image_path = os.path.join(video_path, '{:s}_{:06d}.jpg'.format(video_path.split('/')[-1],i))

        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print(f'{image_path} does not exist')
            return video
        
    return video


def sample_indices(start, end, sample_length):
    return np.linspace(start, end, sample_length).astype(int)


def all_indices(start, end):
    return np.range(start, end+1).astype(int)


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

    def __init__(self, data_path, annotation_path, sample_length, spacial_transforms=get_spacial_transforms, video_loader=get_video_loader):

        self.video_loader = video_loader()
        self.dataset = create_dataset(data_path, annotation_path, sample_length)
        self.spacial_transforms = spacial_transforms()


    def __getitem__(self, index):

        image_path = self.dataset[index]['video'] 
        frame_indices = self.dataset[index]['frame_indices'] 
        frame_number = self.dataset[index]['frame_number']

        clip = self.video_loader(image_path, frame_indices)

        if self.spacial_transforms is not None:
            clip = [self.spacial_transforms(img) for img in clip]

        image_dimension = clip[0].shape[-2:]
        clip = torch.cat(clip, 0).view((frame_number, -1) + image_dimension)

        target = int(self.dataset[index]['label']) 

        return clip, target