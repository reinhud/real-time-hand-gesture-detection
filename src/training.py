import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset

accelerator = Accelerator()
