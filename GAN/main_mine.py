import torch
import pandas as pd
import os
import time
import cv2 as cv
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.autograd import Variable


class MonetDataset(Dataset):
    def __init__()