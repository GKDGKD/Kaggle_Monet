import os
from PIL import Image
import random
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, monet_path, photo_path, transform=None, unaligned=True):
        self.monet_path = monet_path
        self.photo_path = photo_path
        self.transform = transform
        self.unaligned = unaligned  # 是否对齐，即两个数据集的图片是否配对 或 图片数量是否一致
        self.monet_names = [name for name in list(filter(lambda x: x.endswith(".jpg"), 
                                                        os.listdir(self.monet_path)))]
        self.photo_names = [name for name in list(filter(lambda x: x.endswith(".jpg"), 
                                                        os.listdir(self.photo_path)))]

    def __getitem__(self, index):
        img_photo = os.path.join(self.photo_path, self.photo_names[index])
        img_photo = Image.open(img_photo).convert('RGB')

        if self.unaligned:
                    img_monet = os.path.join(self.monet_path, \
                                             self.monet_names[random.randint(0, len(self.monet_names)) - 1])
                    img_monet = Image.open(img_monet).convert('RGB')
        else:         
            img_monet = os.path.join(self.monet_path, self.monet_names[index % len(self.monet_names)])
            img_monet = Image.open(img_monet).convert('RGB')

        if self.transform is not None:
            img_monet = self.transform(img_monet)
            img_photo = self.transform(img_photo)

        return {"monet": img_monet, "photo": img_photo}

    def __len__(self):
        return max(len(self.monet_names), len(self.photo_names))