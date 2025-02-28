import random
import torch.utils.data.dataset
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def get_slice(path, slice_num=500, input_size=None):
    img_tensor = torch.load(path)
    if input_size is None:
        img_slice = img_tensor[slice_num]
    else:
        img_slice = torchvision.transforms.functional.center_crop(img_tensor[slice_num], input_size)
    return img_slice.unsqueeze(0)

def get_dataloader(cfg):
    dataloader = DataLoader(
        BetaSegDataset2D(cfg["DATASET"]),
        batch_size=cfg["DATASET"]["batch_size"],
        num_workers=cfg["DATASET"]["num_workers"],
        pin_memory=True,
        shuffle=True
    )
    return dataloader

class BetaSegDataset2D(torch.utils.data.dataset.Dataset):
    """
    A dataset class that samples 2D slices from 3D volumes and applies data augmentation.

    Args:
        cfg (dict): A dictionary containing configuration parameters for the dataset.

    Attributes:
        path_list (list): A txt file for alist of paths to the input data files.
        input_size (int): The size of the 2D slices to sample from the 3D volumes.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a randomly sampled 2D slice from the input data at the given index.
        data_transform(image, seg): Augments 2D data.
        sample_cord(data_idx, axis): Samples a 2D slice from the input data at the given index and axis.

    """

    def __init__(self, cfg, aug_data=True):
        path = cfg["path_list"]
        self.vol_size = cfg["input_size"]
        self.aug_data = aug_data
        self.data_list = []
        self.seg_list = []
        tt = ToTensor()

        with open(path, 'r') as file:
            paths = file.readlines()
            self.path_list = paths
        for path in self.path_list:
            path_name = path.strip()
            seg_name = path_name.replace('source', 'seg')

            # We convert the tif file to pt in advance (without doing any preprocessing).
            # If you want to load the tif file directly, use tiffile.imread and ToTensor() instead.
            img_tensor = torch.load(path_name)
            seg_tensor = torch.load(seg_name)

            self.data_list.append(img_tensor)
            self.seg_list.append(seg_tensor)

    def __len__(self):
        """
        Returns the number of samples in each epoch.
        """
        return 200

    def __getitem__(self, idx):
        """
        Returns a randomly sampled 2D slice from the input data at the given index.
        Returns:
            A tensor representing a randomly sampled 2D slice from the input data.
        """
        curr_data_idx = random.randrange(0, len(self.data_list))  # select dataset
        axis = random.randrange(0, 3)  # select axis
        return self.sample_cord(curr_data_idx, axis)  # return 2D slice

    def data_transform(self, image, seg):
        """
        Returns an augmented image (augmentation includes flip and rotation.)
        Returns:
            A tensor representing a randomly sampled 2D slice from the input data.
        """
        image = image.unsqueeze(0)
        seg = seg.unsqueeze(0)
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            seg = torch.flip(seg, dims=[1])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            seg = torch.flip(seg, dims=[2])
        angle = random.choice([0, 90, 180, 270])
        image = torchvision.transforms.functional.rotate(image, angle)
        seg = torchvision.transforms.functional.rotate(seg, angle)
        return image.squeeze(), seg.squeeze()

    def sample_cord(self, data_idx, axis):
        """
        Samples a 2D slice from the input data at the given index and axis.

        Args:
            data_idx (int): The index of the input data to sample from.
            axis (int): The axis along which to sample the 2D slice.

        Returns:
            A tensor representing a 2D slice sampled from the input data at the given index and axis.
        """
        data = self.data_list[data_idx] # get dataset
        seg = self.seg_list[data_idx] # get dataset
        d_z, d_x, d_y = data.shape

        if axis < 1:
            # z axis
            x_sample = torch.randint(low=0, high=min(1, int(d_x - self.vol_size - 1)), size=(1,))
            y_sample = torch.randint(low=0, high=min(1, int(d_y - self.vol_size - 1)), size=(1,))
            z_sample = torch.randint(low=0, high=int(d_z), size=(1,))
            sample = data[z_sample, x_sample: x_sample + self.vol_size, y_sample: y_sample + self.vol_size].squeeze()
            seg_sample = seg[z_sample, x_sample: x_sample + self.vol_size, y_sample: y_sample + self.vol_size].squeeze()
        elif axis < 2:
            # x axis
            x_sample = torch.randint(low=0, high=int(d_x), size=(1,))
            y_sample = torch.randint(low=0, high=min(1, int(d_y - self.vol_size - 1)), size=(1,))
            z_sample = torch.randint(low=0, high=min(1, int(d_z - self.vol_size - 1)), size=(1,))
            sample = data[z_sample: z_sample + self.vol_size, x_sample, y_sample: y_sample + self.vol_size].squeeze()
            seg_sample = seg[z_sample: z_sample + self.vol_size, x_sample, y_sample: y_sample + self.vol_size].squeeze()
        else:
            # y axis
            x_sample = torch.randint(low=0, high=min(1, int(d_x - self.vol_size - 1)), size=(1,))
            y_sample = torch.randint(low=0, high=int(d_y), size=(1,))
            z_sample = torch.randint(low=0, high=min(1, int(d_z - self.vol_size - 1)), size=(1,))
            sample = data[z_sample: z_sample + self.vol_size, x_sample: x_sample + self.vol_size, y_sample].squeeze()
            seg_sample = seg[z_sample: z_sample + self.vol_size, x_sample: x_sample + self.vol_size, y_sample].squeeze()
        if self.aug_data:
            return self.data_transform(sample, seg_sample)
        return sample, seg_sample