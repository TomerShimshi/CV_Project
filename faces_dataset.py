"""Custom faces dataset."""

import os
from numpy.lib.type_check import imag

import torch
import numpy as np 
from PIL import Image
from torch.utils.data import Dataset
# addedthe following lib to turn PIL image to tensor
import torchvision.transforms as transforms


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index): #-> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        lable = np.random.randint(0,2) # zero will be real 1 will be fake
        if lable == 0:
            image_path = os.path.join(self.root_path, 'real') 
            image_path = os.path.join(image_path, self.real_image_names[index % len(self.real_image_names)]) 
        else:
            image_path = os.path.join(self.root_path, 'fake') 
            image_path = os.path.join(image_path, self.fake_image_names[index % len(self.fake_image_names)])
        image =  Image.open(image_path) 
        if self.transform != None:
            image= self.transform(image)
        else:
            # the following code is based on : https://www.tutorialspoint.com/how-to-convert-an-image-to-a-pytorch-tensor
             # Define a transform to convert the image to tensor
             transform = transforms.ToTensor()
            # Convert the image to PyTorch tensor
             image = transform(image)
             #image = tensor

        

        
        return (image,lable)
    def __len__(self):
        """Return the number of images in the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        #since both fake and reak DB are of the same size
        #then it will be easyer to implent the get item this way 
        return min(len(self.real_image_names), len(self.fake_image_names))