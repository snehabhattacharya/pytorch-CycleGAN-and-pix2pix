import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np 


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path)
        # split AB image into A and B
        w, h = AB.size
        #print (w,h, "ABSIZEEEEEEEEEEEEEEEEEEEE", AB_path)
        w2 = int(w/2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        width, height = A.size
        
        result_A = Image.new(A.mode, (width, width), (255,255,255))
        result_A.paste(A, (0, (width - height) // 2))
        result_B = Image.new(B.mode, (width, width), (255,255,255))
        result_B.paste(B, (0, (width - height) // 2))
        #print (result_A.size)
        #print (result_B.size)
        if result_A.size[0] !=1280:
            result_A = result_A.crop((0,0,1280,1280))
            result_B = result_B.crop((0,0,1280,1280))
        # result_A = np.array(result_A)
        # result_B = np.array(result_B)
        #print (result_A.size, "shpeeeee A ")
        #print(result_B.size, "shape B")
                # apply the same transform to both A and B
        #transform_params = get_params(self.opt, (result_A.shape[0], result_A.shape[1]))
        transform_params = get_params(self.opt, (result_A.size))
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1),convert=True)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), convert=True)

        A = A_transform(result_A)
        B = B_transform(result_B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
