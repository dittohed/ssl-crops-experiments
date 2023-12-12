import random
import json

from pathlib import Path
from typing import Callable, Tuple, Union, Optional

import PIL
import torchvision.transforms as T

from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset
from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.utils import IMAGENET_NORMALIZE


class FlatImageFolder(Dataset):
    """
    Custom PyTorch dataset for loading images from a single directory 
    (without class-wise subdirectories).
    """

    def __init__(self, data_dir: str, transform: Callable = None):
        super().__init__()

        self.paths = list(Path(data_dir).glob('*'))
        self.transform = transform

    def __getitem__(self, index: int):
        img = PIL.Image.open(self.paths[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img 
    
    def __len__(self):
    	return len(self.paths)


# TODO: docstring and review
class SuperpatchDataset(Dataset):
	"""
  	"""
  
	def __init__(
        self, data_dir: str, spatch_nns_json_path: str,
		spatch_transform_1: Callable = None, spatch_transform_2: Callable = None, 
      	input_height: int = 256, input_width: int = 256, patch_size: int = 16, 
		n_agg: int = 8, max_nearest: int = 8, coord_jitter: int = 0, 
    	return_whole_images: bool = False
    ):
		assert input_height % patch_size == 0
		assert input_width % patch_size == 0
		assert input_height % (n_agg*patch_size) == 0
		assert input_width % (n_agg*patch_size) == 0

		self.paths = sorted(Path(data_dir).glob('*'))
		# Make sure the order matches one used for creating .json
		# TODO: use sth smarter
		print('The following paths order will be used: ')
		print(f'  a) first: {self.paths[0]}')
		print(f'  b) last: {self.paths[-1]}\n')

		self.spatch_nns = self.load_spatch_nns(spatch_nns_json_path)

		self.n_spatches_row = input_height // patch_size // n_agg 
		self.n_spatches_col = input_width // patch_size // n_agg 
		self.n_spatches_per_img = self.n_spatches_row * self.n_spatches_col
		self.patch_size = patch_size
		self.n_agg = n_agg

		self.spatch_transform_1 = spatch_transform_1
		self.spatch_transform_2 = spatch_transform_2

		self.max_nearest = max_nearest
		self.coord_jitter = coord_jitter
		self.return_whole_images = return_whole_images

		self.resize_transform = T.Resize((input_height, input_width))
		self.to_tensor_transform = T.ToTensor()

	def __getitem__(self, img_idx):
		spatch1_idx = self.sample_spatch_idx_from_img_idx(img_idx)
		spatch2_idx = self.sample_spatch_idx_from_spatch_idx(spatch1_idx)

		spatch1_coords = self.get_spatch_coords(spatch1_idx)
		spatch2_coords = self.get_spatch_coords(spatch2_idx)

		img1_idx = spatch1_idx // self.n_spatches_per_img
		img2_idx = spatch2_idx // self.n_spatches_per_img

		img1, spatch1 = self.get_spatch_array(img1_idx, spatch1_coords)
		img2, spatch2 = self.get_spatch_array(img2_idx, spatch2_coords)

		if self.spatch_transform_1 is not None:
			spatch1 = self.spatch_transform_1(spatch1)
		if self.spatch_transform_2 is not None:
			spatch2 = self.spatch_transform_2(spatch2)

		if self.return_whole_images:  # For debug purposes
			return img1, spatch1, img2, spatch2
		else:
			return spatch1, spatch2

	def __len__(self):
		return len(self.paths)
  
	def load_spatch_nns(self, json_path: str) -> dict:
		with open(Path(json_path), 'r') as f:
			spatch_nns = json.load(
        		f, object_pairs_hook=lambda x: {int(k): v for k, v in x}
      		)
			
		return spatch_nns
  
	def sample_spatch_idx_from_img_idx(self, img_idx):
		return random.choice(
    		range(img_idx * self.n_spatches_per_img, (img_idx+1) * self.n_spatches_per_img)
    	)
  
	def sample_spatch_idx_from_spatch_idx(self, spatch_idx):
		# Determine index of furthest neighbor so that only
		# `self.max_nearest` neighbors are taken into account
		max_idx = min(len(self.spatch_nns[spatch_idx]), self.max_nearest)
		return random.choice(self.spatch_nns[spatch_idx][:max_idx])
	
	def get_spatch_coords(self, spatch_idx):
		local_spatch_idx = spatch_idx % self.n_spatches_per_img
		spatch_row_idx = local_spatch_idx // self.n_spatches_row
		spatch_col_idx = local_spatch_idx % self.n_spatches_col

		y_upper = spatch_row_idx * self.patch_size * self.n_agg
		x_upper = spatch_col_idx * self.patch_size * self.n_agg
		y_lower = (spatch_row_idx+1) * self.patch_size * self.n_agg
		x_lower = (spatch_col_idx+1) * self.patch_size * self.n_agg

		# TODO; coord jitter

		return y_upper, x_upper, y_lower, x_lower

	def get_spatch_array(self, img_idx, spatch_coords):
		img = PIL.Image.open(self.paths[img_idx]).convert('RGB')
		img = self.resize_transform(img)

		y_upper, x_upper, y_lower, x_lower = spatch_coords
		spatch = img.crop((x_upper, y_upper, x_lower, y_lower))
		
		return self.to_tensor_transform(img), spatch
	

class DINOViewTransform:
    """
    Version of `lightly.transforms.dino_transform.DINOViewTransform` 
	with no `RandomResizedCrop` (to use directly on crops).
	"""
    def __init__(
        self,
        hf_prob: float = 0.5,
        vf_prob: float = 0,
        rr_prob: float = 0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.4,
        cj_hue: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 1.0,
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        solarization_prob: float = 0.2,
        normalize: Union[None, dict] = IMAGENET_NORMALIZE
    ):
        transform = [
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=cj_strength * cj_bright,
                        contrast=cj_strength * cj_contrast,
                        saturation=cj_strength * cj_sat,
                        hue=cj_strength * cj_hue,
                    )
                ],
                p=cj_prob,
            ),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(
                kernel_size=kernel_size,
                scale=kernel_scale,
                sigmas=sigmas,
                prob=gaussian_blur,
            ),
            RandomSolarization(prob=solarization_prob),
            T.ToTensor(),
        ]

        if normalize:
            transform += [T.Normalize(mean=normalize['mean'], std=normalize['std'])]
        self.transform = T.Compose(transform)

    def __call__(self, img: Image) -> Tensor:
        return self.transform(img)