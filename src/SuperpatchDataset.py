import random
import PIL
import torchvision.transforms as transforms
from torch.utils.data import Dataset


# TODO: docstring and review
class SuperpatchDataset(Dataset):
  def __init__(self, paths: list, spatch_nns: dict,
        input_height: int, input_width: int, patch_size: int, n_agg: int,
        img_transform: transforms.Compose, max_nearest: int = 8,
        coord_jitter: int = 0, return_whole_images: bool = False):
    assert input_height % patch_size == 0
    assert input_width % patch_size == 0
    assert input_height % (n_agg*patch_size) == 0
    assert input_width % (n_agg*patch_size) == 0

    self.paths = paths
    self.spatch_nns = spatch_nns

    self.n_spatches_row = input_height // patch_size // n_agg 
    self.n_spatches_col = input_width // patch_size // n_agg 
    self.n_spatches_per_img = self.n_spatches_row * self.n_spatches_col
    self.patch_size = patch_size
    self.n_agg = n_agg

    self.img_transform = img_transform
    
    self.max_nearest = max_nearest
    self.coord_jitter = coord_jitter
    self.return_whole_images = return_whole_images

  def __getitem__(self, img_idx):
    spatch1_idx = self.sample_spatch_idx_from_img_idx(img_idx)
    spatch2_idx = self.sample_spatch_idx_from_spatch_idx(spatch1_idx)

    spatch1_coords = self.get_spatch_coords(spatch1_idx)
    spatch2_coords = self.get_spatch_coords(spatch2_idx)

    img1_idx = spatch1_idx // self.n_spatches_per_img
    img2_idx = spatch2_idx // self.n_spatches_per_img

    img1, spatch1 = self.get_spatch_array(img1_idx, spatch1_coords)
    img2, spatch2 = self.get_spatch_array(img2_idx, spatch2_coords)

    if self.return_whole_images:
        return img1, spatch1, img2, spatch2
    else:
        return spatch1, spatch2

  def __len__(self):
    return len(self.paths)
  
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
    img = self.img_transform(img)
    y_upper, x_upper, y_lower, x_lower = spatch_coords

    return img, img[:, y_upper : y_lower, x_upper : x_lower]