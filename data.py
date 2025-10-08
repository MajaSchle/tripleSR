import glob

import nibabel as nib
import numpy as np
from torch.utils import data

import utils
from data_utils.data_utils import get_image_patches, get_coords_normed_to_ref, get_coords


class ImgTrain(data.Dataset):
    def __init__(self, in_path_hr, sample_size, is_train, bs, subset=100, mr_contrast='t1'):
        self.is_train = is_train
        self.sample_size = sample_size

        self.patch_hr = [x for x in glob.glob(in_path_hr + '/**') if mr_contrast + '_full' in x]

        # select if a subset is used (100 is max. available training data)
        self.patch_hr = self.patch_hr[:subset]

        # repeat for batch size: so that all elements in batch have same LR / HR dimensions
        self.patch_hr = np.repeat(self.patch_hr, bs)
        self.img_lr_ax = np.array([x.replace('full', 'LR_ax') for x in self.patch_hr])
        self.img_lr_cor = np.array([x.replace('full', 'LR_cor') for x in self.patch_hr])

    def __len__(self):
        return len(self.patch_hr)

    def __getitem__(self, item):
        # load LR and HR images
        hr_path = self.patch_hr[item]
        hr_orig = nib.load(hr_path)
        img_lr_ax_path = self.img_lr_ax[item]
        img_lr_ax = nib.load(img_lr_ax_path)
        img_lr_cor_path = self.img_lr_cor[item]
        img_lr_cor = nib.load(img_lr_cor_path)

        # get resolution of LR image
        zms = img_lr_ax.header.get_zooms()[2]

        # set the patch size (in mm)
        patch_size_mm = 10

        # get the pixel size for the corresponding resolution
        patch_size_hr_px = int(np.round(patch_size_mm * zms))

        # select random patch
        h, w, d = hr_orig.shape
        x0 = np.random.randint(0, h - patch_size_hr_px)
        y0 = np.random.randint(0, w - patch_size_hr_px)
        z0 = np.random.randint(0, d - patch_size_hr_px)
        patch_coords = x0, y0, z0

        # get image patches (HR and LR)
        patch_hr, patch_lr_img_ax, patch_lr_img_cor = get_image_patches(patch_coords, patch_size_hr_px,
                                                                        patch_size_mm, zms, hr_orig,
                                                                        img_lr_ax, img_lr_cor)

        # get coordinates for image patches
        xyz_hr = get_coords_normed_to_ref(patch_hr, hr_orig)
        xyz_lr_ax = get_coords_normed_to_ref(patch_lr_img_ax, hr_orig)
        xyz_lr_cor = get_coords_normed_to_ref(patch_lr_img_cor, hr_orig)
        xyz_hr_pix = get_coords(patch_hr)

        # get_coords_normed_to_ref(patch_hr, patch_hr)
        # randomly sample voxel coordinates
        if self.is_train:
            sample_indices = np.random.choice(len(xyz_hr), self.sample_size, replace=False)
            xyz_hr = xyz_hr[sample_indices]
            xyz_hr_pix = xyz_hr_pix[sample_indices]

        return patch_lr_img_ax.get_fdata(), patch_lr_img_cor.get_fdata(), xyz_hr, xyz_hr_pix, xyz_lr_ax, xyz_lr_cor


def loader_train(in_path_hr, batch_size, sample_size, is_train, subset=100, mr_contrast='t1'):
    dataset = ImgTrain(in_path_hr=in_path_hr, sample_size=sample_size, is_train=is_train, bs=batch_size, subset=subset,
                       mr_contrast=mr_contrast)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )


class ImgFinetune(data.Dataset):
    def __init__(self, in_path_hr, sample_size, is_train, bs):
        self.is_train = is_train
        self.sample_size = sample_size

        self.patch_hr, self.patches_ax, self.patches_cor = utils.patch_img_with_lr(in_path=in_path_hr,
                                                                                   lr_ax=in_path_hr.replace('full',
                                                                                                            'LR_ax'),
                                                                                   lr_cor=in_path_hr.replace('full',
                                                                                                             'LR_cor'))

    def __len__(self):
        return len(self.patch_hr)

    def __getitem__(self, item):
        patch_hr = self.patch_hr[item]
        patch_lr_img_ax = self.patches_ax[item]
        patch_lr_img_cor = self.patches_cor[item]

        # get coordinates for image patches
        xyz_hr = get_coords_normed_to_ref(patch_hr, patch_hr)
        xyz_lr_ax = get_coords_normed_to_ref(patch_lr_img_ax, patch_hr)
        xyz_lr_cor = get_coords_normed_to_ref(patch_lr_img_cor, patch_hr)
        xyz_hr_pix = get_coords(patch_hr)

        # randomly sample voxel coordinates
        if self.is_train:
            sample_indices = np.random.choice(len(xyz_hr), self.sample_size, replace=False)
            xyz_hr = xyz_hr[sample_indices]
            xyz_hr_pix = xyz_hr_pix[sample_indices]

        return patch_lr_img_ax.get_fdata(), patch_lr_img_cor.get_fdata(), xyz_hr, xyz_hr_pix, xyz_lr_ax, xyz_lr_cor


def loader_finetune(in_path_hr, batch_size, sample_size, is_train):
    dataset = ImgFinetune(in_path_hr=in_path_hr, sample_size=sample_size, is_train=is_train, bs=batch_size)

    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )


class ImgTest(data.Dataset):
    def __init__(self, in_path_hr):
        self.img_hr = []
        self.img_hr_norm = []
        self.hr_masks = []
        self.xyz_hr = []
        self.img_lr_ax = []
        self.img_lr_cor = []

        # load lr image
        hr_img = nib.load(in_path_hr)
        self.img_hr.append(hr_img)

        hr_mask = nib.load(in_path_hr.replace('_full', '_full_brainmask')).get_fdata()
        self.hr_masks.append(hr_mask)

        # load lr views
        lr_ax_img = nib.load(in_path_hr.replace('_full', '_LR_ax'))
        self.img_lr_ax.append(lr_ax_img)

        lr_cor_img = nib.load(in_path_hr.replace('_full', '_LR_cor'))
        self.img_lr_cor.append(lr_cor_img)

    def __len__(self):
        return len(self.img_hr)

    def __getitem__(self, item):
        patch_hr = self.img_hr[item]

        patch_lr_img_ax = self.img_lr_ax[item]
        patch_lr_img_cor = self.img_lr_cor[item]

        hr_mask = self.hr_masks[item]
        patch_hr_img = patch_hr.get_fdata()

        xyz_hr = get_coords_normed_to_ref(patch_hr, patch_hr)
        xyz_hr_pix = get_coords(patch_hr)

        return patch_lr_img_ax.get_fdata(), patch_lr_img_cor.get_fdata(), xyz_hr, xyz_hr_pix, patch_hr_img, hr_mask


def loader_test(in_path_hr):
    return data.DataLoader(
        dataset=ImgTest(in_path_hr=in_path_hr),
        batch_size=1,
        shuffle=False
    )
