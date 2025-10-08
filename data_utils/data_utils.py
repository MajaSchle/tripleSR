import nibabel as nib
import numpy as np
import torch


def get_image_patches(patch_coords, patch_size, patch_size_factor, zms, hr_orig, img_lr_ax, img_lr_cor):
    x0, y0, z0 = patch_coords
    patch_hr = hr_orig.slicer[x0:x0 + patch_size, y0:y0 + patch_size, z0:z0 + patch_size]

    # catch edge cases that are at image edge
    if (int(np.round((z0 + patch_size) // zms))) >= img_lr_ax.shape[2]:
        patch_lr_img_ax = img_lr_ax.slicer[x0:x0 + patch_size, y0:y0 + patch_size,
                          (img_lr_ax.shape[2]) - patch_size_factor:img_lr_ax.shape[2]]
    else:
        patch_lr_img_ax = img_lr_ax.slicer[x0:x0 + patch_size, y0:y0 + patch_size,
                          int(np.round(z0 // zms)):int(np.round(z0 // zms)) + patch_size_factor]

    lr_aff = patch_lr_img_ax.affine
    lr_aff[:, 3] = patch_hr.affine[:, 3]
    patch_lr_img_ax = patch_lr_img_ax.get_fdata()
    patch_lr_img_ax = nib.Nifti1Image(patch_lr_img_ax, lr_aff)

    # catch edge cases that are at image edge
    if (int(np.round((y0 + patch_size) // zms))) >= img_lr_cor.shape[1]:
        patch_lr_img_cor = img_lr_cor.slicer[x0:x0 + patch_size,
                           (img_lr_cor.shape[1]) - patch_size_factor:img_lr_cor.shape[1], z0:z0 + patch_size]
    else:
        patch_lr_img_cor = img_lr_cor.slicer[x0:x0 + patch_size,
                           int(np.round(y0 // zms)):int(np.round(y0 // zms)) + patch_size_factor,
                           z0:z0 + patch_size]

    lr_aff = patch_lr_img_cor.affine
    lr_aff[:, 3] = patch_hr.affine[:, 3]
    patch_lr_img_cor = patch_lr_img_cor.get_fdata()
    patch_lr_img_cor = nib.Nifti1Image(patch_lr_img_cor, lr_aff)

    return patch_hr, patch_lr_img_ax, patch_lr_img_cor


def get_coords_normed_to_ref(img, hr_img):
    img_affine = img.affine
    (x_hr, y_hr, z_hr) = img.shape
    h, w, d = hr_img.shape

    X = np.linspace(0, x_hr - 1, x_hr)
    Y = np.linspace(0, y_hr - 1, y_hr)
    Z = np.linspace(0, z_hr - 1, z_hr)
    points = np.meshgrid(X, Y, Z, indexing='ij')
    points = np.stack(points).transpose(1, 2, 3, 0).reshape(-1, 3)
    coordinates = list(nib.affines.apply_affine(img_affine, points))
    coordinates_arr = np.array(coordinates, dtype=np.float32)

    def min_max_scale(X, s_min, s_max):
        x_min_x, x_max_x, x_min_y, x_max_y, x_min_z, x_max_z = 0, h, 0, w, 0, d
        X[:, 0] = (X[:, 0] - x_min_x) / (x_max_x - x_min_x)
        X[:, 1] = (X[:, 1] - x_min_y) / (x_max_y - x_min_y)
        X[:, 2] = (X[:, 2] - x_min_z) / (x_max_z - x_min_z)
        return X * (s_max - s_min) + s_min

    xyz_hr = torch.Tensor(min_max_scale(X=coordinates_arr, s_min=-1, s_max=1))

    return xyz_hr

def get_coords(image):
    patch_hr_img = image.get_fdata()

    Affine_Mat_w = [1, 0, 0, 0]
    Affine_Mat_h = [0, 1, 0, 0]
    Affine_Mat_d = [0, 0, 1, 0]
    Aff_last = [0, 0, 0, 1]

    affine = np.c_[Affine_Mat_w, Affine_Mat_h, Affine_Mat_d, Aff_last].T
    hr_vol_norm = nib.Nifti1Image(patch_hr_img, affine)

    # generate coordinate set for pixel coordinates
    img_affine = hr_vol_norm.affine
    (x, y, z) = hr_vol_norm.shape
    X = np.linspace(0, x - 1, x)
    Y = np.linspace(0, y - 1, y)
    Z = np.linspace(0, z - 1, z)
    points = np.meshgrid(X, Y, Z, indexing='ij')
    points = np.stack(points).transpose(1, 2, 3, 0).reshape(-1, 3)
    coordinates = list(nib.affines.apply_affine(img_affine, points))
    coordinates_arr = np.array(coordinates, dtype=np.float32)

    xyz_hr_pix = torch.Tensor(coordinates_arr)
    return xyz_hr_pix