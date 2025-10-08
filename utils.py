import nibabel as nib
import numpy as np
import torch


def patch_img_with_lr(in_path, lr_ax, lr_cor):
    patches_hr = []
    patches_ax = []
    patches_cor = []

    img_in = nib.load(in_path)
    aff = img_in.affine
    img_in = img_in.get_fdata()
    img_in = nib.Nifti1Image(img_in, aff)

    img_lr_ax = nib.load(lr_ax)
    img_lr_cor = nib.load(lr_cor)

    zms = img_lr_ax.header.get_zooms()[2]
    patch_size = int(np.round(zms * 10))

    h, w, d = img_in.shape

    for x0 in range(0, h - patch_size, patch_size):
        for y0 in range(0, w - patch_size, patch_size):
            for z0 in range(0, d - patch_size, patch_size):

                patch_hr = img_in.slicer[x0:x0 + patch_size, y0:y0 + patch_size, z0:z0 + patch_size]

                if (int(np.round((z0 + patch_size) // zms))) >= img_lr_ax.shape[2]:
                    patch_lr_ax = img_lr_ax.slicer[x0:x0 + patch_size, y0:y0 + patch_size,
                                  (img_lr_ax.shape[2]) - 10:img_lr_ax.shape[2]]
                else:
                    patch_lr_ax = img_lr_ax.slicer[x0:x0 + patch_size, y0:y0 + patch_size,
                                  int(np.round(z0 // zms)):int(np.round(z0 // zms)) + 10]

                lr_aff = patch_lr_ax.affine
                lr_aff[:, 3] = patch_hr.affine[:, 3]
                patch_lr_img_ax = patch_lr_ax.get_fdata()
                patch_lr_ax = nib.Nifti1Image(patch_lr_img_ax, lr_aff)

                if (int(np.round((y0 + patch_size) // zms))) >= img_lr_cor.shape[1]:
                    patch_lr_cor = img_lr_cor.slicer[x0:x0 + patch_size, (img_lr_cor.shape[1]) - 10:img_lr_cor.shape[1],
                                   z0:z0 + patch_size]
                else:
                    patch_lr_cor = img_lr_cor.slicer[x0:x0 + patch_size,
                                   int(np.round(y0 // zms)):int(np.round(y0 // zms)) + 10,
                                   z0:z0 + patch_size]

                lr_aff = patch_lr_cor.affine
                lr_aff[:, 3] = patch_hr.affine[:, 3]
                patch_lr_img_cor = patch_lr_cor.get_fdata()
                patch_lr_cor = nib.Nifti1Image(patch_lr_img_cor, lr_aff)

                patches_hr.append(patch_hr)
                patches_ax.append(patch_lr_ax)
                patches_cor.append(patch_lr_cor)
    return patches_hr, patches_ax, patches_cor


def unison_shuffled_copies_batched(a, b, c, batch_size=1):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(np.arange(0, len(a), batch_size))
    p1 = np.array([x + i for x in p for i in range(batch_size)])
    return a[p1], b[p1], c[p1]


def neighbor_loss(img_lr, img_pre, xyz_lr, xyz_hr, radius):
    # compute loss of matching LR - HR coordinates

    img_lr_flat = img_lr.reshape(len(img_lr), -1)  # (B, N_lr) flatten
    img_pre_flat = img_pre.squeeze(-1)  # (B, N_hr)

    dists = torch.cdist(xyz_hr, xyz_lr)  # (b, S, #LR coords)
    mask = dists <= radius
    M = mask.float() # (b, S, # LR coords)

    # pairwise error for every (HR_coord, LR_coord)
    diff2 = (img_pre_flat.unsqueeze(-1) - img_lr_flat.unsqueeze(1)).pow(2) # (b, S, # LR coords)

    # average over matched pairs with distance <= radius
    pair_counts = M.sum(dim=(1, 2)).clamp(min=1)
    loss_b = (diff2 * M).sum(dim=(1, 2)) / pair_counts
    loss = loss_b.mean()

    return loss
