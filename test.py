import argparse
import json
import os
import time

import nibabel as nib
import numpy as np
import torch
import yaml
from tqdm import tqdm

import data
import model
import utils
import wandb

torch.manual_seed(0)
np.random.seed(seed=0)


def test(path=None, epoch=None, parser=None, lr=None, image_path=None, test_id=None, model_avail=None):
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)

    times_tot = {}
    start_time = time.time()
    starter.record()

    try:
        # about GPU
        parser.add_argument('-is_gpu', type=int, default=1, dest='is_gpu',
                            help='enable GPU (1->enable, 0->disenable)')
        # about file
        parser.add_argument('-scale', type=float, default='1.0', dest='scale',
                            help='the up-sampling scale k')
    except:
        print('Arguments already added')

    config_test = parser.parse_args()
    wandb.config.update(config_test)

    decoder_depth = config_test.decoder_depth
    decoder_width = config_test.decoder_width
    feature_dim = config_test.feature_dim
    gpu = config_test.gpu
    is_gpu = config_test.is_gpu
    output_path = config_test.output_path
    test_mr_contrast = config_test.test_mr_contrast
    scale = config_test.scale

    finetune = config_test.finetune

    if isinstance(path, str):
        pre_trained_model = path
    else:
        pre_trained_model = config_test.pre_trained_model

    if not isinstance(epoch, int):
        epoch = int(config_test.pre_trained_model.split('_')[-1].split('.')[0])

    if isinstance(lr, float):
        finetune_lr = lr
    else:
        finetune_lr = config_test.finetune_lr
    if not isinstance(image_path, str):
        input_path = config_test.input_path
    else:
        input_path = image_path

    if not isinstance(test_id, str):
        test_id = config_test.test_id

    print(yaml.dump(config_test, sort_keys=False, default_flow_style=False))

    if is_gpu == 1 and torch.cuda.is_available():
        DEVICE = torch.device('cuda:{}'.format(str(gpu)))
    else:
        DEVICE = torch.device('cpu')

    if isinstance(model_avail, str):
        TripleSR = model.TripleSR(feature_dim=feature_dim,
                                  decoder_depth=int(decoder_depth / 2),
                                  decoder_width=decoder_width).to(DEVICE)

    else:
        TripleSR = model_avail

    print(f'Pretrained model from path: {pre_trained_model}.')
    TripleSR.load_state_dict(torch.load(pre_trained_model, map_location=DEVICE))

    batch_size = 10
    sample_size = 8000

    node_id = int(pre_trained_model.split('_')[-2])
    filenames = [x for x in np.array(os.listdir(input_path)) ]
    print(filenames)

    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time_cuda = (starter.elapsed_time(ender)) / 1000
    curr_time = time.time() - start_time

    times_tot['before finetune dataloader'] = (curr_time_cuda, curr_time)

    start_time = time.time()
    starter.record()

    for i, f in tqdm(enumerate(filenames)):
        '''test-time finetuning phase'''
        finetune_loader = data.loader_finetune(in_path_hr=r'{}/{}'.format(input_path, f), batch_size=batch_size,
                                               sample_size=sample_size, is_train=True)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time_cuda = (starter.elapsed_time(ender)) / 1000
        curr_time = time.time() - start_time
        times_tot['after finetune dataloader'] = (curr_time_cuda, curr_time)
        start_time = time.time()
        starter.record()

        DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))
        optimizer = torch.optim.Adam(params=TripleSR.parameters(), lr=finetune_lr)

        # patient-specific online fine-tuning
        if finetune == 1:
            finetune_epoch = 10
            for e in range(finetune_epoch):
                TripleSR.train()

                for (img_lr_ax1, img_lr_cor1, xyz_hr,xyz_hr_norm, xyz_lr_ax, xyz_lr_cor) in tqdm(
                        finetune_loader):
                    img_lr_ax = img_lr_ax1.unsqueeze(1).to(DEVICE).float()
                    img_lr_cor = img_lr_cor1.unsqueeze(1).to(DEVICE).float()

                    xyz_hr = xyz_hr.to(DEVICE).float()
                    xyz_hr_norm = xyz_hr_norm.to(DEVICE).float()
                    xyz_lr_ax = xyz_lr_ax.to(DEVICE).float()
                    xyz_lr_cor = xyz_lr_cor.to(DEVICE).float()

                    x = img_lr_ax1.shape[2]
                    img_pre = TripleSR(img_lr_ax, img_lr_cor, xyz_hr, xyz_hr_norm, (x, x, x))

                    # compute loss on available LR coordinates
                    radius_iso = 0
                    loss_ax = utils.neighbor_loss(img_lr_ax, img_pre, xyz_lr_ax, xyz_hr, radius_iso)
                    loss_cor = utils.neighbor_loss(img_lr_cor, img_pre, xyz_lr_cor, xyz_hr, radius_iso)

                    loss = (loss_ax + loss_cor) / 2

                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                torch.cuda.empty_cache()

        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time_cuda = (starter.elapsed_time(ender)) / 1000
        curr_time = time.time() - start_time

        times_tot['before test dataloader'] = (curr_time_cuda, curr_time)

        start_time = time.time()
        starter.record()

        '''final testing'''
        test_loader = data.loader_test(in_path_hr=r'{}/{}'.format(input_path, f))

        aff_img = test_loader.dataset.img_hr[i]
        lr_size = aff_img.shape
        aff_matrix = aff_img.affine
        hr_size = (np.array(lr_size) * scale).astype(int)

        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time_cuda = (starter.elapsed_time(ender)) / 1000
        curr_time = time.time() - start_time

        times_tot['dataloading test'] = (curr_time_cuda, curr_time)

        start_time = time.time()
        starter.record()

        TripleSR.eval()
        with torch.no_grad():
            img_pre = np.zeros((hr_size[0] * hr_size[1] * hr_size[2], 1))

            for i, (img_lr_ax, img_lr_cor, xyz_hr, xyz_hr_norm, img_hr, lr_mask) in enumerate(test_loader):
                img_lr_ax = img_lr_ax.unsqueeze(1).float().to(DEVICE)
                img_lr_cor = img_lr_cor.unsqueeze(1).float().to(DEVICE)

                xyz_hr = xyz_hr.view(1, -1, 3).float()
                xyz_hr_norm = xyz_hr_norm.view(1, -1, 3).float()

                b, x, y, z = img_hr.shape
                # predict each HR slice
                for j in tqdm(range(hr_size[0])):
                    xyz_hr_patch = xyz_hr[:, j * hr_size[1] * hr_size[2]:
                                             j * hr_size[1] * hr_size[2] + hr_size[1] * hr_size[2], :].to(DEVICE)
                    xyz_hr_patch_norm = xyz_hr_norm[:, j * hr_size[1] * hr_size[2]:
                                             j * hr_size[1] * hr_size[2] + hr_size[1] * hr_size[2], :].to(DEVICE)

                    img_pre_path = TripleSR(img_lr_ax, img_lr_cor, xyz_hr_patch, xyz_hr_patch_norm, (x, y, z))
                    img_pre[j * hr_size[1] * hr_size[2]:
                            j * hr_size[1] * hr_size[2] + hr_size[1] * hr_size[2]] = \
                        img_pre_path.cpu().detach().numpy().reshape(hr_size[1] * hr_size[2], 1)
                img_pre = img_pre.reshape((hr_size[0], hr_size[1], hr_size[2]))

        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time_cuda = (starter.elapsed_time(ender)) / 1000
        curr_time = time.time() - start_time

        times_tot['testing'] = (curr_time_cuda, curr_time)

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        with open(
                os.path.join(output_path, f'ours_times_{test_id}_{node_id}_{scale}_{test_mr_contrast}_test.txt'),
                'w') as file:
            file.write(json.dumps(times_tot))

        nib.save(nib.Nifti1Image(img_pre, aff_matrix), rf'{output_path}/tripleSR_recon_{node_id}_{f}')

        img_pre = (img_pre - np.min(img_pre)) / (np.max(img_pre) - np.min(img_pre))

        images = wandb.Image(torch.Tensor(img_pre[hr_size[0] // 2]), caption="Pred")

        wandb.log({"Prediction" + test_id: images}, step=epoch)

        images = wandb.Image(img_hr[0, hr_size[0] // 2], caption="GT")

        wandb.log({"GT" + test_id: images}, step=epoch)
        break


if __name__ == '__main__':
    run = wandb.init(
        project="multi_view_SR",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('-decoder_depth', type=int, default=8, dest='decoder_depth',
                        help='the depth of the decoder network (default=8).')
    parser.add_argument('-decoder_width', type=int, default=512, dest='decoder_width',
                        help='the width of the decoder network (default=256).')
    parser.add_argument('-feature_dim', type=int, default=128, dest='feature_dim',
                        help='the dimension size of the feature vector (default=128)')
    parser.add_argument('-gpu', type=int, default=0, dest='gpu',
                        help='the number of GPU')
    parser.add_argument('-finetune_lr', type=float, default=1e-4, dest='finetune_lr',
                        help='the initial learning rate')
    parser.add_argument('-input_path', type=str, default='./data/test_4/', #TODO
                        help='the file path of LR input image')
    parser.add_argument('-test_id', type=str, default='BraTS19_CBICA_AYI_1',
                        help='ID to test')
    parser.add_argument('-pre_trained_model', type=str, default='./model/model_sr.pkl',
                        dest='pre_trained_model', help='the file path of LR input image for testing')
    parser.add_argument('-output_path', type=str, default='../data/test/out', dest='output_path',
                        help='the file save path of reconstructed result')
    parser.add_argument('-test_mr_contrast', type=str, default='t1', nargs='?', const='',
                        help='choose: t1|t2|t1ce|all')
    parser.add_argument('-finetune', type=int, default=0, help='if true finetuning before final testing')

    test(parser=parser, model_avail='test')
    run.finish()
