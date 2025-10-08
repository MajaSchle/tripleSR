import argparse
import json
import os
import time

import torch
import yaml
from tqdm import tqdm

import data
import model
import utils
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

wandb.login()


def main():
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    times_tot = {}
    start_time = time.time()
    starter.record()

    '''
    ### Define parameters
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-decoder_depth', type=int, default=8, dest='decoder_depth',
                        help='the depth of the decoder network (default=8).')
    parser.add_argument('-decoder_width', type=int, default=512, dest='decoder_width',
                        help='the width of the decoder network (default=256).')
    parser.add_argument('-feature_dim', type=int, default=128, dest='feature_dim',
                        help='Dimension of feature vector F.')

    # about training and validation data
    parser.add_argument('-hr_data_train', type=str, default='./data/train', #TODO
                        help='the file path of HR patches for training')
    parser.add_argument('-hr_data_val', type=str, default='./data/val', #TODO
                        help='the file path of HR patches for validation')
    parser.add_argument('-hr_data_test', type=str, default='./data/test_4', #TODO
                        help='the file path of HR patches for validation')
    parser.add_argument('-output_path', type=str, default='../data/test/out', dest='output_path',
                        help='the file save path of reconstructed result')

    # training and inference parameters
    parser.add_argument('-subset', type=int, default=100, help='How many images used for training')
    parser.add_argument('-finetune', type=int, default=1, help='if true finetuning before final testing')

    # selected MR contrast
    parser.add_argument('-mr_contrast', type=str, default='t1', nargs='?', const='', help='choose: t1|t2|t1ce|all')
    parser.add_argument('-test_mr_contrast', type=str, default='t1', nargs='?', const='', help='choose: t1|t2|t1ce|all')

    # about training hyper-parameters
    parser.add_argument('-lr', type=float, default=1e-4, dest='lr',
                        help='the initial learning rate')
    parser.add_argument('-lr_decay_epoch', type=int, default=200, dest='lr_decay_epoch',
                        help='learning rate multiply by 0.5 per lr_decay_epoch .')
    parser.add_argument('-epoch', type=int, default=35, dest='epoch',
                        help='the total number of epochs for training')
    parser.add_argument('-summary_epoch', type=int, default=1, dest='summary_epoch',
                        help='the current model will be saved per summary_epoch')
    parser.add_argument('-bs', type=int, default=10, dest='batch_size',
                        help='the number of LR-HR patch pairs (i.e., N in Equ. 3)')
    parser.add_argument('-ss', type=int, default=8000, dest='sample_size',
                        help='the number of sampled voxel coordinates (i.e., K in Equ. 3)')
    parser.add_argument('-gpu', type=int, default=0, dest='gpu',
                        help='the number of GPU')
    parser.add_argument('-node_id', type=int, default=0, help='ID of cluster node')

    config = parser.parse_args()
    decoder_depth = config.decoder_depth
    decoder_width = config.decoder_width
    feature_dim = config.feature_dim
    hr_data_train = config.hr_data_train
    hr_data_val = config.hr_data_val
    lr = config.lr
    lr_decay_epoch = config.lr_decay_epoch
    epoch = config.epoch
    summary_epoch = config.summary_epoch
    batch_size = config.batch_size
    sample_size = config.sample_size
    gpu = config.gpu
    node_id = config.node_id
    test_image_path = config.hr_data_test
    mr_contrast = config.mr_contrast
    subset = config.subset

    print(yaml.dump(config, sort_keys=False, default_flow_style=False))

    run = wandb.init(
        project="multi_view_SR",
        tags=['grid sample'],
        config={
            "learning_rate": lr,
            "epochs": epoch,
        },
    )
    wandb.config.update(config)

    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time_cuda = (starter.elapsed_time(ender)) / 1000
    curr_time = time.time() - start_time

    times_tot['before dataset'] = (curr_time_cuda, curr_time)

    start_time = time.time()
    starter.record()

    train_loader = data.loader_train(in_path_hr=hr_data_train, batch_size=batch_size, subset=subset,
                                     sample_size=sample_size, is_train=True, mr_contrast=mr_contrast)
    val_loader = data.loader_train(in_path_hr=hr_data_val, batch_size=1,
                                   sample_size=sample_size, is_train=True, mr_contrast=mr_contrast)

    ender.record()

    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time_cuda = (starter.elapsed_time(ender)) / 1000
    curr_time = time.time() - start_time
    print(curr_time)
    times_tot['dataloading train'] = (curr_time_cuda, curr_time)
    start_time = time.time()
    starter.record()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TripleSR = model.TripleSR(feature_dim=feature_dim, decoder_depth=int(decoder_depth / 2),
                              decoder_width=decoder_width).to(DEVICE)

    optimizer = torch.optim.Adam(params=TripleSR.parameters(), lr=lr)

    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time_cuda = (starter.elapsed_time(ender)) / 1000
    curr_time = time.time() - start_time

    times_tot['before training'] = (curr_time_cuda, curr_time)

    start_time = time.time()
    starter.record()

    for e in range(epoch):
        # change order of images but keep batches (to match patch shapes)
        a, b, c = utils.unison_shuffled_copies_batched(train_loader.dataset.patch_hr, train_loader.dataset.img_lr_ax,
                                                       train_loader.dataset.img_lr_cor, batch_size=batch_size)
        train_loader.dataset.patch_hr = a
        train_loader.dataset.img_lr_ax = b
        train_loader.dataset.img_lr_cor = c
        TripleSR.train()
        loss_train = 0

        for i, (img_lr_ax1, img_lr_cor1, xyz_hr, xyz_hr_pix, xyz_lr_ax, xyz_lr_cor) in tqdm(enumerate(train_loader)):
            # LR image patches
            img_lr_ax = img_lr_ax1.unsqueeze(1).to(DEVICE).float()
            img_lr_cor = img_lr_cor1.unsqueeze(1).to(DEVICE).float()  # N×1×h×w×d

            # coordinates of HR image patch
            xyz_hr = xyz_hr.view(batch_size, -1, 3).to(DEVICE).float()  # N×K×3
            xyz_hr_pix = xyz_hr_pix.view(batch_size, -1, 3).to(DEVICE).float()  # N×K×3

            # coordinates of LR image patches
            xyz_lr_ax = xyz_lr_ax.view(batch_size, -1, 3).to(DEVICE).float()  # N×K×3
            xyz_lr_cor = xyz_lr_cor.view(batch_size, -1, 3).to(DEVICE).float()  # N×K×3

            # LR and HR shape
            b, x, y, z = img_lr_ax1.shape
            interp_shape = (x, x, x)

            # to model
            img_pre = TripleSR(img_lr_ax, img_lr_cor, xyz_hr, xyz_hr_pix, interp_shape)  # N×K×1

            # compute loss on available LR coordinates
            # radius defines on the distance of allowed LR - HR coordiante pairs
            radius_iso = 0

            # compute loss on axial and coronal coordinates
            loss_ax = utils.neighbor_loss(img_lr_ax, img_pre, xyz_lr_ax, xyz_hr, radius_iso)
            loss_cor = utils.neighbor_loss(img_lr_cor, img_pre, xyz_lr_cor, xyz_hr, radius_iso)

            loss = (loss_ax + loss_cor) / 2

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record and print loss
            loss_train += loss.item()

            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(
                f'(TRAIN) Epoch[{e + 1}/{epoch}], Steps[{i + 1}/{len(train_loader)}], Lr:{current_lr}, Loss:{loss.item():.10f}')
            torch.cuda.empty_cache()

        wandb.log({"MES_train": loss_train / len(train_loader)}, step=e + 1)

        TripleSR.eval()
        with torch.no_grad():
            loss_val = 0
            for i, (img_lr_ax, img_lr_cor, xyz_hr, xyz_hr_pix, xyz_lr_ax, xyz_lr_cor) in tqdm(enumerate(val_loader)):
                img_lr_ax = img_lr_ax.unsqueeze(1).to(DEVICE).float()  # N×1×h×w×d
                img_lr_cor = img_lr_cor.unsqueeze(1).to(DEVICE).float()  # N×1×h×w×d

                xyz_hr = xyz_hr.view(1, -1, 3).to(DEVICE).float()  # N×Q×3 (Q=H×W×D)
                xyz_hr_pix = xyz_hr_pix.view(1, -1, 3).to(DEVICE).float()  # N×Q×3 (Q=H×W×D)

                xyz_lr_ax = xyz_lr_ax.view(1, -1, 3).to(DEVICE).float()  # N×K×3
                xyz_lr_cor = xyz_lr_cor.view(1, -1, 3).to(DEVICE).float()  # N×K×3

                H = img_lr_ax.shape[2]
                img_pre = TripleSR(img_lr_ax, img_lr_cor, xyz_hr, xyz_hr_pix, (H, H, H))  # N×Q×1 (Q=H×W×D)

                # compute loss on available LR coordinates
                radius_iso = 0

                loss_ax = utils.neighbor_loss(img_lr_ax, img_pre, xyz_lr_ax, xyz_hr, radius_iso)
                loss_cor = utils.neighbor_loss(img_lr_cor, img_pre, xyz_lr_cor, xyz_hr, radius_iso)

                loss_val = (loss_ax + loss_cor) / 2

            # save validation
            if (e + 1) % summary_epoch == 0:
                # save model
                torch.save(TripleSR.state_dict(), f'./model/model_sr_{node_id}_{(e + 1)}.pkl')

        wandb.log({"MES_val": loss_val / len(val_loader)}, step=e + 1)

        # learning rate decays by half every some epochs.
        if (e + 1) % lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        torch.cuda.empty_cache()

    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    curr_time_cuda = (starter.elapsed_time(ender)) / 1000
    curr_time = time.time() - start_time
    times_tot['training'] = (curr_time_cuda, curr_time)

    with open(os.path.join(test_image_path, f'ours_times_{node_id}_train.txt'), 'w') as file:
        file.write(json.dumps(times_tot))


if __name__ == '__main__':
    main()
