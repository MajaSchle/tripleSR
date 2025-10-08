import decoder
import encoder
import torch
from torch import nn

class TripleSR(nn.Module):
    def __init__(self, feature_dim, decoder_depth, decoder_width):
        super(TripleSR, self).__init__()
        self.encoder = encoder.RDN(feature_dim=feature_dim)
        self.decoder = decoder.MLP(in_dim=feature_dim*2, out_dim=1, depth=decoder_depth, width=decoder_width)


    def forward(self, img_lr_ax, img_lr_cor, xyz_hr, xyz_hr_pix, shape):
        # extract feature map from LR image
        feature_map_ax = self.encoder(img_lr_ax)  # b×F×h_lr×w_lr×d_lr
        feature_map_cor = self.encoder(img_lr_cor)  # b×F_lr×h_lr×w×d_lr


        resize_to = torch.nn.Upsample(size=shape, mode='trilinear')
        feature_vector_ax1 = torch.moveaxis(resize_to(feature_map_ax), 1, -1)
        feature_vector_cor1 = torch.moveaxis(resize_to(feature_map_cor), 1, -1)

        def get_intensities_at_coordinates(features, coords):
            coords = coords.to(torch.long)
            B, S, _ = coords.shape
            b = torch.arange(B, device=features.device).view(B, 1).expand(B, S)
            return features[b, coords[..., 0], coords[..., 1], coords[..., 2], :]

        feature_vector_ax = get_intensities_at_coordinates(feature_vector_ax1, xyz_hr_pix) # (b, S, F)
        feature_vector_cor = get_intensities_at_coordinates(feature_vector_cor1, xyz_hr_pix)# (b, S, F)

        # concatenate coordinate with feature vector
        feature_vector_combined = torch.cat([feature_vector_ax, feature_vector_cor], dim=-1) # (b, S, 2*F)

        # get prediction for coordinate
        N, K = xyz_hr.shape[:2]
        intensity_pre = self.decoder(feature_vector_combined.view(N * K, -1)).view(N, K, -1) #(b, S, 1)

        return intensity_pre
