from typing import Tuple, List, Optional, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from bbox import BBox
from extension.functional import beta_smooth_l1_loss
from support.layer.nms import nms

'''
RegionProposalNetwork
Args:
    num_features_out: in_channels of feature extraction
    anchor_ratios: [(1, 2), (1, 1), (2, 1)](default)
    anchor_sizes: [128, 256, 512](default)
    rpn_pre_nms_top_n: number of roi to keep pre nms
    rpn_post_nms_top_n: number of roi to keep after nms
    anchor_smooth_l1_loss_beta: Beta for rpn
'''
class RegionProposalNetwork(nn.Module):

    def __init__(self, num_features_out: int, anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int],
                 pre_nms_top_n: int, post_nms_top_n: int, anchor_smooth_l1_loss_beta: float):
        super().__init__()

        self._features = nn.Sequential(
            nn.Conv2d(in_channels=num_features_out, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self._anchor_ratios = anchor_ratios
        self._anchor_sizes = anchor_sizes

        num_anchor_ratios = len(self._anchor_ratios)
        num_anchor_sizes = len(self._anchor_sizes)
        num_anchors = num_anchor_ratios * num_anchor_sizes

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self._anchor_smooth_l1_loss_beta = anchor_smooth_l1_loss_beta

        self._anchor_objectness = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self._anchor_transformer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)

    def forward(self, features: Tensor,
                anchor_bboxes: Optional[Tensor] = None, gt_bboxes_batch: Optional[Tensor] = None,
                image_width: Optional[int]=None, image_height: Optional[int]=None) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        batch_size = features.shape[0]
        # forward convolution
        features = self._features(features)
        anchor_objectnesses = self._anchor_objectness(features)
        anchor_transformers = self._anchor_transformer(features)

        anchor_objectnesses = anchor_objectnesses.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        anchor_transformers = anchor_transformers.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        if not self.training:
            return anchor_objectnesses, anchor_transformers
        else:
            # return len(inside_indices) of the inside bbox(value: bool)
            inside_indices = BBox.inside(anchor_bboxes, left=0, top=0, right=image_width, bottom=image_height).nonzero().unbind(dim=1)
            # inside_anchor_bboxes: [batch_size,K(num of inside anchors),4]
            inside_anchor_bboxes = anchor_bboxes[inside_indices].view(batch_size, -1, anchor_bboxes.shape[2])

            # inside_anchor_bboxes: [batch_size,K(num of inside anchors),2]
            # inside_anchor_transformers: [batch_size,K(num of inside anchors),4]
            inside_anchor_objectnesses = anchor_objectnesses[inside_indices].view(batch_size, -1, anchor_objectnesses.shape[2])
            inside_anchor_transformers = anchor_transformers[inside_indices].view(batch_size, -1, anchor_transformers.shape[2])

            # find labels for each `anchor_bboxes`
            # labels: [batch_size,K(num of inside anchors)]
            labels = torch.full((batch_size, inside_anchor_bboxes.shape[1]), -1, dtype=torch.long, device=inside_anchor_bboxes.device)
            # ious: [[batch_size,K(num of inside anchors),N(num of gt_bboxes)]
            ious = BBox.iou(inside_anchor_bboxes, gt_bboxes_batch)
            # get max iou and its indices of each anchor(dim=2)
            anchor_max_ious, anchor_assignments = ious.max(dim=2)
            # get max iou and its indices of each gt_bbox(dim=1)
            gt_max_ious, gt_assignments = ious.max(dim=1)

            # those which reach gt_max_iou but may lower then 0.3
            anchor_additions = ((ious > 0) & (ious == gt_max_ious.unsqueeze(dim=1))).nonzero()[:, :2].unbind(dim=1)
            # threshold
            # note that labels are initialed as -1, so there remains some -1 as 0.3<=anchor_max_ious<0.7 and not in anchor_additions
            labels[anchor_max_ious < 0.3] = 0
            labels[anchor_additions] = 1
            labels[anchor_max_ious >= 0.7] = 1

            # select 256 x `batch_size` samples
            # fg_indices： [N(num of nonzero element),2] value：(which batch,which index)(matrix indices)
            # bg_indices： [N(num of nonzero element),2] value：(which batch,which index)(matrix indices)
            fg_indices = (labels == 1).nonzero()
            bg_indices = (labels == 0).nonzero()

            fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 128 * batch_size)]]
            bg_indices = bg_indices[torch.randperm(len(bg_indices))[:256 * batch_size - len(fg_indices)]]
            selected_indices = torch.cat([fg_indices, bg_indices], dim=0)
            selected_indices = selected_indices[torch.randperm(len(selected_indices))].unbind(dim=1)

            inside_anchor_bboxes = inside_anchor_bboxes[selected_indices]
            gt_bboxes = gt_bboxes_batch[selected_indices[0], anchor_assignments[selected_indices]]
            gt_anchor_objectnesses = labels[selected_indices]
            gt_anchor_transformers = BBox.calc_transformer(inside_anchor_bboxes, gt_bboxes)

            batch_indices = selected_indices[0]

            # Loss Calculation
            anchor_objectness_losses, anchor_transformer_losses = self.loss(inside_anchor_objectnesses[selected_indices],
                                                                            inside_anchor_transformers[selected_indices],
                                                                            gt_anchor_objectnesses,
                                                                            gt_anchor_transformers,
                                                                            batch_size, batch_indices)

            return anchor_objectnesses, anchor_transformers, anchor_objectness_losses, anchor_transformer_losses

    def loss(self, anchor_objectnesses: Tensor, anchor_transformers: Tensor,
             gt_anchor_objectnesses: Tensor, gt_anchor_transformers: Tensor,
             batch_size: int, batch_indices: Tensor) -> Tuple[Tensor, Tensor]:
        cross_entropies = torch.empty(batch_size, dtype=torch.float, device=anchor_objectnesses.device)
        # Using smooth_l1_losses so we can avoid outliers to dominant loss function
        # details： https://zhuanlan.zhihu.com/p/48426076
        smooth_l1_losses = torch.empty(batch_size, dtype=torch.float, device=anchor_transformers.device)

        for batch_index in range(batch_size):
            selected_indices = (batch_indices == batch_index).nonzero().view(-1)

            cross_entropy = F.cross_entropy(input=anchor_objectnesses[selected_indices],
                                            target=gt_anchor_objectnesses[selected_indices])

            fg_indices = gt_anchor_objectnesses[selected_indices].nonzero().view(-1)
            smooth_l1_loss = beta_smooth_l1_loss(input=anchor_transformers[selected_indices][fg_indices],
                                                 target=gt_anchor_transformers[selected_indices][fg_indices],
                                                 beta=self._anchor_smooth_l1_loss_beta)

            cross_entropies[batch_index] = cross_entropy
            smooth_l1_losses[batch_index] = smooth_l1_loss

        return cross_entropies, smooth_l1_losses

    def generate_anchors(self, image_width: int, image_height: int, num_x_anchors: int, num_y_anchors: int) -> Tensor:
        # generate center based on feature map scale, mapping to original image scale
        center_ys = np.linspace(start=0, stop=image_height, num=num_y_anchors + 2)[1:-1]
        center_xs = np.linspace(start=0, stop=image_width, num=num_x_anchors + 2)[1:-1]
        ratios = np.array(self._anchor_ratios)
        ratios = ratios[:, 0] / ratios[:, 1]
        sizes = np.array(self._anchor_sizes)
        # NOTE: it's important to let `center_ys` be the major index (i.e., move horizontally and then vertically) for consistency with 2D convolution
        # giving the string 'ij' returns a meshgrid with matrix indexing, i.e., with shape [len(center_ys), len(center_xs), len(ratios), len(sizes)]
        center_ys, center_xs, ratios, sizes = np.meshgrid(center_ys, center_xs, ratios, sizes, indexing='ij')
        # Then we'll have a full permutation of different center_ys, center_xs, ratios, sizes, that's why the shape is [len(center_ys), len(center_xs), len(ratios), len(sizes)]

        center_ys = center_ys.reshape(-1)
        center_xs = center_xs.reshape(-1)
        ratios = ratios.reshape(-1)
        sizes = sizes.reshape(-1)
        # W: width of base anchor
        # H: height of base anchor
        # ratio: height/width
        # w: new width
        # h: new height
        # size: sqrt(W*H)
        # w*h=W*H*scale^2
        # w*w*ratio=W*H*scale^2
        # now scale=1
        # w=sqrt(W*H/ratio)
        widths = sizes * np.sqrt(1 / ratios)
        heights = sizes * np.sqrt(ratios)
        # stack into center_based_anchor_bounding_boxes
        center_based_anchor_bboxes = np.stack((center_xs, center_ys, widths, heights), axis=1)
        center_based_anchor_bboxes = torch.from_numpy(center_based_anchor_bboxes).float()
        anchor_bboxes = BBox.from_center_base(center_based_anchor_bboxes)
        # return features_width*features_height*K(len(ratios)*len(sizes)) anchor_bboxes per image
        # (X_min,Y_min,X_max,Y_max)
        return anchor_bboxes

    def generate_proposals(self, anchor_bboxes: Tensor, objectnesses: Tensor, transformers: Tensor, image_width: int, image_height: int) -> Tensor:
        batch_size = anchor_bboxes.shape[0]
        # get proposal_bboxes
        proposal_bboxes = BBox.apply_transformer(anchor_bboxes, transformers)
        proposal_bboxes = BBox.clip(proposal_bboxes, left=0, top=0, right=image_width, bottom=image_height)
        proposal_probs = F.softmax(objectnesses[:, :, 1], dim=-1)

        _, sorted_indices = torch.sort(proposal_probs, dim=-1, descending=True)
        nms_proposal_bboxes_batch = []
        # nms
        for batch_index in range(batch_size):
            sorted_bboxes = proposal_bboxes[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]
            sorted_probs = proposal_probs[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]
            threshold = 0.7
            kept_indices = nms(sorted_bboxes, sorted_probs, threshold)
            nms_bboxes = sorted_bboxes[kept_indices][:self._post_nms_top_n]
            nms_proposal_bboxes_batch.append(nms_bboxes)

        max_nms_proposal_bboxes_length = max([len(it) for it in nms_proposal_bboxes_batch])
        padded_proposal_bboxes = []
        # padding for the same size
        for nms_proposal_bboxes in nms_proposal_bboxes_batch:
            padded_proposal_bboxes.append(
                torch.cat([
                    nms_proposal_bboxes,
                    torch.zeros(max_nms_proposal_bboxes_length - len(nms_proposal_bboxes), 4).to(nms_proposal_bboxes)
                ])
            )

        padded_proposal_bboxes = torch.stack(padded_proposal_bboxes, dim=0)
        return padded_proposal_bboxes
