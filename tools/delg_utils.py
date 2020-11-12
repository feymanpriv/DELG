import sys, os
import numpy as np

import torch
import torch.nn as nn
from model.resnet import ResNet
from model.delg_model import SpatialAttention2d
import torch.nn.functional as F


class DelgExtraction(nn.Module):
    def __init__(self):
        super(DelgExtraction, self).__init__()
        self.globalmodel = ResNet()
        self.localmodel = SpatialAttention2d(1024)
    
    def forward(self, x, targets):
        global_feature, feamap = self.globalmodel(x)
        
        block3 = feamap.detach()
        local_feature, att_score = self.localmodel(block3)
        return global_feature, local_feature, att_score


def GenerateCoordinates(h, w):
    '''generate coorinates
    Returns: [h*w, 2] FloatTensor
    '''
    x = torch.floor(torch.arange(0, float(w*h)) / w)
    y = torch.arange(0, float(w)).repeat(h)

    coord = torch.stack([x,y], dim=1)
    return coord


def CalculateReceptiveBoxes(height, width, rf, stride, padding):
    coordinates = GenerateCoordinates(height, width)
    point_boxes = torch.cat([coordinates, coordinates], dim=1)
    bias = torch.FloatTensor([-padding, -padding, -padding + rf - 1, -padding + rf - 1])
    rf_boxes = stride * point_boxes + bias
    return rf_boxes


def CalculateKeypointCenters(rf_boxes):
    '''compute feature centers, from receptive field boxes (rf_boxes).
    Args:
        rf_boxes: [N, 4] FloatTensor.
    Returns:
        centers: [N, 2] FloatTensor.
    '''
    xymin = torch.index_select(rf_boxes, dim=1, index=torch.LongTensor([0,1]).cuda())
    xymax = torch.index_select(rf_boxes, dim=1, index=torch.LongTensor([2,3]).cuda())
    return (xymax + xymin) / 2.0


def GetDelgFeature(
    delg_features, 
    delg_scores,
    scale,
    rf,
    stride,
    padding,
    attn_thres,
    ):

    # save original size attention (used for attention visualization.)
    selected_original_attn = None
    if scale == 1.0:
        selected_original_attn = torch.clamp(delg_scores, 0, 255) # 1 1 h w
        
    # calculate receptive field boxes.
    rf_boxes = CalculateReceptiveBoxes(
        height=delg_features.size(2),
        width=delg_features.size(3),
        rf=rf,
        stride=stride,
        padding=padding)
    
    # re-projection back to original image space.
    rf_boxes = rf_boxes / scale
    # perform l2norm
    delg_features = F.normalize(delg_features, p=2, dim=1)

    delg_scores = delg_scores.view(-1)
    delg_features = delg_features.view(delg_features.size(1), -1).t()
    
    # use attention score to select feature.
    # indices = torch.gt(delg_scores, attn_thres).nonzero().squeeze()
    indices = None
    while(indices is None or len(indices) == 0 or len(indices) < 1000):
        indices = torch.gt(delg_scores, attn_thres).nonzero().squeeze()
        attn_thres = attn_thres * 0.5   # use lower threshold if no indexes are found.
        if attn_thres < 0.1:
            break;
    #try:

    selected_boxes = torch.index_select(rf_boxes.cuda(), dim=0, index=indices)
    selected_features = torch.index_select(delg_features, dim=0, index=indices)
    selected_scores = torch.index_select(delg_scores, dim=0, index=indices)
    selected_scales = torch.ones_like(selected_scores) * scale
    '''
    except Exception as e:
        selected_boxes = None
        selected_features = None
        selected_scores = None
        selected_scales = None
        print(e)
        pass;
    '''    
    return selected_boxes, selected_features, selected_scales, selected_scores, selected_original_attn


# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Returns:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def concat_tensors_in_list(tensor_list, dim):
    res = None
    tensor_list = [x for x in tensor_list if x is not None]
    for tensor in tensor_list:
        if res is None:
            res = tensor
        else:
            res = torch.cat((res, tensor), dim=dim)
    return res


if __name__ == "__main__":
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('tools.py command', file=sys.stderr)
