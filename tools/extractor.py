import sys
import os
import numpy as np
import cv2
import math
import datetime
import pickle

import torch
from core.config import cfg
import core.config as config

from util import walkfile
import delg_utils


""" common settings """
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]
MODEL_WEIGHTS = 'extraction/weights/model_epoch_0100.pyth'
PATH = "extraction/f43b9af15990f7de.jpg"
INFER_DIR = 'extraction/datasets/rparis6k/jpg'
COMBINE_DIR = "extraction/datasets/rparis6k/"


SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0]
#SCALE_LIST = [0.7071, 1.0, 1.4142]
#SCALE_LIST = [2.0]

IOU_THRES = 0.98
ATTN_THRES = 260.0
TOP_K = 1000

RF = 291.0
STRIDE = 16.0
PADDING = 145.0



def setup_model():
    model = delg_utils.DelgExtraction()
    print(model)
    load_checkpoint(MODEL_WEIGHTS, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def extract(im_array, model):
    input_data = torch.from_numpy(im_array)
    if torch.cuda.is_available(): 
        input_data = input_data.cuda() 
    fea = model(input_data, targets=None)
    _, delg_features, delg_scores = model(input_data, targets=None)
    #print(delg_features.size(), delg_scores.size())
    return delg_features, delg_scores


def delg_extract(img, model):
    """ multiscale process """
    # extract features for each scale, and concat.
    output_boxes = []
    output_features = []
    output_scores = []
    output_scales = []
    output_original_scale_attn = None
    for scale_factor in SCALE_LIST:
        im = preprocess(img.copy(), scale_factor)
        im_array = np.asarray([im], dtype=np.float32)
        delg_features, delg_scores = extract(im_array, model)
       
        #tmp = delg_scores.squeeze().view(-1)
        #print(torch.median(tmp))

        selected_boxes, selected_features, \
        selected_scales, selected_scores, \
        selected_original_scale_attn = \
                    delg_utils.GetDelgFeature(delg_features, 
                                        delg_scores,
                                        scale_factor,
                                        RF,
                                        STRIDE,
                                        PADDING,
                                        ATTN_THRES)

        output_boxes.append(selected_boxes) if selected_boxes is not None else output_boxes
        output_features.append(selected_features) if selected_features is not None else output_features
        output_scales.append(selected_scales) if selected_scales is not None else output_scales
        output_scores.append(selected_scores) if selected_scores is not None else output_scores
        if selected_original_scale_attn is not None:
            output_original_scale_attn = selected_original_scale_attn
    if output_original_scale_attn is None:
        output_original_scale_attn = im.clone().uniform()
    # concat tensors precessed from different scales.
    output_boxes = delg_utils.concat_tensors_in_list(output_boxes, dim=0)
    output_features = delg_utils.concat_tensors_in_list(output_features, dim=0)
    output_scales = delg_utils.concat_tensors_in_list(output_scales, dim=0)
    output_scores = delg_utils.concat_tensors_in_list(output_scores, dim=0)
    # perform Non Max Suppression(NMS) to select top-k bboxes arrcoding to the attn_score.
    keep_indices, count = delg_utils.nms(boxes = output_boxes,
                              scores = output_scores,
                              overlap = IOU_THRES,
                              top_k = TOP_K)
    keep_indices = keep_indices[:TOP_K]

    output_boxes = torch.index_select(output_boxes, dim=0, index=keep_indices)
    output_features = torch.index_select(output_features, dim=0, index=keep_indices)
    output_scales = torch.index_select(output_scales, dim=0, index=keep_indices)
    output_scores = torch.index_select(output_scores, dim=0, index=keep_indices)
    output_locations = delg_utils.CalculateKeypointCenters(output_boxes)
    
    data = {
        'locations':to_numpy(output_locations),
        'descriptors':to_numpy(output_features),
        'scores':to_numpy(output_scores)
        #'attention':to_numpy(output_original_scale_attn)
        }
    return data


def main(spath):
    model = setup_model()
    feadic = {}
    for index, imgfile in enumerate(walkfile(spath)):
        ext = os.path.splitext(imgfile)[-1]
        name = os.path.basename(imgfile)
        print(index, name)
        if ext.lower() in ['.jpg', '.jpeg', '.bmp', '.png', '.pgm']:
            im = cv2.imread(imgfile)
            im = im.astype(np.float32, copy=False)
            data =  delg_extract(im, model)
            feadic[name] = data
    with open(spath.split("/")[-2]+"localfea.pickle", "wb") as fout:
        pickle.dump(feadic, fout, protocol=2)


def main_multicard(spath, cutno, total_num):
    model = setup_model()
    feadic = {}
    for index, imgfile in enumerate(walkfile(spath)):
        if index % total_num != cutno - 1:
            continue
        ext = os.path.splitext(imgfile)[-1]
        name = os.path.basename(imgfile)
        print(index, name)
        if ext.lower() in ['.jpg', '.jpeg', '.bmp', '.png', '.pgm']:
            im = cv2.imread(imgfile)
            im = im.astype(np.float32, copy=False)
            data =  delg_extract(im, model)
            print(data['locations'].shape, data['descriptors'].shape)
            feadic[name] = data
    with open(COMBINE_DIR+"localfea.pickle"+'_%d'%cutno, "wb") as fout:
        pickle.dump(feadic, fout, protocol=2)


def test(impath):
    model = setup_model()
    im = cv2.imread(impath)
    im = im.astype(np.float32, copy=False)
    data =  delg_extract(im, model)
    for k, v in data.items():
        print(k)
        print(v.shape)
   

def preprocess(im, scale_factor):
    im = im_scale(im, scale_factor) 
    im = im.transpose([2, 0, 1])
    im = im / 255.0
    im = color_norm(im, _MEAN, _SD)
    return im

def im_scale(im, scale_factor):
    h, w = im.shape[:2]
    h_new = int(round(h * scale_factor))
    w_new = int(round(w * scale_factor))
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.float32)

def color_norm(im, mean, std):
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    try:
        state_dict = checkpoint["model_state"]
    except KeyError:
        state_dict = checkpoint
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model
    model_dict = ms.state_dict()

    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    if len(pretrained_dict) == len(state_dict):
        print('All params loaded')
    else:
        print('construct model total {} keys and pretrin model total {} keys.'.format(len(model_dict), len(state_dict)))
        print('{} pretrain keys load successfully.'.format(len(pretrained_dict)))
        not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
        print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
    model_dict.update(pretrained_dict)
    ms.load_state_dict(model_dict)
    #ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    #return checkpoint["epoch"]
    return checkpoint



if __name__ == '__main__':
    print(sys.argv)
    config.load_cfg_fom_args("Extract feature.")
    config.assert_and_infer_cfg()
    cfg.freeze()

    total_card = cfg.INFER.TOTAL_NUM
    assert total_card > 0, 'cfg.TOTAL_NUM should larger than 0. ~'
    assert cfg.INFER.CUT_NUM <= total_card, "cfg.CUT_NUM <= cfg.TOTAL_NUM. ~"
    if total_card == 1:
        main(INFER_DIR)
        #test(PATH)
    else:
        main_multicard(INFER_DIR, cfg.INFER.CUT_NUM, cfg.INFER.TOTAL_NUM)
