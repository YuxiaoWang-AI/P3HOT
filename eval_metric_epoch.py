
import os
import time
import argparse
import pickle
from distutils.version import LooseVersion

import numpy as np
import torch
import torch.nn as nn

from hot.config import cfg
from hot.dataset import ValDataset
from hot.models import ModelBuilder, SegmentationModule
from hot.utils import AverageMeter, colorEncode, accuracy, precision, accuracy_binary, contact_acc, intersectionAndUnion, setup_logger, adjust_iou
from hot.lib.nn import user_scattered_collate, async_copy_to
from hot.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
import CLIP.clip as clip
from hot.models.decoder import Decoder
import matplotlib.pyplot as plt



with open('data/colors.npy', 'rb') as f:
    colors = np.load(f)

def evaluate(segmentation_module, loader, cfg, gpu, epoch):
    print('evaluating epoch:', epoch)
    acc_meter = AverageMeter()
    prec_meter = AverageMeter()
    acc_b_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    ad_iou_meter = AverageMeter()
    neg_iou_meter = AverageMeter()
    time_meter = AverageMeter()
    ious = np.arange(0.05, 1, 0.05)
    tp = np.zeros_like(ious)
    contact_sum = 0

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for idx, batch_data in enumerate(loader):
   
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'])
        img_ori = batch_data["img_ori"]
        img_path = batch_data["info"]
        person_mask = batch_data["person_mask"]
        
        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
     
            feed_dict = batch_data.copy()
            feed_dict['img_data'] = batch_data['img_data'].unsqueeze(0)
            feed_dict['depth_label'] = batch_data['depth_label'].unsqueeze(0)
            feed_dict['person_mask'] = batch_data['person_mask'].unsqueeze(0)
            
            del feed_dict['img_ori']
            del feed_dict['info']
            feed_dict = async_copy_to(feed_dict, gpu)
          
            scores_tmp, final_mask = segmentation_module(feed_dict, segSize=segSize)

            _, pred = torch.max(scores_tmp, dim=1)   
            
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)
        
        
        acc, pix = accuracy(pred, seg_label)
        prec, pix_prec = precision(pred, seg_label)
        acc_b, pix_b = accuracy_binary(pred, seg_label)
        ad_iou, neg_iou = adjust_iou(img_ori, img_path, pred, seg_label, person_mask)
        tp_t, contact_sum_t = contact_acc(pred,seg_label) 
        tp = tp + tp_t
        contact_sum += contact_sum_t
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        prec_meter.update(prec, pix_prec)
        acc_b_meter.update(acc_b, pix_b)
        intersection_meter.update(intersection)
        union_meter.update(union)
        ad_iou_meter.update(ad_iou)
        neg_iou_meter.update(neg_iou)
        pbar.update(1)
        with open(os.path.join("./ckpt/hot-c1", f"epoch_{epoch}.txt"), "a+") as f:
            f.write(os.path.basename(img_path) + " " + str(neg_iou) + "\n")

    
    save_result = dict()
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    save_result['iou'] = iou
    for i, _iou in enumerate(iou):
        
        save_result['iou'][i] = _iou
    save_result['mean_iou'] = iou.mean()
    save_result['acc'] = acc_meter.average()*100
    save_result['prec'] = prec_meter.average()*100
    save_result['f1'] = 2*(acc_meter.average()*prec_meter.average())/(acc_meter.average()+prec_meter.average()+1e-10)
    save_result['acc_b'] = acc_b_meter.average()*100
    save_result['tp'] = tp
    save_result['contact_sum'] = contact_sum
    save_result['ad_iou'] = ad_iou_meter.average()
    save_result['neg_iou'] = neg_iou_meter.average()

    
    
    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))
    with open("./save_all_result.txt", "a+") as f:
        f.write(str(epoch) + " " + str(iou.mean()) + " " + str(acc_meter.average()*100) + "\n")
    with open(os.path.join(cfg.DIR, 'validation_metric_epoch_'+epoch+ '.pkl'), 'wb') as f:
        pickle.dump(save_result, f, pickle.HIGHEST_PROTOCOL)
    

def main(cfg, gpu, epoch):
    torch.cuda.set_device(gpu)

    net_decoder = Decoder()

    
    tim_model, preprocess = clip.load("RN50", device="cpu")
    tim_model.eval()
    
    tim_model.load_state_dict(torch.load(cfg.MODEL.weights_encoder))
    net_decoder.load_state_dict(torch.load(cfg.MODEL.weights_decoder))

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(tim_model=tim_model, decoder=net_decoder, crit=crit)

    
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    
    evaluate(segmentation_module, loader_val, cfg, gpu, epoch)

    


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/hot-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
    )
    parser.add_argument(
        "--epoch",
        default=0,
        help="which epoch"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    

    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_epoch_' + args.epoch + '.pth')
        
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_epoch_' + args.epoch + '.pth')
      
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"
    if not os.path.isdir(os.path.join(cfg.DIR, "result_epoch_"+args.epoch)):
        os.makedirs(os.path.join(cfg.DIR, "result_epoch_"+args.epoch))

    main(cfg, args.gpu, args.epoch)
