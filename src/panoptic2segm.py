#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tool for converting panoptic GT into semantic segmentation uint8 pngs and/or instance segmentation uint16 pngs
# see https://github.com/ozendelait/wilddash_scripts
# by Oliver Zendel and Bernhard Rainer, AIT Austrian Institute of Technology GmbH
#
# Use this tool on your own risk!
# Copyright (C) 2022 AIT Austrian Institute of Technology GmbH
# All rights reserved.
#******************************************************************************
import cv2
import numpy as np
import os
import sys
import glob
import json
import argparse
from collections import defaultdict


wd2tocityscan_mapping = defaultdict(lambda: 255, {
    0: 255,   # unlabeled
    1: 255,   # ego vehicle
    2: 255,   # rectification border
    3: 255,   # out of roi
    4: 255,   # static
    5: 255,   # dynamic
    6: 255,   # ground
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,   # parking
    10: 255,  # rail track
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail
    15: 255,  # bridge
    16: 255,  # tunnel
    17: 5,    # pole
    18: 255,  # polegroup (ignored)
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan
    30: 255,  # trailer
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
    34: 13,   # pickup (mapped to car)
    35: 13,   # van (mapped to car)
    36: 7,    # billboard (mapped to traffic sign)
    37: 5,    # street-light (mapped to pole)
    38: 255   # road-marking (ignored)
})



def tqdm_none(l, desc='', total=None):
    return l
try:
    from tqdm import tqdm as tqdm_con
    from tqdm.notebook import tqdm as tqdm_nb
except:
    #install/update tqdm needed
    tqdm_con = tqdm_nb = tqdm_none

def bgrids_to_intids(orig_mask_bgr):
    return  orig_mask_bgr[:, :, 0].astype(np.uint32) * 65536 + \
            orig_mask_bgr[:, :, 1].astype(np.uint32) * 256 + \
            orig_mask_bgr[:, :, 2].astype(np.uint32)
            
def check_set_cont(n):
    if not n is None and not n.flags['C_CONTIGUOUS']:
        return np.asarray(n, order='C')
    return n
    
def intids_to_bgrids(masksegm):
    return check_set_cont(masksegm).view(np.uint8).reshape((masksegm.shape[0],masksegm.shape[1],4))[:,:,::-1][:,:,1:4]

def panoptic2segm(json_path, outp_dir_sem=None, label_png_dir=None, tqdm_vers=tqdm_nb):
    #default: masks are in a directory with the same name as the panoptic json filename
    if label_png_dir is None: label_png_dir = json_path[:json_path.rfind('.')]
    pano0 = json.load(open(json_path))
    id2image = {image["id"]: image for image in pano0["images"]}
    if outp_dir_sem and not os.path.exists(outp_dir_sem): os.makedirs(outp_dir_sem)
    cnt_success = 0
    for a in tqdm_vers(pano0["annotations"]): 
        image_id = a["image_id"]
        if image_id in id2image: 
            ids_path = label_png_dir+'/'+ a["file_name"]
            bgr_labels = cv2.imread(ids_path)
            ids = bgrids_to_intids(np.asarray(bgr_labels))
            semantic = np.zeros_like(ids, dtype="uint8")
            for s in a["segments_info"]: 
                id0 = s["id"]
                category_id = s["category_id"]
                semantic[ids == id0] = category_id
            if outp_dir_sem:
                semantic_name = id2image[image_id]["file_name"].replace(".jpg", "_labelIds.png")
                cv2.imwrite(outp_dir_sem+'/'+semantic_name, semantic)
            cnt_success += 1
    return cnt_success
    
def pano2sem_main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default="panoptic.json",
                        help="Path to panoptic COCO json")
    parser.add_argument('--outp_dir_sem', type=str, default=None,
                        help="Target directory for semantic uint8 pngs")
    parser.add_argument('--label_png_dir', type=str, default=None,
                        help="Specify directory of panoptic COCO png BGR masks (default: use json_path as hint)")
    parser.add_argument('--silent', action='store_true', help="Suppress all outputs")
    parser.add_argument('--verbose', action='store_true', help="Print extra information")
    args = parser.parse_args(argv)
    if not args.outp_dir_sem :
        if not args.silent:
            print("Error: no output operation selected.")
        return -1
    tqdm_vers = tqdm_none if args.silent else tqdm_con
    cnt_success = panoptic2segm(json_path=args.json_path, outp_dir_sem=args.outp_dir_sem, label_png_dir=args.label_png_dir, tqdm_vers=tqdm_vers)
    if not args.silent:
        print("Finished converting panoptic COCO GT with %i successes."%(cnt_success))

if __name__ == "__main__":
    sys.exit(pano2sem_main())