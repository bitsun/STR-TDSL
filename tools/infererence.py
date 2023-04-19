
import argparse
import sys,os
#get the path of the script
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_path,".."))
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
import torch
from torch import nn
from torchvision.transforms import functional as F
from maskrcnn_benchmark.structures.image_list import ImageList,to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
import cv2
from  imageio import imread
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
def create_tensor_PIL(image_path:str,longer_side:int):
    """
    create tensor with PIL image
    returns image as uint8 numpy array
    """
    img = imread(image_path)
    ori_h, ori_w, _ = img.shape
    img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    #image preprocessing is encoded in the evaluation file
    if ori_h > ori_w:
        h = longer_side
        w = int(ori_w*1.0/ori_h*h)
    else:
        w = longer_side
        h = int(ori_h*1.0/ori_w*w)  
    pad_h = h if h%32==0 else (h//32+1)*32
    pad_w = w if w%32==0 else (w//32+1)*32
    # image resize wiht PIL
    new_image = Image.Image.resize(img, (pad_w, pad_h), Image.BILINEAR)
    #new_image = imresize(img.copy(), (pad_h,pad_w))
    image_tensor = F.to_tensor(new_image)
    image_tensor = image_tensor[[2, 1, 0]] * 255
    image_tensor = image_tensor.float()
    return image_tensor

def create_tensor_cv(image_path:str,longer_side:int):
    img = imread(image_path)
    if len(img.shape)==2:
        #convert gray image to rgb image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        print(f"image {image_path} is gray image, convert to RGB")
    if len(img.shape)!=3:
        print(f"ignoring image {image_path}")
        return None
    if img.shape[2]==4:
        #convert RGBA image to RGB image
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        print(f"image {image_path} is RGBA image, convert to RGB")
    elif img.shape[2]==1:
        #convert gray image to rgb image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        print(f"image {image_path} is gray image, convert to RGB")
    elif img.shape[2]!=3:
        print(f"ignoring image {image_path},which has {img.shape[2]} channels")
        return None
    ori_h, ori_w, _= img.shape
    if ori_h > ori_w:
        h = longer_side
        w = int(ori_w*1.0/ori_h*h)
    else:
        w = longer_side
        h = int(ori_h*1.0/ori_w*w)  
    pad_h = h if h%32==0 else (h//32+1)*32
    pad_w = w if w%32==0 else (w//32+1)*32
    new_image = cv2.resize(img, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(new_image).permute(2,0,1).float()
    return image_tensor

# "--input","D:\\data\\datasets\\IIIT_STR_V1.0\\imgDatabase",
#                 "--longer-side","960",
#                 "--config-file","D:\\code\\Research\\STR-TDSL\\configs\\evaluation.yaml",
#                 "--ckpt","D:\\data\\datasets\\model_7709.pth"
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="text detection and and its feature extraction")
    parser.add_argument(
        "--input",
        required=True,
        help="path to the input image or input image folder"
    )
    parser.add_argument(
        "--config-file",
        default="D:\\code\\Research\\STR-TDSL\\configs\\evaluation.yaml",
        metavar="FILE",
        help="path to config file which specifies how to create the model",
    )
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        required=True
    )	# --config-file "configs/align/line_bezier0732.yaml" 
	# --skip-test \
    parser.add_argument(
        "--longer-side",
        help="longer side of the image",
        default=640
    )
    args = parser.parse_args()

    model_path = args.ckpt
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    model = build_detection_model(cfg)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    _ = checkpointer.load(model_path, use_latest=args.ckpt is None)
    model.eval()
    #check if torch has cuda
    has_cuda = torch.cuda.is_available()

    if has_cuda:
        model.cuda()
    #check if os.input_file is dir or file
    if os.path.isdir(args.input):
        #find all images in the folder, with file name extension of .jpg, .png, .jpeg
        images = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(('.jpg', '.png', '.jpeg'))]
    else:
        images = [args.input]
    
    for image_path in tqdm(images,"processing images.."):
    #image_path = "D:\\data\\datasets\\IIIT_STR_V1.0\\imgDatabase\\img_001312.jpg"
        #read image with pillow
        image_tensor = create_tensor_PIL(image_path, int(args.longer_side))
        image_tensor = F.normalize(image_tensor, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])
        #add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        pad_w = image_tensor.shape[3]
        pad_h = image_tensor.shape[2]
        if has_cuda:
            image_tensor = image_tensor.cuda()
        
        imagelist = to_image_list(image_tensor)
        with torch.no_grad():
            #produce FPN conv features
            features = model.neck(model.backbone(image_tensor))
            #features is a list of tensors or feature pyramid, each tensor is a feature map
            #now we can use the internel FCOS detector to detect text boxes or text boxes are provided(from external module) as input
            bboxes, losses = model.decoder.detector(imagelist, features[1:], None)
            bboxes = bboxes[0]
            #boxes is detected text boxes, each box is a tensor of shape [N,4]
            rec_features = features[:len(model.decoder.scales)]
            scores = bboxes.get_field("scores")
        
            #box confidence score threshold is 0.2
            pos_idxs = torch.nonzero(scores>0.2).view(-1)#75.43
            confident_bboxes = bboxes[pos_idxs]
            dim = model.decoder.head.image_embedding.rnn.embedding.out_features
            aligned_roi_height = cfg.MODEL.ALIGN.POOLER_RESOLUTION[1]
            if confident_bboxes.bbox.size()[0] == 0:
                imgs_embedding_nor = torch.zeros([0,aligned_roi_height*dim])
            else:
                rois = model.decoder.head.pooler(rec_features, [confident_bboxes])
                imgs_embedding = model.decoder.head.image_embedding(rois)
                #image_embedding is a tensor of shape [N,15,128], N is the number of text boxes
                imgs_embedding_nor = nn.functional.normalize((imgs_embedding).tanh().view(imgs_embedding.size(0),-1))
    print(imgs_embedding_nor.shape)