import argparse
import sys
sys.path.append("D:\\code\\Research\\STR-TDSL")
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
import torch
from torch import nn
from torchvision.transforms import functional as F
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
import cv2
from  imageio import imread
import os
from tqdm import tqdm
import numpy as np
from infererence import create_tensor_cv,create_tensor_PIL

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="retireve text from a given image set")
    parser.add_argument(
        "--input",
        required=False,
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
    model.cuda()
    
    
    has_cuda = torch.cuda.is_available()
    #check if os.input_file is dir or file
    if os.path.isdir(args.input):
        #find all images in the folder, with file name extension of .jpg, .png, .jpeg
        images = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(('.jpg', '.png', '.jpeg'))]
    else:
        images = [args.input]
    
    image_text = []
    for image_path in tqdm(images,"processing images.."):
        if len(image_text)>=1000:
            break
    
        #convert new image to tensor
        #image_tensor = create_tensor_cv(image_path, int(args.longer_side))
        #image_tensor = F.normalize(image_tensor, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])
        image_tensor = create_tensor_PIL(image_path, int(args.longer_side))
        image_tensor = F.normalize(image_tensor, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])
        #add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        if has_cuda:
            image_tensor = image_tensor.cuda()
        pad_w = image_tensor.shape[3]
        pad_h = image_tensor.shape[2]
        imagelist = ImageList(image_tensor, [(pad_w,pad_h)])
        with torch.no_grad():
            #produce FPN conv features
            features = model.neck(model.backbone(image_tensor))
            #boxes is detected text boxes, each box is a tensor of shape [N,4]
            rec_features = features[:len(model.decoder.scales)]
            #features is a list of tensors or feature pyramid, each tensor is a feature map
            #now we can use the internel FCOS detector to detect text boxes or text boxes are provided(from external module) as input
            bboxes, losses = model.decoder.detector(imagelist, features[1:], None)
            bboxes = bboxes[0]
            scores = bboxes.get_field("scores")
        
            #box confidence score threshold is 0.2
            pos_idxs = torch.nonzero(scores>0.2).view(-1)#75.43
            confident_bboxes = bboxes[pos_idxs]
            #bbox provided by external module
            # x1 = 187/960*pad_w
            # y1 = 414/540*pad_h
            # x2 = (187+277)/960*pad_w
            # y2 = (414+32)/540*pad_h
            # confident_bboxes = BoxList(torch.tensor([[x1,y1,x2,y2],[x1+20,y1+3,x2+30,y2+2],[x1+40,y1+3,x2+30,y2+5]]), (pad_w,pad_h), mode="xyxy")
            dim = model.decoder.head.image_embedding.rnn.embedding.out_features
            aligned_roi_height = cfg.MODEL.ALIGN.POOLER_RESOLUTION[1]
            if confident_bboxes.bbox.size()[0] == 0:
                #imgs_embedding_nor = torch.zeros([0,aligned_roi_height*dim])
                continue
            else:
                rois = model.decoder.head.pooler(rec_features, [confident_bboxes])
                imgs_embedding = model.decoder.head.image_embedding(rois)
                #image_embedding is a tensor of shape [N,15,128], N is the number of text boxes
                imgs_embedding_nor = nn.functional.normalize((imgs_embedding).tanh().view(imgs_embedding.size(0),-1))
                #get roi relative coordinates
                for n in range(confident_bboxes.bbox.size()[0]):
                    bbox = confident_bboxes.bbox[n]
                    x1 = bbox[0].item()/pad_w
                    y1 = bbox[1].item()/pad_h
                    x2 = bbox[2].item()/pad_w
                    y2 = bbox[3].item()/pad_h
                    image_text.append((image_path, (x1,y1,x2,y2),imgs_embedding_nor[n,:].cpu().numpy()))
    #search for the word "welcome" in the image
    texts = ['Donaukurief']
    #torch no_grad
    query_words_embedding_list = []
    with torch.no_grad():
        words = [torch.tensor(model.decoder.head.text_generator.label_map(text.lower())).long().cuda() for text in texts]
        words_embedding = model.decoder.head.word_embedding(words)
        #words_embedding_list.append(words_embedding)
        words_embedding_nor = nn.functional.normalize((words_embedding).tanh().view(words_embedding.size(0),-1))
        query_words_embedding_list.append(words_embedding_nor[0].cpu().numpy())
    #search for the word "welcome" in the image
    similarity_list = []
    for text_box in image_text:
        image_path = text_box[0]
        bbox = text_box[1]
        imgs_embedding_nor = text_box[2]
        similarity = np.dot(imgs_embedding_nor, query_words_embedding_list[0])
        similarity_list.append(similarity)
        if similarity > 0.6:
            #print(f"image {image_path} contains the word {texts[0]}")
            #print(f"bbox: {bbox}")
            #print(f"similarity score: {similarity}")
            #load image with cv2
            image = cv2.imread(image_path)
            image_width = image.shape[1]
            image_height = image.shape[0]
            #draw bbox
            cv2.rectangle(image, (int(bbox[0]*image_width),int(bbox[1]*image_height)), (int(bbox[2]*image_width),int(bbox[3]*image_height)), (0,255,0), 2)
            #format a float with two significant digits
            cv2.putText(image, '{:.3f}'.format(similarity), (int(bbox[0]*image_width),int(bbox[1]*image_height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            #show image
            cv2.imshow("image", image)
    cv2.waitKey(0)
    print('done')