import argparse
import sys,os
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_path,".."))
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
import torch
from torch import nn
from torchvision.transforms import functional as F
from maskrcnn_benchmark.structures.image_list import ImageList,to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
import cv2
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
    parser.add_argument("--ref-text", required=True, help="path to the reference text file,in which reference words are stored in each line")
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
    parser.add_argument("--conf-thres", type=float, default=0.2, help="text box confidence threshold for text retrieval")
    parser.add_argument("--similarity-thres", type=float, default=0.5, help="similarity threshold for text retrieval")
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    args = parser.parse_args()
    sim_th = args.similarity_thres
    conf_th = args.conf_thres
    ref_words = []
    with open(args.ref_text, "r", encoding="utf-8") as f:
        for line in f:
            ref_words.append(line.strip())

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
        imagelist = to_image_list(image_tensor) #ImageList(image_tensor, [(pad_h,pad_w)])
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
            pos_idxs = torch.nonzero(scores>conf_th).view(-1)#75.43
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
    #torch no_grad
    ref_words_embedding = []
    with torch.no_grad():
        words = [torch.tensor(model.decoder.head.text_generator.label_map(word.lower())).long().cuda() for word in ref_words]
        words_embedding = model.decoder.head.word_embedding(words)
        #words_embedding_list.append(words_embedding)
        words_embedding_nor = nn.functional.normalize((words_embedding).tanh().view(words_embedding.size(0),-1))
        ref_words_embedding = words_embedding_nor.cpu().numpy()
    #search for the word "welcome" in the image
    similarity_list = []
    match_text_boxes = {}
    for text_box in image_text:
        image_path = text_box[0]
        bbox = text_box[1]
        imgs_embedding_nor = text_box[2]
        for n in range(len(ref_words_embedding)):
            similarity = np.dot(imgs_embedding_nor, ref_words_embedding[n])
            similarity_list.append(similarity)
            if similarity > sim_th:
                #check if match_text_boxes has the image_path
                if image_path in match_text_boxes:
                    match_text_boxes[image_path].append((bbox,ref_words[n], similarity))
                else:
                    match_text_boxes[image_path] = [(bbox,ref_words[n], similarity)]
    
    #iterate through match_text_boxes with key as image_path and value as a list of bbox and similarity
    for image_path,bboxes in match_text_boxes.items():
        image = cv2.imread(image_path)
        image_width = image.shape[1]
        image_height = image.shape[0]
        #draw bbox
        for bbox,ref_word,similarity in bboxes:
            cv2.rectangle(image, (int(bbox[0]*image_width),int(bbox[1]*image_height)), (int(bbox[2]*image_width),int(bbox[3]*image_height)), (255,0,255), 2)
            #format a float with two significant digits
            if not args.hide_labels:
                viz_text = '{} {:.3f}'.format(ref_word, similarity)
                viz_text_rect = cv2.getTextSize(viz_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                #text height should be 20% of the bbox height
                scale = ((bbox[3]-bbox[1])*image_height)*0.2/viz_text_rect[1]
                #scale = scale
                cv2.putText(image,viz_text , (int(bbox[0]*image_width),int(bbox[1]*image_height)-2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,255), 1)
        #show image
        #get image name from path
        image_name = image_path.split('/')[-1]
        cv2.imshow(image_name, image)
        cv2.waitKey(0)
    print('done')