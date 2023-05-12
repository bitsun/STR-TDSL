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
import json
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_path,".."))

def pop_buffered_frames(buffered_frames,recognition_results,tracked_results,visualize=False):
    frame = buffered_frames.pop(0)
    if visualize:
        cur_frame_idx = frame[0]
        cur_frame = frame[1]
        #any result if recognition?
        if cur_frame_idx in recognition_results:
            for bbox,label,similarity in recognition_results[cur_frame_idx]:
                cv2.rectangle(cur_frame,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),1)
                viz_text = '{} {:.3f}'.format(label, similarity)
                viz_text_rect = cv2.getTextSize(viz_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                #text height should be 20% of the bbox height
                scale = (bbox[3]-bbox[1])*0.2/viz_text_rect[1]
                if(bbox[1]<5):
                    cv2.putText(cur_frame,viz_text,(int(bbox[0]),int(bbox[3])-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                else:
                    cv2.putText(cur_frame,viz_text,(int(bbox[0]),int(bbox[1])-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        #any result if tracking?
        if cur_frame_idx in tracked_results:
            for bbox,label in tracked_results[cur_frame_idx]:
                cv2.rectangle(cur_frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),1)
                viz_text = '{}'.format(label)
                viz_text_rect = cv2.getTextSize(viz_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                #text height should be 20% of the bbox height
                scale = (bbox[3]-bbox[1])*0.2/viz_text_rect[1]
                if(bbox[1]<5):
                    cv2.putText(cur_frame,label,(int(bbox[0]),int(bbox[3])-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                else:
                    cv2.putText(cur_frame,label,(int(bbox[0]),int(bbox[1])-3),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imshow("frame1",cur_frame)
        cv2.waitKey(20)
        cv2.imwrite('F:\\code\\Research\\STR-TDSL\\output_track\\test_images_'+str(cur_frame_idx).zfill (4)+'.jpg',cur_frame)

def get_iou(bbox1,bbox2):
    """
    bbox1 and bbox2 are both in xyxy mode
    """
    x1 = max(bbox1[0],bbox2[0])
    y1 = max(bbox1[1],bbox2[1])
    x2 = min(bbox1[2],bbox2[2])
    y2 = min(bbox1[3],bbox2[3])
    if(x1>=x2 or y1>=y2):
        return 0
    intersection = (x2-x1)*(y2-y1)
    union = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])+(bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])-intersection
    return intersection/union

def track_bbox_forward(model,tracker,bboxes,buffered_images,tracked_results,recognition_result):
    """
    image_size: a tuple of (width,height)
    model: the model used to extract roi features
    bboxes: a list of 2 elements tuple, first element is a 4 element tuple,reprensting a bbox, 2nd element is the label
    """
    start_idx = len(buffered_images)-1
    if(start_idx<0 or start_idx-1 <0 or start_idx>=len(buffered_images)):
        return None
    #for each bbox initialize its own tracker
    for bbox,_,label,_ in bboxes:
        #bbox is xyxy mode, convert it to xywh mode
        bbox_xywh = (int(bbox[0]),int(bbox[1]),int(bbox[2]-bbox[0]),int(bbox[3]-bbox[1]))
        tracker.init(buffered_images[start_idx][1],bbox_xywh)
        idx = start_idx-1
        old_bbox = bbox_xywh
        tracked_frames = 0
        while True:
            if(idx<0 or idx>=len(buffered_images)):
                break

            ok, new_bbox = tracker.update(buffered_images[idx][1])
            print(tracker.getTrackingScore(),buffered_images[idx][0], label,old_bbox,new_bbox)
            if ok and new_bbox is not None and tracker.getTrackingScore()>0.6:
                cur_frame_idx = buffered_images[idx][0]
                
                #new_bbox is xywh,convert it back to xyxy
                new_bbox = (new_bbox[0],new_bbox[1],new_bbox[0]+new_bbox[2],new_bbox[1]+new_bbox[3])
                #check if the new_bbox is in the recognition result
                new_box_in_recognition_result = False
                if cur_frame_idx in recognition_result:
                    for bbox,recog_label,_ in recognition_result[cur_frame_idx]:
                        if recog_label!=label:
                            continue
                        #get the intersection over union of two bboxes, both boxes are in xyxy mode
                        iou = get_iou(bbox,new_bbox)
                        if iou>0.5:
                            new_box_in_recognition_result = True
                            break
                if new_box_in_recognition_result:
                    #do not track any more
                    break
                tracked_frames += 1
            else:
                break
            #if tracked more than 20 frames,we do one detection
            verify_tracked_bbox = True
            if tracked_frames>10:
                tracked_frames = 0
                verify_tracked_bbox = False
                
                image_tensor = torch.from_numpy(buffered_images[idx][1]).permute(2,0,1).float()
                image_tensor = F.normalize(image_tensor, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])
                #add batch dimension
                image_tensor = image_tensor.unsqueeze(0)
                if has_cuda:
                    image_tensor = image_tensor.cuda()
                imagelist = to_image_list(image_tensor) #Image
                with torch.no_grad():
                    #produce FPN conv features
                    features = model.neck(model.backbone(image_tensor))
                    #boxes is detected text boxes, each box is a tensor of shape [N,4]
                    rec_features = features[:len(model.decoder.scales)]
        
                    bbox_tensor = np.zeros((1,4),dtype=np.float32)
                    bbox_tensor[0,0] = new_bbox[0]
                    bbox_tensor[0,1] = new_bbox[1]
                    bbox_tensor[0,2] = new_bbox[2]
                    bbox_tensor[0,3] = new_bbox[3]
                    bbox_tensor = torch.from_numpy(bbox_tensor)
                    new_bboxes = BoxList(bbox_tensor, (1088,1920), mode="xyxy")
                    #extract roi features
                    rois  = model.decoder.head.pooler(rec_features, [new_bboxes])
                    imgs_embedding = model.decoder.head.image_embedding(rois)
                    #image_embedding is a tensor of shape [N,15,128], N is the number of text boxes
                    imgs_embedding_nor = nn.functional.normalize((imgs_embedding).tanh().view(imgs_embedding.size(0),-1))
                    imgs_embedding_nor = imgs_embedding_nor.detach().cpu().numpy()
                sim = np.dot(bboxes[0][1],imgs_embedding_nor[0])
                print("tracked more than 10 frames, do one detection, sim:",sim)
                if np.dot(bboxes[0][1],imgs_embedding_nor[0])>0.5:
                    verify_tracked_bbox = True
            if(verify_tracked_bbox):
                if cur_frame_idx not in tracked_results:
                    tracked_results[cur_frame_idx] = [(new_bbox,label)]
                else:
                    tracked_results[cur_frame_idx].append((new_bbox,label))
            else:
                break
            old_bbox= new_bbox
            idx = idx - 1

def load_ref_image_embedding(model,image_dir:str):
    #load the images in the folder, which has extension jpg or png
    image_paths = []
    for file in os.listdir(image_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_paths.append(os.path.join(image_dir,file))
    #find its corresponding label file in the same folder,label file has the same name as the image file, but with extension json
    result = []
    for image_path in image_paths:
        label_path = os.path.splitext(image_path)[0]+".json"
        with open(label_path,"r") as f:
            label = json.load(f)
        image = cv2.imread(image_path)
        h = image.shape[0]
        w = image.shape[1]

        pad_h = h if h%32==0 else (h//32+1)*32
        pad_w = w if w%32==0 else (w//32+1)*32
        new_image = cv2.resize(image, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(new_image).permute(2,0,1).float()
        image_tensor = F.normalize(image_tensor, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])
        #add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        if has_cuda:
            image_tensor = image_tensor.cuda()
        imagelist = to_image_list(image_tensor) #Image
        with torch.no_grad():
            #produce FPN conv features
            features = model.neck(model.backbone(image_tensor))
            #boxes is detected text boxes, each box is a tensor of shape [N,4]
            rec_features = features[:len(model.decoder.scales)]
        
        bbox_tensor = np.zeros((len(label['Logo']),4),dtype=np.float32)
        logo_labels = []
        for i in range(len(label['Logo'])):
            for key,value in label['Logo'][i].items():
                logo_labels.append(key)
                #resize the bbox value to the padded image
                value = np.array(value,dtype=np.float32)
                value[0] = value[0]*pad_w/w
                value[1] = value[1]*pad_h/h
                value[2] = value[2]*pad_w/w
                value[3] = value[3]*pad_h/h
                bbox_tensor[i] = value
        bbox_tensor = torch.from_numpy(bbox_tensor)
        new_bboxes = BoxList(bbox_tensor, (pad_h,pad_w), mode="xyxy")
        #extract roi features
        with torch.no_grad():
            rois  = model.decoder.head.pooler(rec_features, [new_bboxes])
            imgs_embedding = model.decoder.head.image_embedding(rois)
            #image_embedding is a tensor of shape [N,15,128], N is the number of text boxes
            imgs_embedding_nor = nn.functional.normalize((imgs_embedding).tanh().view(imgs_embedding.size(0),-1))
            imgs_embedding_nor = imgs_embedding_nor.detach().cpu().numpy()

        for i in range(len(logo_labels)):
            result.append([image_path,logo_labels[i],imgs_embedding_nor[i,:]])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="retireve text from a given image set")
    parser.add_argument(
        "--input",
        required=False,
        help="path to the input image or input image folder"
    )
    parser.add_argument("--ref-text", required=True, help="path to the reference text file,in which reference words are stored in each line")
    parser.add_argument("--ref-image",required=False,help="path to the reference images and their labels"
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
    parser.add_argument(
        "--longer-side",
        help="longer side of the image",
        default=640
    )
    parser.add_argument(
        "--bbox-file",
        help="path to the bboxes file, which is generated by tamedAI",
    )
    
    parser.add_argument("--conf-thres", type=float, default=0.2, help="text box confidence threshold for text retrieval")
    parser.add_argument("--similarity-thres", type=float, default=0.5, help="similarity threshold for text retrieval")
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')

    args = parser.parse_args()
    sim_th = args.similarity_thres
    conf_th = args.conf_thres
    model_path = args.ckpt
    if not os.path.exists(model_path):
        raise ValueError("Model file not found!")
    if not os.path.exists(args.ref_text):
        raise ValueError("Reference text file not found!")
    if not os.path.exists(args.input):
        raise ValueError("Input video not found!")
    if not os.path.exists(args.bbox_file):
        raise ValueError("Bbox file not found!")
    
    #read bboxes generated by tamedAI
    #read a file line by line
    lines = open(args.bbox_file, 'r').readlines()
    detections = []
    for line in lines:
        #load the json data
        detection_per_frame = json.loads(line)
        detections.append(detection_per_frame)

    #open reference text file
    ref_words = []
    with open(args.ref_text, "r", encoding="utf-8") as f:
        for line in f:
            ref_words.append(line.strip())

    
    longer_side = int(args.longer_side)
    
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    model = build_detection_model(cfg)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    _ = checkpointer.load(model_path, use_latest=args.ckpt is None)
    model.eval()
    has_cuda = False#torch.cuda.is_available()
    if has_cuda:
        model.cuda()

    ref_images_embedding = load_ref_image_embedding(model,args.ref_image)
    #create reference word embeddings
    #torch no_grad
    with torch.no_grad():
        words = [torch.tensor(model.decoder.head.text_generator.label_map(word.lower())).long() for word in ref_words]
        if has_cuda:
            words = [word.cuda() for word in words]
        words_embedding = model.decoder.head.word_embedding(words)
        #words_embedding_list.append(words_embedding)
        words_embedding_nor = nn.functional.normalize((words_embedding).tanh().view(words_embedding.size(0),-1))
        ref_words_embedding = words_embedding_nor.cpu().numpy()
        print(ref_words_embedding.shape)
    if ref_images_embedding is not None:
        for i in range(len(ref_images_embedding)):
            ref_words.append(ref_images_embedding[i][1])
            ref_words_embedding = np.vstack((ref_words_embedding,ref_images_embedding[i][2]))
    #open video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise ValueError("Cannot open input video!")
    frame_idx = 0

    #create a python queue
    buffered_images = []
    
    param=cv2.TrackerNano.Params()
    param.backbone="F:\\code\\Research\\STR-TDSL\\tools\\nanotrack_backbone_sim.onnx"
    param.neckhead="F:\\code\\Research\\STR-TDSL\\tools\\nanotrack_head_sim.onnx"
    tracker = cv2.TrackerNano.create(param)
    tracked_results = {}
    #open video writer to encode the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi',fourcc, 10.0, (1280,736))
    recognition_result = {}
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        ori_h, ori_w, _= frame.shape
        if ori_h > ori_w:
            h = longer_side
            w = int(ori_w*1.0/ori_h*h)
        w = longer_side
        h = int(ori_h*1.0/ori_w*w)  
        pad_h = h if h%32==0 else (h//32+1)*32
        pad_w = w if w%32==0 else (w//32+1)*32
        new_image = cv2.resize(frame, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(new_image).permute(2,0,1).float()
        image_tensor = F.normalize(image_tensor, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])
        #add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        if has_cuda:
            image_tensor = image_tensor.cuda()
        imagelist = to_image_list(image_tensor) #ImageList(image_tensor, [(pad_h,pad_w)])
        with torch.no_grad():
            #produce FPN conv features
            features = model.neck(model.backbone(image_tensor))
            #boxes is detected text boxes, each box is a tensor of shape [N,4]
            rec_features = features[:len(model.decoder.scales)]
            #put the rec_feature into the queue

            #if the queue is full, pop the first element
            buffered_images.append((frame_idx,new_image))
            if len(buffered_images) > 300:
                #buffered_images.pop(0)
                pop_buffered_frames(buffered_images,recognition_result, tracked_results,True)

            #features is a list of tensors or feature pyramid, each tensor is a feature map
            #now we can use the internel FCOS detector to detect text boxes or text boxes are provided(from external module) as input
            bboxes, losses = model.decoder.detector(imagelist, features[1:], None)
            bboxes = bboxes[0]
            scores = bboxes.get_field("scores")
        
            #box confidence score threshold is 0.2
            pos_idxs = torch.nonzero(scores>conf_th).view(-1)#75.43
            confident_bboxes = bboxes[pos_idxs]
        confident_bboxes_np = confident_bboxes.bbox.detach().cpu().numpy()
        #get the intersection of confident bboxes and bboxes from tamedAI
        bboxes_tamedAI = []
        if frame_idx>= len(detections):
            break
        for det in detections[frame_idx]['detections']:
            #get the bounding box
            # target = det['target']
            # if target != 'logo' and target != 'brand':
            #     continue
            bbox = det['box']
            #bboxes provided by tamedAI have coordinates relative to the original image,convert them to the resized image
            x1 = max(0,bbox[0])/ori_w*pad_w
            y1 = max(0,bbox[1])/ori_h*pad_h
            x2 = min(frame.shape[1],bbox[2])/ori_w*pad_w
            y2 = min(frame.shape[0],bbox[3])/ori_h*pad_h
            bboxes_tamedAI.append((x1,y1,x2,y2))
        
        #get the intersection of confident bboxes and bboxes from tamedAI
        new_logo_region = []
        for bbox1 in confident_bboxes_np:
            box1area = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
            if box1area == 0 :
                continue
            shouldAdd = False
            for bbox2 in bboxes_tamedAI:
                x1 = max(bbox1[0],bbox2[0])
                y1 = max(bbox1[1],bbox2[1])
                x2 = min(bbox1[2],bbox2[2])
                y2 = min(bbox1[3],bbox2[3])
                box2area = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
                if x2<=x1 or y2<=y1:
                    continue
                if (y2-y1)*(x2-x1)>box2area*0.8:
                    #detecyted text box is well covered by tamedAI bbox, so do not have to add them again
                    break
                if (y2-y1)*(x2-x1)>box1area*0.5:
                    shouldAdd = True
                    break
            if shouldAdd:
                new_logo_region.append(bbox1)  
                break
        #now create new  bboxes for text recognition
        #BoxList(confident_bboxes.image_size, confident_bboxes.box).convert("xyxy")    
        #create np.array of shape (2,4)
        bbox_tensor = np.zeros((len(new_logo_region)+len(bboxes_tamedAI),4),dtype=np.float32)
        for i in range(len(new_logo_region)):
            #create np array from tuple
            bbox_tensor[i] =  new_logo_region[i]
        for i in range(len(bboxes_tamedAI)):
            bbox_tensor[i+len(new_logo_region)] = np.array(bboxes_tamedAI[i])
        #create tensor from np array
        bbox_tensor = torch.from_numpy(bbox_tensor)
        new_bboxes = BoxList(bbox_tensor, (pad_h,pad_w), mode="xyxy")

        dim = model.decoder.head.image_embedding.rnn.embedding.out_features
        aligned_roi_height = cfg.MODEL.ALIGN.POOLER_RESOLUTION[1]
        if new_bboxes.bbox.size()[0] == 0:
            #imgs_embedding_nor = torch.zeros([0,aligned_roi_height*dim])
            #just visualize the image
            #cv2.imshow('bbox_visualize_image',new_image)
            #cv2.waitKey(20)
            #cv2.imwrite('F:\\code\\Research\\STR-TDSL\\test\\test_images_'+str(frame_idx).zfill (4)+'.jpg',frame)
            frame_idx += 1
            #out.write(bbox_visualize_image)
            continue
        #visualize the text boxes
        bbox_visualize_image = new_image.copy()
        for i in range(new_bboxes.bbox.size()[0]):
            bbox = new_bboxes.bbox[i]
            bbox = bbox.detach().cpu().numpy()
            cv2.rectangle(bbox_visualize_image,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),2)
        #cv2.imshow('bbox_visualize_image',bbox_visualize_image)
        #cv2.waitKey(20)
        with torch.no_grad():
            rois = model.decoder.head.pooler(rec_features, [new_bboxes])
            imgs_embedding = model.decoder.head.image_embedding(rois)
            #image_embedding is a tensor of shape [N,15,128], N is the number of text boxes
            imgs_embedding_nor = nn.functional.normalize((imgs_embedding).tanh().view(imgs_embedding.size(0),-1))
        
            #check if we find any matches
            imgs_embedding_nor = imgs_embedding_nor.detach().cpu().numpy().copy()

        matches = []    
        for i in range(imgs_embedding_nor.shape[0]):
            max_sim = 0
            max_k = -1
            for k in range(ref_words_embedding.shape[0]):
                #cosine similarity
                similarity = np.dot(imgs_embedding_nor[i],ref_words_embedding[k])
                if similarity > max_sim:
                    max_sim = similarity
                    max_k = k
            if max_sim>sim_th:
                matches.append((new_bboxes.bbox[i].detach().cpu().numpy(),imgs_embedding_nor[i],ref_words[max_k],max_sim))
        #now we have matches, we visualize them
        if len(matches) == 0:
            #v2.imwrite('F:\\code\\Research\\STR-TDSL\\test\\test_images_'+str(frame_idx).zfill (4)+'.jpg',frame)
            frame_idx += 1
            #out.write(bbox_visualize_image)
            continue
        track_bbox_forward(model,tracker,matches,buffered_images,tracked_results,recognition_result)
        for bbox,_,label,similarity in matches:
            viz_text = '{} {:.3f}'.format(label, similarity)
            viz_text_rect = cv2.getTextSize(viz_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            #text height should be 20% of the bbox height
            scale = ((bbox[3]-bbox[1])*bbox_visualize_image.shape[0])*0.2/viz_text_rect[1]
            if(bbox[1]<5):
                cv2.putText(bbox_visualize_image,viz_text,(int(bbox[0]),int(bbox[3])),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,255),2)
            else:
                cv2.putText(bbox_visualize_image,viz_text,(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,255),2)
            cv2.rectangle(bbox_visualize_image,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
        #cv2.imshow('bbox_visualize_image',bbox_visualize_image)
        #cv2.waitKey(20)
        #write image to a folder,frame_idx should be 4 significant digits padded with 0
        
        #cv2.imwrite('F:\\code\\Research\\STR-TDSL\\test\\test_images_'+str(frame_idx).zfill (4)+'.jpg',frame)
        #add items to the dictionary detections
        stored_matches = [(bbox.tolist(),label,float(similarity)) for bbox,_,label,similarity in matches]
        recognition_result[frame_idx] =stored_matches
        #out.write(bbox_visualize_image)
        frame_idx += 1


    with open("F:\\code\\Research\\STR-TDSL\\test\\my_data.json", "w") as f:
        #write each line to the file
        json.dump(recognition_result, f,indent=2)