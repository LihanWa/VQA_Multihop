import networkx as nx
import argparse
import os
import sys
import numpy as np
import json
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision.transforms as transforms

from groundingdino.util.inference import load_model, predict, annotate
import cv2
from matplotlib import pyplot as plt
from collections import defaultdict
import gc
# sys.path.append('/root/autodl-tmp/GroundedSAM_src/Grounded-Segment-Anything')
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import pickle
def save_graph(graph, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(graph, f)

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    new_size = (640, 427) 
    image_pil = image_pil.resize(new_size, Image.LANCZOS)
    transform = T.Compose(
        [
            # T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cuda"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        # import pdb; pdb.set_trace()
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


# work_path = '/root/autodl-tmp/GroundedSAM_src/Grounded-Segment-Anything'
config_file = '/root/projects/mmcot/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py'  # change the path of the model config file
# # ram_checkpoint = '/root/autodl-tmp/eva_weights/ram_plus_swin_large_14m.pth'  # change the path of the model
grounded_checkpoint = '/root/projects/mmcot/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth'  # change the path of the model

# device = 'cuda'
# # det_model = load_model(config_file, grounded_checkpoint, device=device)
# config_file = '/root/projects/mmcot/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'  # change the path of the model config file
# ram_checkpoint = '/root/autodl-tmp/eva_weights/ram_plus_swin_large_14m.pth'  # change the path of the model
# grounded_checkpoint = '/root/projects/mmcot/GroundingDINO/weights/groundingdino_swint_ogc.pth'  # change the path of the model

device = 'cuda'

def cropping(    
    model,
    image_path, 
    phrase,
    box_threshold = 0.25,
    text_threshold = 0.1,
    iou_threshold = 0.38,
    device='cuda'):
    
    image_pil, image = load_image(image_path)
    boxes_filt, scores, pred_phrases = get_grounding_output(
    model, image, phrase, box_threshold, text_threshold, device=device)
    # print("boxes_filt2: ",boxes_filt)
    # boxes_filt, scores, pred_phrases = get_grounding_output(
    # model, image, phrase, box_threshold, text_threshold, device=device)
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    # import pdb; pdb.set_trace()
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    return boxes_filt, pred_phrases, image_pil, image


from PIL import Image, ImageDraw, ImageFont

from collections import defaultdict

objname2id = defaultdict(dict)
# objname2id - {
#   'woman' = [1,2,4]
#}

def draw_circle_with_number(image_path, bbox, pred_phrases, number, output_path, qid, prev_image=None, topk=5):
    # number is the tacker for how many objects we extracted so far
    # Open the input image
    if not prev_image:
        image = Image.open(image_path)
    else:
        image = prev_image
    number = int(number)

    # Initialize drawing context
    draw = ImageDraw.Draw(image)
    
    # Unpack bounding box coordinates
    count = 1
    for idx, (bbox, phrase) in enumerate(zip(bbox, pred_phrases)):
        if idx >= topk:
            break
        
        if objname2id[qid].get(phrase, False):
            objname2id[qid][phrase].append(number + count)
        else:
            objname2id[qid][phrase] = [number + count]

        bbox=[int(num) for num in bbox]
        x_min, y_min, x_max, y_max = bbox

        # Calculate circle center coordinates
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Calculate circle radius
        radius = max(min(x_max - x_min, y_max - y_min) // 2 * 0.2, 15)

        # Draw white circle
        draw.ellipse([(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)], fill='white')

        # Load a font
        font_size = 24
        font = ImageFont.truetype('/root/projects/code_lihan/arial.ttf', font_size)

        # # Get text size
        text_width, text_height = draw.textsize(str(number), font=font)

        max_text_width = 2 * (radius - 2)  # Subtract a small margin
        max_text_height = 2 * (radius - 2)  # Subtract a small margin

    # Adjust font size to fit within circle size
        while text_width > max_text_width or text_height > max_text_height:
            font_size -= 1
            font = ImageFont.truetype("/root/projects/code_lihan/arial.ttf", font_size)
            text_width, text_height = draw.textsize(str(number), font=font)

        # Calculate text position
        text_x = center_x - text_width // 2
        text_y = center_y - text_height // 2

        
        # Draw number inside the circle
        draw.text((text_x, text_y), str(number+count), fill='black', font=font)
        count += 1
    # Save the modified image

    return number+count, image

# Example usage:
# Replace 'input_image.jpg' with the path of your input image
# Replace (x_min, y_min, x_max, y_max) with the bounding box coordinates
# Replace 'output_image.jpg' with the desired output path
# Replace '42' with the desired number
# draw_circle_with_number('input_image.jpg', (x_min, y_min, x_max, y_max), 42, 'output_image.jpg')



objname2id = defaultdict(dict)
# objname2id - {
#   'woman' = [1,2,4]
#}
import re
def new_draw_number_save(image_path, bbox, obj_ids,obj, output_path):
    def draw_box(area,img):
        draw = ImageDraw.Draw(img)
        draw.rectangle(area, outline="yellow", width=(idx+2)%3+1) 
        return img
    def draw_text_mark( text, mark,area,img):
        #text font size
        draw = ImageDraw.Draw(img)
        image_width, image_height = img.size
        font_size_text = int(image_width * 3 / 100)
        font_size_mark=font_size_text+5
        # print(font_size_text)
        try:
            text_font = ImageFont.truetype("/root/projects/code_lihan/arial.ttf", font_size_text)  # 修改为实际可用的字体路径和大小
            mark_font = ImageFont.truetype("/root/projects/code_lihan/arial.ttf", font_size_mark)  # 修改为实际可用的字体路径和大小
        except IOError:
            text = ImageFont.load_default()
        #text area
        text_width, text_height = draw.textsize(text, font=text_font)
        mark_width, mark_height = draw.textsize(mark, font=mark_font)

        text_x = area[0]+3
        mark_x = text_x+text_width+8
        if ((area[2]-area[0])<3*(text_width)) or ((area[3]-area[1])<3*mark_height):
            text_y = max(area[1] - mark_height,0)
            mark_y=text_y
        else:
            text_y = area[1]
            mark_y=text_y
        text_area = (text_x, text_y, text_x + text_width, text_y + text_height)
        mark_area = (mark_x, mark_y, mark_x + mark_width, mark_y + mark_height)
        #text color
        text_region = img.crop(text_area)
        mark_region = img.crop(mark_area)
        text_average_color = np.array(text_region).mean(axis=(0, 1))
        mark_average_color = np.array(mark_region).mean(axis=(0, 1))
        # print(text_average_color)
        # print(mark_average_color)
        red_color = np.array([255, 0, 0])
        text_distance = np.linalg.norm(text_average_color - red_color)
        mark_distance = np.linalg.norm(mark_average_color - red_color)
        text_color = (255, 255, 255) if text_distance < 120 else (255, 0, 0)
        mark_color = (255, 255, 255) if mark_distance < 120 else (255, 0, 0)

        draw.text((text_x, text_y), text, font=text_font, fill=text_color)
        draw.text((mark_x, mark_y), mark, font=mark_font, fill=mark_color)
        return img

    
    image = Image.open(image_path)
    new_size = (640, 427)  
    image = image.resize(new_size, Image.ANTIALIAS)

    count = 1
    for idx, (bbox, obj_id) in enumerate(zip(bbox, obj_ids)):



        # new_phrase = obj.split('(')[0]
        # pattern = r'[^a-zA-Z0-9\s-]'  # Matches any character that is not alphanumeric or whitespace
        # new_phrase = re.sub(pattern, '', new_phrase)
        bbox=[int(num) for num in bbox]
        x_min, y_min, x_max, y_max = bbox
        area=(x_min, y_min, x_max, y_max)
        draw_text_mark(obj.lower(), str(obj_id),area,image)
        draw_box(area,image)  
    image.save(output_path)



def draw_number(image_path, bbox, pred_phrases, obj,number, output_path, qid, prev_image=None,prev_image_black=None, topk=5):
    # number is the tacker for how many objects we extracted so far
    # Open the input image
    

    def blackout_region(image_black, xmin, ymin, xmax, ymax):
        image_black = image_black.convert("RGBA")
        pixels = image_black.load()
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                pixels[x, y] = (0, 0, 0, 255)  # 设为黑色（0, 0, 0, 255）
        return image_black
    def draw_box(area,img):
        draw = ImageDraw.Draw(img)
        draw.rectangle(area, outline="yellow", width=(idx+2)%3+1) 
        return img
    def draw_text_mark( text, mark,area,img):
        #text font size
        
        draw = ImageDraw.Draw(img)
        image_width, image_height = img.size
        font_size_text = int(image_width * 3 / 100)
        font_size_mark=font_size_text+5
        # print(font_size_text)
        try:
            text_font = ImageFont.truetype("/root/projects/code_lihan/arial.ttf", font_size_text)  # 修改为实际可用的字体路径和大小
            mark_font = ImageFont.truetype("/root/projects/code_lihan/arial.ttf", font_size_mark)  # 修改为实际可用的字体路径和大小
        except IOError:
            text = ImageFont.load_default()
        #text area
        text_bbox = draw.textbbox((0, 0), text, font=text_font)
        mark_bbox = draw.textbbox((0, 0), mark, font=mark_font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        mark_width, mark_height = mark_bbox[2] - mark_bbox[0], mark_bbox[3] - mark_bbox[1]

        # print("text_width,text_height",(text_width,text_height))
        # print("mark_width,mark_height",(mark_width,mark_height))
        text_x = area[0]+3
        mark_x = text_x+text_width+8
        if ((area[2]-area[0])<3*(text_width)) or ((area[3]-area[1])<3*mark_height):
            text_y = max(area[1] - mark_height,0)
            mark_y=text_y
        else:
            text_y = area[1]
            mark_y=text_y
        text_area = (text_x, text_y, text_x + text_width, text_y + text_height)
        mark_area = (mark_x, mark_y, mark_x + mark_width, mark_y + mark_height)
        #text color
        text_region = img.crop(text_area)
        mark_region = img.crop(mark_area)
        text_average_color = np.array(text_region).mean(axis=(0, 1))
        mark_average_color = np.array(mark_region).mean(axis=(0, 1))
        # print(text_average_color)
        # print(mark_average_color)
        red_color = np.array([255, 0, 0])
        text_distance = np.linalg.norm(text_average_color - red_color)
        mark_distance = np.linalg.norm(mark_average_color - red_color)
        text_color = (255, 255, 255) if text_distance < 120 else (255, 0, 0)
        mark_color = (255, 255, 255) if mark_distance < 120 else (255, 0, 0)
        
    

        draw.text((text_x, text_y), text, font=text_font, fill=text_color)
        draw.text((mark_x, mark_y), mark, font=mark_font, fill=mark_color)
        return img

    
    obj_dict = dict()
    if not prev_image:
        image = Image.open(image_path)
        image_mode = image.mode
        if image_mode == 'L':
            image = image.convert('RGB')
    else:
        image = prev_image
    number = int(number)
    new_size = (640, 427)  
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    #open another one for black
    if not prev_image_black:
        image_black = Image.open(image_path)
    else:
        image_black = prev_image_black
    number = int(number)
    new_size = (640, 427)  
    image_black = image_black.resize(new_size, Image.Resampling.LANCZOS)
    # Initialize drawing context
    # draw = ImageDraw.Draw(image)
    
    # Unpack bounding box coordinates
    count = 1
    for idx, (bbox, phrase) in enumerate(zip(bbox, pred_phrases)):
        if idx >= topk:
            break


        # new_phrase = phrase.split('(')[0]
        # pattern = r'[^a-zA-Z0-9\s-]'  # Matches any character that is not alphanumeric or whitespace
        # new_phrase = re.sub(pattern, '', new_phrase)
        new_phrase=obj
        bbox=[int(num) for num in bbox]
        x_min, y_min, x_max, y_max = bbox
        new_phrase1 = f'{new_phrase}_{x_min}_{y_min}_{x_max}_{y_max}'
        obj_dict[new_phrase1] = number + count

        # center_x,center_y=centers[idx]
        # font_size = 42
        # font = ImageFont.truetype('/root/projects/code_lihan/arial.ttf', font_size)
        # text_width, text_height = draw.textsize(str(number), font=font)
        # text_x = center_x - text_width // 2
        # text_y = center_y - text_height // 2
        
        area=(x_min, y_min, x_max, y_max)
        draw_text_mark(new_phrase, str(number+count),area,image)
        draw_box(area,image)  
        image_black=blackout_region(image_black, x_min, y_min, x_max, y_max)
        # Draw number inside the circle
        # draw.text((text_x, text_y), str(number+count), fill='red', font=font)
        count += 1
    # Save the modified image

    return number+count-1, image,image_black, obj_dict

# Example usage:
# Replace 'input_image.jpg' with the path of your input image
# Replace (x_min, y_min, x_max, y_max) with the bounding box coordinates
# Replace 'output_image.jpg' with the desired output path
# Replace '42' with the desired number
# draw_circle_with_number('input_image.jpg', (x_min, y_min, x_max, y_max), 42, 'output_image.jpg')

# with open('MIC/decomp_testvalb_cate.pkl', 'rb') as ff:
#     decomp_cate = pickle.load(ff)

# qs2qid = dict()
# qs2imgid = dict()


# with open('gqa/testdev_balanced_questions.json', 'r') as f:
#     testdev = json.load(f)

# from tqdm import tqdm
# for k, qdict in tqdm(testdev.items()):
#     question = qdict['question']
#     qid = k
#     imageid = qdict['imageId']
#     qs2qid[question] = qid
#     qs2imgid[question] = imageid
#     # num_of_nouns = count_nouns(question)
    
#     # count_dict[num_of_nouns] += 1
#     # if num_of_nouns >= 3:
#     #     long_dict[num_of_nouns].append(question)

# import pickle
# with open('MIC/objdict.pkl', 'rb') as fobj:
#     objdict = pickle.load(fobj)

# 只考虑两个物体可能重合的情况
def get_area(xmin, ymin, xmax, ymax):
    return (xmax - xmin) * (ymax - ymin)
def new_centers(boxes_filt_p):
    def is_overlapping(x1, y1, x2, y2, threshold=10):
        return abs(x1 - x2) < threshold and abs(y1 - y2) < threshold
    def get_center(xmin, ymin, xmax, ymax):
        return (xmin + xmax) / 2, (ymin + ymax) / 2

    def restrict_center(new_center_m_x,new_center_m_y,xmin_m, ymin_m, xmax_m, ymax_m,threshhold=50):
        x_threshhold=min((xmax_m-xmin_m)/2,threshhold)
        y_threshhold=min((ymax_m-ymin_m)/2,threshhold)
        new_center_m_x=min(xmax_m-x_threshhold,new_center_m_x)
        new_center_m_x=max(xmin_m+x_threshhold,new_center_m_x)
        new_center_m_y=min(ymax_m-y_threshhold,new_center_m_y)
        new_center_m_y=max(ymin_m+y_threshhold,new_center_m_y)
        return new_center_m_x,new_center_m_y
    #find centers and x y pos
    centers=[]
    for pos in boxes_filt_p:
        xmin, ymin, xmax, ymax=pos
        centers.append(get_center(xmin, ymin, xmax, ymax))
    overlaps=[]
    for c_i in range(len(centers)):
        x1,y1=centers[c_i]
        for c_j in range(c_i+1,len(centers)):
            x2,y2=centers[c_j]
            if is_overlapping(x1,y1,x2,y2,40):
                overlaps.append((c_i,c_j))
    #move label position if overlaps
    for overlap in overlaps:
        obj1_i=overlap[0]
        obj2_i=overlap[1]
        xmin, ymin, xmax, ymax=boxes_filt_p[obj1_i]
        A_obj1=get_area(xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax=boxes_filt_p[obj2_i]
        A_obj2=get_area(xmin, ymin, xmax, ymax)
        #index of obj
        obj_move_i= obj1_i if A_obj1>A_obj2 else obj2_i
        obj_stay_i= obj1_i if A_obj1<A_obj2 else obj2_i
        #centers of move and stay obj
        x_m,y_m=centers[obj_move_i]
        x_s,y_s=centers[obj_stay_i]
        # move directions
        move_xdir= "left" if x_m<x_s else "right"
        move_ydir= "down" if y_m<y_s else "up"
        #pos of move and stay obj
        xmin_m, ymin_m, xmax_m, ymax_m=boxes_filt_p[obj_move_i]
        xmin_s, ymin_s, xmax_s, ymax_s=boxes_filt_p[obj_stay_i]
        #inside 1/3
        move_xlen=(xmin_s-xmin_m if move_xdir== "left" else xmax_m-xmax_s)*1/3
        move_ylen=(ymin_s-ymin_m if move_ydir== "down" else ymax_m-ymax_s)*1/3
        #new center of moving obj
        new_center_m_x=xmin_m+move_xlen if move_xdir== "left" else xmax_m-move_xlen
        new_center_m_y=ymin_m+move_ylen if move_ydir== "down" else ymax_m+move_ylen
        # restrict centers in case of moving too much to border
        new_center_m_x,new_center_m_y=restrict_center(new_center_m_x,new_center_m_y,xmin_m, ymin_m, xmax_m, ymax_m)
        centers[obj_move_i]=(new_center_m_x,new_center_m_y)
    return centers
 # labled img save path

counter2 = 0
# topk=5
def adjust_indexed_arr(indexed_arr, i, temp=1):
    if i < 1 or i - temp < 0:  # 终止条件
        return indexed_arr
    
    if indexed_arr[i][0] - indexed_arr[i-1][0] < 24:
        if abs(indexed_arr[i][1] - indexed_arr[i-temp][1]) < 65:
            indexed_arr[i-1] = (max(0, indexed_arr[i][0] - 24), indexed_arr[i-1][1], indexed_arr[i-1][2])
        else:
            return adjust_indexed_arr(indexed_arr, i, temp + 1)  # 递归调用
    return indexed_arr
def adjust_numbers(arr):
    indexed_arr=arr
    n = len(indexed_arr)
    for j in range(3):
        indexed_arr.sort(key=lambda x: x[0])
        # print("before sort",indexed_arr)
        for i in range(1, n):
            for tmp in range(1,i+1):
                if indexed_arr[i][0] - indexed_arr[i-tmp][0] < 48:
                    if abs(indexed_arr[i][1] - indexed_arr[i-tmp][1]) < 75:
                        indexed_arr[i-tmp] = (max(0, indexed_arr[i][0] - 48), indexed_arr[i-tmp][1], indexed_arr[i-tmp][2])
                        break
                else: 
                    break
    # print("mid sort",indexed_arr)
    if indexed_arr[0][0] < 0:
        indexed_arr[0] = [0,indexed_arr[0][1],indexed_arr[0][2]]
    indexed_arr.sort(key=lambda x: x[2])
    print("after sort",indexed_arr)
    adjusted_arr = [item[0] for item in indexed_arr]
    return adjusted_arr
class ObjModel:
    def __init__(self, grounded_checkpoint=grounded_checkpoint, config_file=config_file, device='cuda'):
        self.model = load_model(config_file, grounded_checkpoint, device=device)
        self.model = self.model.to('cuda:0')
        print('==========loading model successfully=============')

    def find_obj(self, objlist, img_name,img_name_out,img_ori='/root/projects/mmcot/gqa/images',out_origin= './test_morethan2',need_obj_boxes=False):
        obj_dict = defaultdict(dict)
        instance_dict = defaultdict(int)
        full_image_path = f'{img_ori}/{img_name}'
        output_path = f'{out_origin}/{img_name_out}.jpg'#f'{out_origin}/{qid}/{imgid}.png'
        output_path_black = f'{out_origin}/{img_name_out}_black.jpg'#f'{out_origin}/{qid}/{imgid}.png'
        # for obj in objlist: #process substring
        # temp_dict = defaultdict(dict)
        # objs = ' . '.join(objlist)
        objflag = 0
        # import pdb; pdb.set_trace()

        number = 0
        prev_img = None
        prev_img_black=None
        boxes_filt_p=[]
        collect_boxes_filt=[]
        collect_pred_phrases=[]
        ymax_numbers=[]

        if len(objlist)==0:
            image_pil, image = load_image(full_image_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image_pil.save(output_path)
            return {},[],{}
        obj_not_found=[]
        
        for obj in objlist:
            obj=obj.lower()
            boxes_filt, pred_phrases, image_pil, image = cropping(self.model,full_image_path, obj) # xiugai self.model
            #make the filt a bit larger
            for i in range(len(boxes_filt)):
                boxes_filt[i][0]=max(0,boxes_filt[i][0]-3)
                boxes_filt[i][1]=max(0,boxes_filt[i][1]-3)
                boxes_filt[i][2]=min(640,boxes_filt[i][2]+3)
                boxes_filt[i][3]=min(427,boxes_filt[i][3]+3)
            print("boxes_filt--",(boxes_filt))
            print("pred_phrase--",pred_phrases)
            to_be_deleted = set()

            # 去除重复的大边框
            if(len(boxes_filt)==0):
                print("Did not find: ",obj)
                obj_not_found.append(obj)
            if(len(boxes_filt)>0):
                for i in range(len(boxes_filt)):
                    for j in range(len(boxes_filt)):
                        if i != j and i not in to_be_deleted:
                            # 检查 bbox[i] 是否包含 bbox[j]
                            


                            if get_area(boxes_filt[i, 0],boxes_filt[i, 1],boxes_filt[i, 2],boxes_filt[i, 3])>get_area(boxes_filt[j, 0],boxes_filt[j, 1],boxes_filt[j, 2],boxes_filt[j, 3])and(boxes_filt[i, 0] <= boxes_filt[j, 0]+4 and boxes_filt[i, 1] <= boxes_filt[j, 1]+4 and
                                boxes_filt[i, 2] >= boxes_filt[j, 2]-4 and boxes_filt[i, 3] >= boxes_filt[j, 3]-4):
                                to_be_deleted.add(i)

                
                boxes_filt = torch.stack([boxes_filt[i] for i in range(len(boxes_filt)) if i not in to_be_deleted])
                pred_phrases = [pred_phrases[i] for i in range(len(pred_phrases)) if i not in to_be_deleted]
                print("=======================")
                # print("boxes_filt--",(boxes_filt))
                # for i in boxes_filt:
                #     print((i[2]-i[0]),i[3]-i[1])
                # print("pred_phrase--",pred_phrases)

            #收集
            collect_boxes_filt.append(boxes_filt)
            collect_pred_phrases.append(pred_phrases)

            for i in range(len(boxes_filt)):
                filt=boxes_filt[i]
                boxes_filt_p.append([int(num) for num in filt])
                ymax_numbers.append((int(filt[1]),int(filt[0]),len(ymax_numbers)))
        
        centers=new_centers(boxes_filt_p)
        #更新边框上边来调整text mark位置
        if len(ymax_numbers)>1:
            adjusted_numbers = adjust_numbers(ymax_numbers)
            cnt=0
            for i in boxes_filt_p:
                i[1]=adjusted_numbers[cnt]
            cnt=0
            for tensor in collect_boxes_filt:
                for row in tensor:
                    row[1]=adjusted_numbers[cnt]
                    cnt+=1
            print("Adjusted collect_boxes_filt",collect_boxes_filt)
            if cnt!=len(adjusted_numbers):
                sys.exit("adjusted_numbers is not correct") 
        #实验用的（生成紫色标记
        def annotate(image, boxes, scores, phrases):
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            font_size = 20
            font = ImageFont.truetype('/root/projects/code_lihan/arial.ttf', font_size)

            # # Get text size
            for box, score, phrase in zip(boxes, scores, phrases):
                xmin, ymin, xmax, ymax = box
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
                draw.text((xmin, ymin), f"{phrase}", fill="purple", font=font)
            return image
        
        #逐个node标记和生成obj-dict obj_filt_dict
        obj_filt_dict={}
        for obj_i in range(len(objlist)):
            topk=5
            pred_phrases=collect_pred_phrases[obj_i]
            boxes_filt=collect_boxes_filt[obj_i]
            obj_filt_dict[objlist[obj_i].lower()]=boxes_filt.tolist()
            center_obj=centers[number:number+len(boxes_filt)]
            number, out_img,out_img_black, obj2num = draw_number(full_image_path, boxes_filt, pred_phrases, objlist[obj_i].lower(),number, output_path, 1, prev_img,prev_img_black)
            prev_img = out_img
            prev_img_black=out_img_black
            #实验用的（生成紫色标记
            if need_obj_boxes:
                if obj_i==0:
                    annotated_frame=annotate((image_pil),boxes_filt,torch.tensor([0]*len(pred_phrases)),pred_phrases)
                else:
                    annotated_frame=annotate((annotated_frame),boxes_filt,[0]*len(pred_phrases),pred_phrases)
                    # print("annotated_frame type",type(annotated_frame))
                if obj_i==len(objlist)-1:
                    (annotated_frame).save(f'{out_origin}/{img_name_out}_obj_box.jpg')
            #生成obj-dict,
            for idx, (bbox, phrase) in enumerate(zip(boxes_filt, pred_phrases)):
                
                temp_dict = dict()
                if idx >= topk:
                    break
                # new_phrase = phrase.split('(')[0]
                #TODO get accurate score
                score = phrase.split('(')[1]
                # new_phrase.replace('SEP', '')
                score = float(score.replace(')', ''))
                # pattern = r'[^a-zA-Z0-9\s-]'  # Matches any character that is not alphanumeric or whitespace
                # new_phrase = re.sub(pattern, '', new_phrase)
                new_phrase=objlist[obj_i].lower()
                #e.g. "apple"
                instance_dict[new_phrase] += 1
                instance_name = f'{new_phrase}_{instance_dict[new_phrase]}'
                bbox=[int(num) for num in bbox]
                x_min, y_min, x_max, y_max = bbox

                    # plist = []
                    # for ph in new_phrase.split(' '):
                    #     if ph not in plist:
                    #         plist.append(ph)
                    #     else:
                    #         continue
                    # new_phrase2 = ' '.join(plist)
                # import pdb; pdb.set_trace()
                
                temp_dict['number'] = obj2num[f'{new_phrase}_{x_min}_{y_min}_{x_max}_{y_max}']
                temp_dict['bbox'] = [int(num) for num in bbox] #x_max,
                temp_dict['score'] = score
                obj_dict[new_phrase][instance_name] = temp_dict
        
        if out_img: # save generated images
            # import pdb; pdb.set_trace()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out_img.save(output_path)
        # if out_img_black: # save generated images
        #     # import pdb; pdb.set_trace()
        #     out_img_black = out_img_black.convert('RGB')
            # os.makedirs(os.path.dirname(output_path_black), exist_ok=True)
            # out_img_black.save(output_path_black)
            # img=Image.open(output_path)
            # img.show()
        return obj_dict,obj_not_found,obj_filt_dict
        

# model = ObjModel(grounded_checkpoint, config_file)

# # 调用方法
# objlist = ['bottle . cat . laptop']
# img_name = 'computer_twoBottles.jpg'
# testobj = model.find_obj(objlist, img_name)
# print(testobj)


    

# for q, objlist in tqdm(objdict.items()):
#     # if counter2 > 1000:
#     #     import pdb;pdb.set_trace()
#     #     break
#     counter2 += 1
#     imgid = qs2imgid[q]
#     qid = qs2qid[q]
#     full_image_path = img_origin + imgid + '.jpg'
#     output_path = f'{out_origin}/{qid}/{imgid}.png'
#     # print(imgid)
#     # print(qdict)
#     number = 0
#     # print(objlist)
   
#     objlist = [obj.replace('this ', '') for obj in objlist if 'that' not in obj and 'top' not in obj and 'wh' not in obj]
#     objlist = [obj.replace('other ', '') for obj in objlist if 'you' not in obj and 'any' not in obj and 'some' not in obj]
#     objlist = [obj.replace('bottom ', '') for obj in objlist if 'size' not in obj and 'color' not in obj and 'side' not in obj]
#     objlist = list(set(objlist))
#     if len(objlist) >= 3:
#         pass
#         # import pdb; pdb.set_trace()
#     new_objlist = []
#     for obs in objlist: #process substring
#         objflag = 0
#         for obj2 in objlist:
#             if obs in obj2 and obs != obj2: #a substring others and not a substring of itself.
#                 objflag = 1
#                 break
#         if not objflag:
#             new_objlist.append(obs)
        


#     # obj = ' . '.join(new_objlist)
#     # print(obj)
#     prev_img = None
#     for obj in new_objlist:
#         # import pdb; pdb.set_trace()
#         boxes_filt, pred_phrases, image_pil, image = cropping(det_model, full_image_path, obj)
#         number, out_img = draw_number(full_image_path, boxes_filt, pred_phrases, number, output_path, qid, prev_img, 3)
#         prev_img = out_img
#     # prev_image = None
#     # for subq, subqdict in qdict.items():
#     #     # if not subqdict:
#     #     #     continue
#     #     obj = ''
#     #     # try:
#     #     if subqdict['category'] == 'detection':
#     #         obj = ' . '.join(subqdict['objects'].split(', '))
#     #         # print(obj)
#     #         boxes_filt, pred_phrases, image_pil, image = cropping(det_model, full_image_path, obj)
#     #             # save_obj_image(full_image_path, image_pil, boxes_filt, pred_phrases, topk=1)
#     #         pred_phrases = [phrase.split('(')[0] for phrase in pred_phrases]
#     #         # print(pred_phrases)
#     #         number, prev_image = draw_number(full_image_path,boxes_filt, pred_phrases, number, output_path, qid, prev_image, topk=3)
#     if out_img:
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         out_img.save(output_path)
#         # except:
#         #     # print(subq, obj)
#         #     continue

# print(objname2id)
# # import pdb;pdb.set_trace()
# with open('objname2mark.pkl', 'wb') as ff:
#     pickle.dump(objname2id, ff)