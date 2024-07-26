from langchain.tools import tool
from langchain.tools.render import format_tool_to_openai_function
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
import bs4
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import FunctionMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from PIL import Image,ImageDraw,ImageFont
from langgraph.prebuilt import ToolExecutor
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pprint import pprint
import base64
import json
import mimetypes
import os
import requests
import sys
import cv2
import numpy as np
import re
from gpt4_img import create_payload_edge,judge_few_objs,query_openai,create_payload_node,create_payload_size_tool

#========== Functions for Gpt4 vqa tool=========

@tool("VQA_tool", return_direct=False)
def VQA_tool(info:dict) -> str:
    # '''Wikipedia tool. This tool is used to provide background information from wikipedia (such as history, the job of a person, the habits of animals).'''
    '''This tool is ChatGPT4, which is very powerful and useful. '''

    image_path=info['img']
    # print("image_pathimage_pathimage_path",image_path)
    question_type=info['question_type']
    # label_sentence=dict_to_sentence(obj_dict)
    label_sentence=info['label_sentence']
    # print(question_type)
    if question_type=='edgeQ':
        payload = create_payload_edge(image_path, info['subQ'],label_sentence)

    if question_type=='nodeQ':
        payload = create_payload_node(image_path, info['subQ'],label_sentence)
    # print("payload",payload)
    # print("payloadpayload",payload)
    response = query_openai(payload)
    # print(response)
    res=response['choices'][0]['message']['content']
    print("Complete result from VQA tool: ",res)
    match = re.search(r"Answer:\s*(.*)", res)


    if match:
        res = match.group(1)
    # res="The current object is "+obj+ ". I have already used tool 'VQA_tool' to process it."
    # print(" The result of VQA tool: ",res)
    print('---------------')
    return res
# tools= [VQA_tool]
# tool_executor = ToolExecutor(tools)
@tool("size_tool", return_direct=False)
def size_tool(info:dict) -> str:
    # '''Wikipedia tool. This tool is used to provide background information from wikipedia (such as history, the job of a person, the habits of animals).'''
    '''This tool is ChatGPT4, which is very powerful and useful. '''

    image_path=info['img']
    question_type=info['question_type']
    label_sentence=info['label_sentence']
    print("info['subQ']",info['subQ'])
    question=info['subQ'][:-1]+f", compared with usual {info['obj']}?"
    payload = create_payload_size_tool(image_path, question,label_sentence)
    response = query_openai(payload)
    # print(response)
    res=response['choices'][0]['message']['content']
    print("Complete result from size tool: ",res)
    match = re.search(r"Answer:\s*(.*)", res)

    if match:
        res = match.group(1)
    print('---------------')
    return res


@tool("bottom_top_tool", return_direct=False)
def bottom_top_tool(info:dict) -> str:
    '''This tool can be used to determine if the object is at botom or top part of the image. '''
    question_type=info['question_type']
    question=info['subQ']
    question_words=question[:-1].split(" ")
    question_words=[word.lower() for word in question_words]
    axis_dict=json.loads(info['axis_dict'])
    axis_dict={int(k): v for k, v in axis_dict.items()}
    obj_dict=json.loads(info['obj_dict'])
    bracket_contents = re.findall(r'\[\s*\d+(?:,\s*\d+)*\s*\]', question)
    numbers = []
    numberstr_list=[]
    for content in bracket_contents:
        numbers.append(re.findall(r'\d+', content))
        numberstr_list.extend(re.findall(r'\d+', content))
    numberint_list=[int(num) for num in numberstr_list]
    print("number_list: ",numberstr_list)
    id_name_dict={}
    for num in numberint_list:
        for key, values in obj_dict.items():
            if num in values:
                id_name_dict[num]=key
                break
    
    if question_type=='nodeQ' :
        number_list = [int(num) for num in numbers[0]]
        img=Image.open(info['img'])
        answer=''
        for i in range(len(number_list)):
            obj_box=axis_dict[number_list[i]]
            obj_ymid=(obj_box[1]+obj_box[3])/2
            ylen=img.size[1]
            if obj_ymid<= ylen/2:
                answer_n=f"{id_name_dict[number_list[i]]} [{number_list[i]}] is on the top of the image. "
            else:
                answer_n=f"{id_name_dict[number_list[i]]} [{number_list[i]}] is on the bottom of the image. "
            answer+=answer_n
    return answer
@tool("left_right_tool", return_direct=False)
def left_right_tool(info:dict) -> str:
    # '''Wikipedia tool. This tool is used to provide background information from wikipedia (such as history, the job of a person, the habits of animals).'''
    '''This tool can be used to determine the left and right direction. If the question is asking left or right, you should use this tool'''
    question_type=info['question_type']
    question=info['subQ']
    question_words=question[:-1].split(" ")
    question_words=[word.lower() for word in question_words]
    axis_dict=json.loads(info['axis_dict'])
    axis_dict={int(k): v for k, v in axis_dict.items()}
    obj_dict=json.loads(info['obj_dict'])
    print('obj_dict',obj_dict)
    print('axis_dict',axis_dict)
    bracket_contents = re.findall(r'\[\s*\d+(?:,\s*\d+)*\s*\]', question)
    numbers = []
    numberstr_list=[]
    for content in bracket_contents:
        numbers.append(re.findall(r'\d+', content))
        numberstr_list.extend(re.findall(r'\d+', content))
    numberint_list=[int(num) for num in numberstr_list]
    print("number_list: ",numberstr_list)
    id_name_dict={}
    for num in numberint_list:
        for key, values in obj_dict.items():
            if num in values:
                id_name_dict[num]=key
                break
    
    # pattern = r'of the (image|picture|photo|photograph)'
    # of_the_match = re.search(pattern, question)
    if question_type=='nodeQ' :
        number_list = [int(num) for num in numbers[0]]
        img=Image.open(info['img'])
        answer=''
        for i in range(len(number_list)):
            obj_box=axis_dict[number_list[i]]
            obj_xmid=(obj_box[0]+obj_box[2])/2
            xlen=img.size[0]
            if obj_xmid<= xlen/2:
                answer_n=f"{id_name_dict[number_list[i]]} [{number_list[i]}] is on the left side of the image. "
            else:
                answer_n=f"{id_name_dict[number_list[i]]} [{number_list[i]}] is on the right side of the image. "
            answer+=answer_n

        
    if question_type=="edgeQ":
        
        pattern = r'\b(right|left)\b'
        l_r_list=re.findall(pattern, question, flags=re.IGNORECASE)
        l_r=l_r_list[0]
        answer=''
        number_list0=[int(num) for num in numbers[0]]
        number_list1=[int(num) for num in numbers[1]]
        for i in range(len(number_list0)):
            for j in range(len(number_list1)):
                num_i=number_list0[i]
                num_j=number_list1[j]
                # print(num_i)
                # print(axis_dict)
                obji_box=axis_dict[num_i]
                objj_box=axis_dict[num_j]
                obji_xmid=(obji_box[0]+obji_box[2])/2
                objj_xmid=(objj_box[0]+objj_box[2])/2
                if 'right' in l_r_list and len(l_r_list)==1:
                    if obji_xmid>=objj_box[2]:
                        answer=answer+f"{id_name_dict[num_i]} [{num_i}] is to the right of {id_name_dict[num_j]} [{num_j}]. "
                    elif objj_xmid>=obji_box[2]:
                        answer=answer+f"{id_name_dict[num_j]} [{num_j}] is to the right of {id_name_dict[num_i]} [{num_i}]. "
                    else:
                        if obji_xmid>objj_xmid:
                            answer=answer+f"{id_name_dict[num_i]} [{num_i}] is not to the right of {id_name_dict[num_j]} [{num_j}]. "
                        else:
                            answer=answer+f"{id_name_dict[num_j]} [{num_j}] is not to the right of {id_name_dict[num_i]} [{num_i}]. "
                elif 'left' in l_r_list and len(l_r_list)==1:
                    if obji_xmid<=objj_box[0]:
                        answer=answer+f"{id_name_dict[num_i]} [{num_i}] is to the left of {id_name_dict[num_j]} [{num_j}]. "
                    elif objj_xmid<=obji_box[0]:
                        answer=answer+f"{id_name_dict[num_j]} [{num_j}] is to the left of {id_name_dict[num_i]} [{num_i}]. "
                    else:
                        if obji_xmid<objj_xmid:
                            answer=answer+f"{id_name_dict[num_i]} [{num_i}] is not to the left of {id_name_dict[num_j]} [{num_j}]. "
                        else:
                            answer=answer+f"{id_name_dict[num_j]} [{num_j}] is not to the left of {id_name_dict[num_i]} [{num_i}]. "
                else:
                    if obji_xmid<=objj_xmid+4:
                        answer=answer+f"{id_name_dict[num_i]} [{num_i}] is to the left of {id_name_dict[num_j]} [{num_j}]. "
                    if obji_xmid>=objj_xmid-4:
                        answer=answer+f"{id_name_dict[num_i]} [{num_i}] is to the right of {id_name_dict[num_j]} [{num_j}]. "

    print(" The result of left right tool: ",answer)
    print('---------------')
    return answer

def final_left_right_tool(info:dict) -> str:
    
    axis_dict=json.loads(info['axis_dict'])
    axis_dict={int(k): v for k, v in axis_dict.items()}
    obj_dict=info['obj_dict']
    # bracket_contents = re.findall(r'\[\s*\d+(?:,\s*\d+)*\s*\]', question)
    # numbers = []
    # numberstr_list=[]
    # for content in bracket_contents:
    #     numbers.append(re.findall(r'\d+', content))
    #     numberstr_list.extend(re.findall(r'\d+', content))
    # numberint_list=[int(num) for num in numberstr_list]
    answer_node=''
    if info['side_of_the_match']:
        for k,v in obj_dict.items():
            img=Image.open(info['img'])
            for i in range(len(v)):
                obj_box=axis_dict[v[i]]
                obj_xmid=(obj_box[0]+obj_box[2])/2
                xlen=img.size[0]
                if obj_xmid<= xlen/2:
                    answer_n=f"{k} [{v[i]}] is on the left side of the image. "
                else:
                    answer_n=f"{k} [{v[i]}] is on the right side of the image. "
                answer_node+=answer_n
        answer_node=f'{answer_node}'
        # answer_node=f'The positions of entities on the image are: {answer_node}'
        if info['num_pair']>3 or info['left_right']==False:
            print("answer_node: ",answer_node)
            return answer_node
    if info['left_right']:
        pattern = r'\b(right|left)\b'
        question=info['question']
        l_r_list=re.findall(pattern, question, flags=re.IGNORECASE)
        l_r=l_r_list[0]
        answer=''
        answer_edge=''
        number_list=[]
        numberint_list=[]
        print("obj_dict",obj_dict)
        for i,(k,v) in enumerate(obj_dict.items()):
            number_list.append(v)
            numberint_list.extend(v)
        id_name_dict={}
        for num in numberint_list:
            for key, values in obj_dict.items():
                if num in values:
                    id_name_dict[num]=key
                    break
        for x in range(len(number_list)):
            for y in range(x+1,len(number_list)):
                number_list0=number_list[x]
                number_list1=number_list[y]
                for i in range(len(number_list0)):
                    for j in range(len(number_list1)):
                        num_i=number_list0[i]
                        num_j=number_list1[j]
                        # print(num_i)
                        # print(axis_dict)
                        obji_box=axis_dict[num_i]
                        objj_box=axis_dict[num_j]
                        obji_xmid=(obji_box[0]+obji_box[2])/2
                        objj_xmid=(objj_box[0]+objj_box[2])/2
                        print(obji_box,objj_box)
                        print(id_name_dict[num_i],obji_xmid)
                        print(id_name_dict[num_j],objj_xmid)
                        if obji_xmid<objj_xmid-3:
                            if l_r=='right' and len(l_r_list)==0:
                                answer_e=answer+f"{id_name_dict[num_i]} [{num_i}] is not to the right of {id_name_dict[num_j]} [{num_j}]. "
                            else:
                                answer_e=answer+f"{id_name_dict[num_i]} [{num_i}] is to the left of {id_name_dict[num_j]} [{num_j}]. "
                        elif obji_xmid>objj_xmid+3:
                            if l_r=='left' and len(l_r_list)==0:
                                answer_e=answer+f"{id_name_dict[num_i]} [{num_i}] is not to the left of {id_name_dict[num_j]} [{num_j}]. "
                            else:
                                answer_e=answer+f"{id_name_dict[num_i]} [{num_i}] is to the right of {id_name_dict[num_j]} [{num_j}]. "
                        else: continue
                        answer_edge+=answer_e
        if answer_edge!='':
            # answer_edge=f'The relative spatial postitions among entities are: {answer_edge}'
            answer_edge=f'{answer_edge}'
        print("answer_edge: ",answer_edge)
        answer=answer_node+answer_edge
    print(" The result of left right tool: ",answer)
    print('---------------')
    return answer
def get_area(xmin, ymin, xmax, ymax):
    return (xmax - xmin) * (ymax - ymin)
@tool("judge_obj_tool", return_direct=False)
def judge_obj_tool(info:dict) -> str:
    ''' '''
    answer=''
    obj=info['obj']
    img_file=info['img'].split('/')[-1]#1_234.jpg
    img_id=img_file.split('.')[0] #1_234
    img_name=img_id.split('_')[1]#234
    image_dir=f'/root/projects/mmcot/gqa/images/{img_name}.jpg' #234.jpg
    obj_dict=json.loads(info['obj_dict'])
    obj_filt_dict=json.loads(info['obj_filt_dict'])
    
    def extract_and_enhance(name,image_path, area, scale_factor,enlarge=True):
        new_size = (640, 427) 
        image = Image.open(image_path).convert("RGB")
        image = image.resize(new_size, Image.ANTIALIAS)
        wid=area[2]-area[0]
        hei=area[3]-area[1]
        ratio=0.2
        area=max(0,area[0]-wid*ratio),max(0,area[1]-hei*ratio),min(new_size[0],area[2]+wid*ratio),min(new_size[1],area[3]+hei*ratio)
        cropped_img = image.crop(area)
        if cropped_img.width<200:
            scale_factor=int(200/cropped_img.width)
            new_size = (int(cropped_img.width * scale_factor), int(cropped_img.height * scale_factor))
            cropped_img = cropped_img.resize(new_size, Image.LANCZOS)
            # enhanced_img = cv2.cvtColor(np.array(enlarged_img), cv2.COLOR_RGB2BGR)
        # else:
        #     area=area[0],area[1],area[2],area[3]
        
        # img_cv = cv2.cvtColor(np.array(enlarged_img), cv2.COLOR_RGB2BGR)
        # kernel = np.array([[0, -1, 0],
        #                 [-1, 5, -1],
        #                 [0, -1, 0]])
        # sharpened_img = cv2.filter2D(img_cv, -1, kernel)
        # enhanced_img = Image.fromarray(cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2RGB))
        dir=info['img'][:-4]
        path=f'{dir}_{name}.jpg'
        cropped_img.save(path)
        return path
    subimg_path=[]
    # print('obj_filt_dict',obj_filt_dict)
    # print('obj_dict',obj_dict)
    for area,idx in zip(obj_filt_dict[obj],obj_dict[obj]):
        subimg_path.append(extract_and_enhance(str(idx),image_dir,  area, 5,True))
    def merge_images(image_paths, output_path, marks,direction='horizontal', spacing=0, bg_color=(255, 255, 255)):
        images = [Image.open(image_path) for image_path in image_paths]
        if direction == 'horizontal':
            x_offset = 5
            y_offset = 2
            total_width = sum(image.width for image in images) + spacing * (len(images) - 1)+x_offset+1
            max_height = max(image.height for image in images)+y_offset+2
            merged_image = Image.new('RGB', (total_width, max_height), bg_color)
            for i in range(len(images)):
                image=images[i]
                merged_image.paste(image, (x_offset, y_offset))
                draw = ImageDraw.Draw(merged_image)
                font_size_mark = 18
                try:
                    mark_font = ImageFont.truetype("/root/projects/code_lihan/arial.ttf", font_size_mark)  # 修改为实际可用的字体路径和大小
                except IOError:
                    mark_font = ImageFont.load_default()
                draw.text((x_offset, 2), str(marks[i]), font=mark_font, fill=(255, 0, 0))
                area_rec=x_offset-1,1,x_offset+image.width,y_offset+image.height
                draw.rectangle(area_rec, outline="red", width=1) 
                x_offset += image.width + spacing
        # merged_image.show()
        merged_image.save(output_path)
    marks=obj_dict[obj]
    dir=info['img'][:-4] 
    obj_str=obj.replace(" ",'')
    output_path = f'{dir}_{obj_str}_merged_image.jpg'
    merge_images(subimg_path, output_path, marks,direction='horizontal', spacing=10)
    marks=obj_dict[obj]
    entities=[]
    for k,v in obj_dict.items():
        entities.append(k)
    # entity_str=(' or ').join(entities)
    question=f"Can you find {obj} in the sub-image marked by {marks}?"
    print( 'question',)
    print(f'{dir}_{obj_str}_merged_image.jpg')
    payload = judge_few_objs(f'{dir}_{obj_str}_merged_image.jpg', question,obj)
    response = query_openai(payload)
    ans_dict=json.loads(response['choices'][0]['message']['content'])
    judge_size_dict={}
    for axis,mark in zip(obj_filt_dict[obj],obj_dict[obj]):
        xmin, ymin, xmax, ymax=axis
        if get_area(xmin, ymin, xmax, ymax)<100*30:
            judge_size_dict[str(mark)]=[ans_dict[str(mark)],"small"]
        else:
            judge_size_dict[str(mark)]=[ans_dict[str(mark)],"large"]
    # answer=''
    # print('ans_dict',ans_dict)
    # for k,v in ans_dict.items():
    #     if v=='No':
    #         answer=answer+f'{obj} [{k}] is actually not a {obj}, it is marked by mistake. '
    #     # elif v!=obj and v!='Neither' and v!='both':
    #     #     answer=answer+f'obj [{k}] is a actually {obj}, it is marked by mistake. '
    #     else:
    #         answer=answer+f'{obj} [{k}] is truly a {obj}. '
    return json.dumps(judge_size_dict)


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
    # print('bbox, obj_ids',bbox, obj_ids)
    for idx, (bbox, obj_id) in enumerate(zip(bbox, obj_ids)):



        new_phrase = obj.split('(')[0]
        pattern = r'[^a-zA-Z0-9\s]'  # Matches any character that is not alphanumeric or whitespace
        new_phrase = re.sub(pattern, '', new_phrase)
        bbox=[int(num) for num in bbox]
        x_min, y_min, x_max, y_max = bbox
        area=(x_min, y_min, x_max, y_max)
        # print(new_phrase, str(obj_id),area,image)
        draw_text_mark(new_phrase, str(obj_id),area,image)
        draw_box(area,image)  
    image.save(output_path)
    # image.show()


tools= [VQA_tool,left_right_tool,judge_obj_tool,bottom_top_tool,size_tool]
tool_executor = ToolExecutor(tools)

def form_info(state):
    cont=state ['Context'][0].content
    subQ=state ['Subquestions'][0].content
    obj=state ['Entity'][0].content
    img=state ['Image'][0].content
    question_type=state ['question_type'][0].content
    label_sentence=state ['label_sentence'][0].content
    axis_dict=state ['axis_dict'][0].content
    obj_dict=state ['obj_dict'][0].content
    obj_filt_dict=state ['obj_filt_dict'][0].content
    info={}
    info['obj']=obj
    info['img']=img
    info['subQ']=subQ
    info['cont']=cont
    info['question_type']=question_type
    info['label_sentence']=label_sentence
    info['axis_dict']=axis_dict
    info['obj_dict']=obj_dict
    info['obj_filt_dict']=obj_filt_dict
    return info
def call_VQA_tool(state):
    # print("call_VQA_tool")
    info=form_info(state) 
    action = ToolInvocation(
        tool='VQA_tool',
        tool_input={'info':info},
    )
    
    response = tool_executor.invoke(action)#函数返回值
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return { "Tool_return" : [function_message] }

def call_left_right_tool(state):
    # print("call_left_right_tool")
    info={}
    info=form_info(state) 
    action = ToolInvocation(
        tool='left_right_tool',
        tool_input={'info':info},
    )
    
    response = tool_executor.invoke(action)#函数返回值
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return { "Tool_return" : [function_message] }
def call_bottom_top_tool(state):
    # print("call_left_right_tool")
    info={}
    info=form_info(state) 
    action = ToolInvocation(
        tool='bottom_top_tool',
        tool_input={'info':info},
    )
    
    response = tool_executor.invoke(action)#函数返回值
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return { "Tool_return" : [function_message] }
def call_judge_obj_tool(state):
    # print("call_left_right_tool")
    info={}
    info=form_info(state) 

    
    action = ToolInvocation(
        tool='judge_obj_tool',
        tool_input={'info':info},
    )
    
    response = tool_executor.invoke(action)#函数返回值
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return { "Tool_return" : [function_message] }
def call_size_tool(state):
    # print("call_left_right_tool")
    info={}
    info=form_info(state) 

    
    action = ToolInvocation(
        tool='size_tool',
        tool_input={'info':info},
    )
    
    response = tool_executor.invoke(action)#函数返回值
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return { "Tool_return" : [function_message] }



#create an agent executor
def create_size_prompt(question):
    size_prompt = ChatPromptTemplate.from_messages([
    ("system", '''AI assistant should help me figure out if the question is asking the size of an object. If so, you should say 'yes'; Otherwise, say 'no'. Next are a few examples.'''),
    ("human", '''The question is: "How fast is the large horse?"'''),
    ("ai", '''No'''),
    ("human", '''The question is: "Is the paper large or small?"'''),
    ("ai", '''Yes'''),
    ("human", '''The question is: "How large is the bottle on the table?"'''),
    ("ai", '''Yes'''),
    ("human", f'''The question is: "{question}"'''),

    ])
    return size_prompt
#function in langgraph that calls agent 
def create_agents(state):
    question=state['Subquestions'][0].content
    obj=state ['Entity'][0].content
    question_words=question[:-1].split(" ")
    question_words=[word.lower() for word in question_words]
    pattern = r'of the (image|picture|photo|photograph)'
    of_the_match = re.search(pattern, question)
    question_type=state['question_type'][0].content

    bracket_contents = re.findall(r'\[\s*\d+(?:,\s*\d+)*\s*\]', question)



    if question_type=="edgeQ":
        if ('left' in question_words or 'right' in question_words) and (len(bracket_contents)>1):
            tool_selected="left_right_tool"
        
        else:
            tool_selected="VQA_tool"
    if question_type=="nodeQ":

        if 'bottom' in question_words and 'top' in question_words:
            tool_selected="bottom_top_tool"
        elif 'large' in question_words or 'small' in question_words or 'big' in question_words:
            
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
            if_size_tool=(create_size_prompt(question)|llm|StrOutputParser()).invoke({} )
            if if_size_tool=='Yes':
                tool_selected='size_tool'
            else:
                tool_selected="VQA_tool"
        else: 
            if (('left' in question_words or 'right' in question_words) or ("side" in question_words or of_the_match is not None)) and question_words[0]!='what':
                tool_selected="left_right_tool"
            elif f'Is {obj}' in question and 'in the image?' in question:
                tool_selected="judge_obj_tool"
            else:
                tool_selected="VQA_tool"


    # print("tool_selected: ",tool_selected)
    return {"next":tool_selected}

