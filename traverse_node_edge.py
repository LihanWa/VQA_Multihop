from langgraph_process import *
import graph_generation 
import re
import ast
from tools import *

# from dynamic_graph import 
from gpt4_img import create_payload_final_ans,conclude_answer
os.environ["OPENAI_API_KEY"] = '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'
# questions = ['What is color of the bottle to the left of the computer?']
def simplify_ans(question,answer):
    simplify_ans_prompt=ChatPromptTemplate.from_messages([
    ("system", '''You are a smart AI assistant simplifying the answer to one word based on the question and the answer. Next are a few examples for you.'''),
    ("human", '''The question: "Which side of the image is the refrigerator on, the left or the right?". The answer is "The refrigerator is on the right side of the image."'''),
    ("ai", '''right'''),
    ("human", '''The question: "Is the woman to the left or to the right of the stroller which is to the right of the car?". The answer is "The woman is to the left of the stroller which is to the right of the car in the image."'''),
    ("ai", '''left'''),
    ("human", '''The question: "What item of furniture is patterned?". The answer is "Computer monitor stand."'''),
    ("ai", '''stand'''),
    ("human", '''The question: "What does the person that to the right of the people wear?". The answer is "The person to the right of the people is wearing a black and white patterned shirt."'''),
    ("ai", '''shirt'''),
    ("human", '''The question: "{question}". The answer is "{answer}"'''),
    ])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
    answer=(simplify_ans_prompt|llm|StrOutputParser()).invoke( {
    "question": question,
    "answer":answer
    })
    return answer
def answer_the_main_question(image_path,question,context,label_sentence):
    print("The question is :",question)
    print("\nThe context is: ",context)
    payload = create_payload_final_ans(image_path, question,label_sentence,context)
    # print("payload",payload)
    response = query_openai(payload)
    print(response)
    answer=response['choices'][0]['message']['content']
    return answer
def find_lf_entities(question,entity_list):
    nodeQ_remove_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assitant for figure out the 'left' and 'right' in the question are related to which entity in the entity list. Just output the entity in the entity list. Next are a few examples"),
        ("human",'''The question is "Is the person to the left of the frisbee wearing a glove?". The entity list is ['person','frisbee','glove']. '''),
        ("ai",'''['person','frisbee']'''),
        ("human",'''The question is " "What type of vehicle is to the right of the boy that is wearing a shirt?"". The entity list is ['vehicle','boy','shirt']. '''),
        ("ai",'''['vehicle','boy']'''),
        ("human", '''The question is "{question}". The entity list is {entity}. '''),
    ])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
    rel_entity=(nodeQ_remove_template|llm|StrOutputParser()).invoke( {
            "question": question,
            "entity": entity_list
        })
    clean_str = rel_entity.strip("[]").replace("'", "")
    lr_entities = [item.strip() for item in clean_str.split(',')]
    return lr_entities

# 使用正则表达式查找所有单词并进行替换
def replace_with_dict(text, dictionary):
    # 查找所有单词，对每个单词尝试进行替换
    pattern = r'\b(' + '|'.join(re.escape(key) for key in dictionary.keys()) + r')\b'
    def replace(match):
        word = match.group(0)
        # 在单词后添加其对应的列表值
        return f"{word} {dictionary[word]}"
    if len(dictionary)==0: return ''
    return re.sub(pattern, replace, text)
def update_objdict_nodeQ(obj_dict,node_question,answer):
    nodeQ_remove_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assitant for helping me remove irrelevant labels from the dictionary. No matter what the situation, you must ensure your answer adheres to this format without other words: '{{'the left objects':[marks]}}'. Next are a few examples."),
        ("human", '''The original dictionary is {{'shirt':[4,5,6]}}. The question is "Is the shirt [4,5,6] blue?". The answer is "The shirt [4] is not blue, the shirt [5] is not blue, and the shirt [6] is blue." What is the update dictionary?"'''),
        ("ai", "{{'shirt':[6]}}"),
        ("human", '''The original dictionary is {{'apple':[2,3]}}. The question is "What is the apples [1, 2] on?". The answer is "The apple [1] is on the table, the apple [2] is on the chair."'''),
        ("ai", "{{'apple':[2,3]}}"),
        ("human", '''The original dictionary is {{'woman':[1]}}. The question is "Does the woman [1] has long hair?". The answer is "The woman [1] does not have long hair."'''),
        ("ai", "{{'woman':[]}}"),
        ("human", '''The original dictionary is {ori_dictionary}. The question is {question}. The answer is {answer}. What is the update dictionary?'''),
    ])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
    new_obj_dict=(nodeQ_remove_template|llm|StrOutputParser()).invoke( {
            "ori_dictionary": obj_dict,
            "question": node_question,
            "answer": answer
        })
    pattern = r'\{[^\{]*?\}'

    match = re.search(pattern, new_obj_dict)
    if match:
        dictionary = ast.literal_eval(match.group(0))
    return dictionary
def update_objdict_edgeQ(obj_dict,edge_question,answer):

    edgeQ_remove_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assitant for helping me remove irrelevant labels from the dictionary. Irrelevant means negative sentence or the number is not mentioned. No matter what the situation, you must ensure your answer adheres to this format {{'obj':[],..}} without other words. Next are a few examples."),
    ("human", '''The original dictionary is {{'person':[2,3],'shirt':[4,5]}}. The question is "Is the person [2,3] wearing a shirt [4,5]?" The answer is "The person [2] is wearing shirt [5]. The person [3] is not wearing shirt [4,5]. What is the update dictionary?"'''),
    ("ai", "{{'person':[2],'shirt':[5]}}"),
    ("human", '''The original dictionary is {{'person':[2,3],'shirt':[4,5,6]}}. The question is "Is the person [2,3] wearing a shirt [4,5,6]?" The answer is "Person [2] is wearing shirt [5], and person [3] is wearing shirt [6]." What is the update dictionary?'''),
    ("ai", "{{'person':[2,3],'shirt':[5,6]}}"),
    ("human", '''The original dictionary is {{'bread': [1], 'table': [2]}}. The question is "Is the bread on the table?". The answer is "Yes, the bread [1] is on the table [2]."'''),
    ("ai", "{{'bread':[1],'table':[2]}}"),
    ("human", '''The original dictionary is {{'bread': [1], 'table': [2]}}. The question is "Is the bread on the table?". The answer is "No, the bread [1] is not on the table [2]."'''),
    ("ai", "{{'bread':[],'table':[]}}"),
    ("human", '''The original dictionary is {ori_dictionary}. The question is {question}. The answer is {answer}. What is the update dictionary?'''),
    ])


    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
    new_obj_dict=(edgeQ_remove_template|llm|StrOutputParser()).invoke( {
            "ori_dictionary": obj_dict,
            "question": edge_question,
            "answer": answer
        })
    return ast.literal_eval(new_obj_dict)

def dict_to_sentence(objects_dict):
        if len(objects_dict)==0: return ''
        number_words = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
        sentences = []
        for obj, markers in objects_dict.items():
            count = len(markers)
            number_word = number_words.get(count, str(count))  # 获取对应的数量词，超过 10 直接使用数字
            markers_str = ' and '.join(map(str, markers))  # 将标记数字转换为字符串并连接
            if count == 1:
                sentence = f"{number_word} {obj} marked by {markers_str}"
            else:
                sentence = f"{number_word} {obj}s marked by {markers_str}"
            sentences.append(sentence)
        if len(sentences)>1:
            full_sentence = "In the image, there are " + ', '.join(sentences[:-1]) + ', and ' + sentences[-1] + '.'
        else:
            full_sentence = "In the image, there are " +  sentences[0] + '.'
        return full_sentence

def main(questions,img_dirs,obj_dicts,axis_dicts,entity_lists,obj_not_founds,obj_filt_dicts):

    ans_graph_entity={}
    
    graphs=graph_generation.main(questions,entity_lists,obj_dicts)
    # graph_entity={'bottle': {'node_question': ["None, because there is no characteristic about 'bottle' itself."], 'edge': [{'computer': {'edge_question': '"Is the bottle to the left of the computer?"', 'node_question': ['Where is the computer?']}}]}}
    # print(graph_entity)
    for i in range(len(graphs)):
        img_dir=img_dirs[i]
        graph=graphs[i]
        obj_dict=obj_dicts[i]
        original_obj_dict=obj_dict
        axis_dict=json.dumps(axis_dicts[i])
        entity_list=entity_lists[i]
        obj_not_found=obj_not_founds[i]
        obj_filt_dict=obj_filt_dicts[i]
        if len(obj_not_found)>0:
            pass
        left_right_used=False
        bottom_up_used=False
        if graph=="No graph":
            print("No graph for this question")
            question=questions[i]

            # final_ans=answer_the_main_question(img_dir,question,f"The object dictionary is {obj_dict}, the key is object and the value is labels, you can check with it.")
            final_ans=answer_the_main_question(img_dir,question,"","")
            return final_ans
        
        all_answers=[]
        all_idx_to_remove=[]
        label_sentence=dict_to_sentence(obj_dict)
        for node, node_info in graph.items():
            # 打印节点问题
            if_tool_used=False
            print(f"Main_node: {node}")
            ans_graph_entity[node]={'node_question':[]}
            main_node_context=""
            
            for question in node_info['node_question']:
                print(f"  Main_node Question: {question}\n")
                if question[:4]=="None" or not node in obj_dict or len(obj_dict[node])==0:
                    print('obj_dict',obj_dict)
                    ans_graph_entity[node]={}
                    main_node_context=""
                else:
                    # 替换 question 中的关键词
                    print("obj_dict to update the question",obj_dict)
                    question = replace_with_dict(question, obj_dict)
                    print("updated_question ",question)
                    # print("obj_dict",type((obj_dict)))
                    # print('obj_filt_dict',obj_filt_dict)
                    # print('obj_dict',obj_dict)
                    ans,tool_used=lang_graph(img_dir,node,question,"None",'nodeQ',label_sentence,axis_dict,json.dumps(obj_dict),json.dumps(obj_filt_dict))
                    print('ans',ans)
                    if_tool_used=True
                    if tool_used!='judge_obj_tool':
                        temp_dict={}
                        temp_dict[node]=obj_dict[node]
                        obj_dict_outnode = update_objdict_nodeQ(temp_dict,question,ans)
                        print("obj_dict after update",obj_dict_outnode)
                        if (len(obj_dict_outnode)==len(temp_dict) and len(obj_dict_outnode[node])!=0):
                            all_answers.append(ans)
                            obj_dict[node]=obj_dict_outnode[node]
                            main_node_context=ans
                        else:
                            main_node_context=''
                    else:
                        idx_to_remove=[]
                        try:
                            # print('obj_dict,obj_dict',obj_dict)

                            ans_dict=json.loads(ans)
                            print('ans_dict',ans_dict)
                            for l,(k,v) in enumerate(ans_dict.items()):
                                # print('v',v[0])
                                if v[0]=='No': 
                                    idx_to_remove.append(l)
                                elif v[0]=='Yes':
                                    if v[1]=='small':
                                       all_answers.append(f"The object marked by {node} {int(k)} is truely a {node}. ") 
                            indices_to_remove = sorted(idx_to_remove, reverse=True)
                            for index in indices_to_remove:
                                # print('obj_dict[node][index]',obj_dict[node][index])
                                del obj_filt_dict[node][index]
                                all_idx_to_remove.append(obj_dict[node][index])
                                # print('all_idx_to_remove',all_idx_to_remove)
                                del obj_dict[node][index]
                        except:
                            print('*'*30)
                            print(202202)
                            del obj_filt_dict[node]
                            del obj_dict[node]
                        

                        
                            # if v=="Yes":
                                # all_answers.append(f'{node} [{k}] is truly a {node}. ')
                                # main_node_context+=f'{node} [{k}] is truly a {node}. '
                        
                        
                        print("obj_dict after update",obj_dict)
                        print("obj_filt_dict after update",obj_filt_dict)
                        img_file=img_dir.split('/')[-1] #1_234.jpg
                        img_path=os.path.dirname(img_dir) #gqa
                        img_id=img_file.split('.')[0] #1_234
                        img_name=img_id.split('_')[1]#234
                        image_ori_dir=f'/root/projects/mmcot/gqa/images/{img_name}.jpg' #234.jpg
                        input_path=image_ori_dir
                        output_path=f'{img_path}/{img_id}_new.jpg' #gqa/1_234_new.jpg
                        for ((kf,vf),(ko,vo)) in zip(obj_filt_dict.items(),obj_dict.items()):
                            if kf!= ko: sys.exit()
                            
                            print('image_ori_dir',image_ori_dir)
                            new_draw_number_save(input_path, vf,vo, kf, output_path)
                            input_path=output_path
                        print(output_path)
                        img_dir=output_path
                        print('img_dir',img_dir)
                    
                    if tool_used=='left_right_tool':
                        left_right_used=True
                    if tool_used=='bottom_top_tool': 
                        bottom_up_used=True
                    ans_graph_entity[node]['node_question'].append(ans)
                    
            # 遍历每个节点的边
            if 'edge' in node_info and len(node_info['edge'])>0:
                ans_graph_entity[node]['edge']=[]
                
                for edge_num in range(len(node_info['edge'])):
                    edge=node_info['edge'][edge_num]
                    
                    for target_node, edge_info in edge.items():
                        # 打印边的问题

                        ans_graph_entity[node]['edge'].append({target_node:{}})
                        edge_ans=''
                        main_node_edge_context=main_node_context
                        # if  (if_tool_used and tool_used!="judge_obj_tool") or not if_tool_used:
                        print(f"  Edge to {target_node}")
                        if node in obj_dict and target_node in obj_dict:
                            for edge_question in edge_info['edge_question']:
                                print(f"    Edge Question: {edge_question}\n")
                                # if edge_question!="No edge_question":
                                # print('main_node_context',main_node_context)
                                print("obj_dict to update the question",obj_dict)
                                edge_question = replace_with_dict(edge_question, obj_dict)
                                print("updated_question ",edge_question)
                                edge_ans,tool_used=lang_graph(img_dir,f'{node} {target_node}',edge_question,main_node_context,"edgeQ",label_sentence,axis_dict,json.dumps(obj_dict),json.dumps(obj_filt_dict))
                                temp_dict={}
                                temp_dict[node]=obj_dict[node]
                                temp_dict[target_node]=obj_dict[target_node]
                                obj_dict_outedge = update_objdict_edgeQ(temp_dict,edge_question,edge_ans)
                                print("obj_dict_outedge",obj_dict_outedge)
                                if (node in obj_dict_outedge and len(obj_dict_outedge[node])>0) and (target_node in obj_dict_outedge and len(obj_dict_outedge[target_node])>0):
                                    obj_dict[node]=obj_dict_outedge[node]
                                    obj_dict[target_node]=obj_dict_outedge[target_node]
                                    all_answers.append(edge_ans)
                                    main_node_edge_context=main_node_edge_context+" "+edge_ans
                                else:
                                    main_node_edge_context=main_node_context
                                if tool_used=='left_right_tool':
                                    left_right_used=True
                                if tool_used=='bottom_top_tool':
                                    bottom_up_used=True
                                print("obj_dict after update",obj_dict)
                            
                        
                        ans_graph_entity[node]['edge'][edge_num][target_node]['edge_question']=edge_ans
                        for subnode_question in edge_info['node_question']:
                            print(f"    Node Question at {target_node}: {subnode_question}\n")
                            ans_graph_entity[node]['edge'][edge_num][target_node]['node_question']=[]

                            if subnode_question[:4]!="None" and target_node in obj_dict and len(obj_dict[target_node])==0:
                                print("obj_dict to update the question",obj_dict)
                                subnode_question = replace_with_dict(subnode_question, obj_dict)
                                print("updated_question ",subnode_question)
                                ans_subnode,tool_used=lang_graph(img_dir,target_node,subnode_question,main_node_edge_context,"nodeQ",label_sentence,axis_dict,json.dumps(obj_dict),json.dumps(obj_filt_dict))
                                if tool_used=='left_right_tool':
                                    left_right_used=True
                                if tool_used=='bottom_top_tool':
                                    bottom_up_used=True
                                print('ans_subnode',ans_subnode)
                                    # 选择性删改
                                if tool_used!='judge_obj_tool':
                                    temp_dict={}
                                    temp_dict[target_node]=obj_dict[target_node]
                                    obj_dict_outnode = update_objdict_nodeQ(temp_dict,subnode_question,ans_subnode)
                                    print("obj_dict after update",obj_dict_outnode)
                                    if (len(obj_dict_outnode)==len(temp_dict) and len(obj_dict_outnode[target_node])!=0) :
                                        all_answers.append(ans_subnode)
                                        obj_dict[target_node]=obj_dict_outnode[target_node]
                                else:
                                    # print('ans_subnode',ans_subnode)
                                    idx_to_remove=[]
                                    try:
                                        # print('obj_dict,obj_dict',obj_dict)
                                        ans_dict=json.loads(ans_subnode)
                                        # print('ans_dict',ans_dict)
                                        for l,(k,v) in enumerate(ans_dict.items()):
                                            # print('v')
                                            if v[0]=='No': 
                                                idx_to_remove.append(l)
                                            elif v[0]=='Yes':
                                                
                                                if v[1]=='small':
                                                    all_answers.append(f"The object marked by {target_node} {int(k)} is truely a {target_node}. ") 
                                        # print('idx_to_remove',idx_to_remove)
                                        indices_to_remove = sorted(idx_to_remove, reverse=True)
                                        # print('indices_to_remove',indices_to_remove)
                                        # print('obj_dict',obj_dict)   
                                        for index in indices_to_remove:

                                            del obj_filt_dict[target_node][index]
                                            all_idx_to_remove.append(obj_dict[target_node][index])
                                            del obj_dict[target_node][index]
                                    except:
                                        del obj_filt_dict[target_node]
                                        del obj_dict[target_node]

                                                # print("Append: ",l)
                                            # if v=="Yes":
                                            #     all_answers.append()
                                            #     main_node_context+=f'{target_node} [{k}] is truly a {target_node}. '
                                        
                                    
                                    # print("obj_dict after update",obj_dict)
                                    # print("obj_filt_dict after update",obj_filt_dict)
                                    img_file=img_dir.split('/')[-1] #1_234_new.jpg
                                    img_path=os.path.dirname(img_dir) #gqa
                                    img_id=img_file.split('.')[0] #1_234_new
                                    img_name=img_id.split('_')[1]#234
                                    image_ori_dir=f'/root/projects/mmcot/gqa/images/{img_name}.jpg' #234.jpg
                                    input_path=image_ori_dir
                                    output_path=f'{img_path}/{img_id}_new.jpg' #gqa/1_234_new.jpg
                                    for ((kf,vf),(ko,vo)) in zip(obj_filt_dict.items(),obj_dict.items()):
                                        if kf!= ko: sys.exit()
                                        # print( vf,vo, kf)
                                        new_draw_number_save(input_path, vf,vo, kf, output_path)
                                        input_path=output_path
                                    print(output_path)
                                    img_dir=output_path
                            else:
                                print('here')
                                ans_graph_entity[node]['edge'][edge_num][target_node].pop('node_question')
        print('\n======ans_graph_entity===========')
        print(ans_graph_entity)
        #---------------------
        # print('questions',questions)
        question_words=questions[i][:-1].split(" ")
        question_words=[word.lower() for word in question_words]
        pattern = r'of the (image|picture|photo|photograph)'
        of_the_match = re.search(pattern, question)
        
        if (not bottom_up_used) and (not left_right_used) and (('left' in question_words or 'right' in question_words) or ("side" in question_words or of_the_match is not None)):
            num_pair_ori=1
            num_pari_f=1
            entity_list_ori=[]
            entity_list_f=[]
            
            if len(entity_list)>2:
                lr_entities=find_lf_entities(questions[i],entity_list)
            else:
                lr_entities=entity_list
            original_lr_dict={}
            for k,v in original_obj_dict.items():
                if len(v)==0:continue
                if k not in lr_entities: continue
                num_pair_ori=num_pair_ori*len(v)
                entity_list_ori.append(k)
                original_lr_dict[k]=v
            lr_dict={}
            for k,v in obj_dict.items():
                if len(v)==0:continue
                if k not in lr_entities: continue
                # print(v)
                num_pari_f=num_pari_f*len(v)
                entity_list_f.append(k)
                lr_dict[k]=v
            # print('num_pari_f',num_pari_f)
            if num_pari_f<=6 and num_pari_f!=0:
                print('\n=======Final_left_right_judge===========')
                info={}

                if num_pair_ori<=6 and not ("side" in question_words or of_the_match is not None):
                    info['num_pair']=num_pair_ori
                    info['obj_dict']=original_lr_dict
                    info['entity_list']=entity_list_ori
                else:
                    info['num_pair']=num_pari_f
                    info['obj_dict']=lr_dict
                    info['entity_list']=entity_list_f
                info['question']=questions[i]
                info['img']=img_dir
                info['left_right']=('left' in question_words or 'right' in question_words)
                info['side_of_the_match']="side" in question_words or of_the_match is not None
                info['axis_dict']=axis_dict
                print("info: ",info)
                ans=final_left_right_tool(info)
                all_answers.append(ans)
        print('\n=======All-answers===========')
        all_answers=' '.join(all_answers)
        print(all_idx_to_remove)
        print(all_answers)
        def modify_sentence(text,all_idx_to_remove):
            text=re.sub(r'\s+', ' ', text)
            for index in all_idx_to_remove:
                text = re.sub(r'\b' + str(index) + r'\b', '', text)
            # 删除中括号内的逗号，如果逗号后面有数字
            text = re.sub(r',+', ',', text)
            # 移除开头和结尾的逗号
            text = re.sub(r'\[(,*)', '[', text)
            text = re.sub(r'(,*)\]', ']', text)
            # 删除中括号为空的整个句子
            text = re.sub(r'\s*[^.]*\[\]\s*[^.]*\.', '', text)
            text = re.sub(r'\.\s*\.', '', text)
            return text
        all_answers=modify_sentence(all_answers,all_idx_to_remove)
        print(all_answers)
        print('\n=======The Final Answer===========')
        context=all_answers
        question=questions[i]
        # final_ans=answer_the_main_question(img_dir,question,context,label_sentence,axis_dict)
        final_ans=answer_the_main_question(img_dir,question,context,label_sentence)
        print("\nThe final answer is: ",final_ans)
        def modify_ans(question,final_ans,image_ori_dir):
            logit=final_ans
            pattern = r'\b(a|an|the|very)\b\s*'
            logit=re.sub(pattern, '', logit, flags=re.IGNORECASE)
            logit=logit.lower().replace('.','')
            logit_words=logit.split(' ')
            if len(logit_words)>2 and (',' in logit or 'and' in logit):
                img=Image.open(image_ori_dir)
                img.show()
                payload = conclude_answer(image_ori_dir, question,logit)
                logit = query_openai(payload)['choices'][0]['message']['content']
                if logit>1:
                    logit=simplify_ans(question,logit)
                return logit
            elif len(logit_words)==2: 
                logit=logit_words[1]
            elif len(logit_words)>2:
                logit=simplify_ans(question,logit)

            return logit
        img_file=img_dir.split('/')[-1] #1_234_new.jpg
        img_path=os.path.dirname(img_dir) #gqa
        img_id=img_file.split('.')[0] #1_234_new
        img_name=img_id.split('_')[1]#234
        image_ori_dir=f'/root/projects/mmcot/gqa/images/{img_name}.jpg' #234.jpg
        final_ans=modify_ans(questions[i],final_ans,image_ori_dir)
    return final_ans
if __name__=='__main__':
    # questions=["Does the man in a helmet and a shirt sit at the table with the cup?","What color is the hair of the man at the table?","Are there men to the left of the person that is holding the umbrella?","Of which color is the gate?","What is in front of the green fence?","Which place is this?","Are there any horses to the left of the man?","Is the person’s hair brown and long?","What kind of fish inspired the kite design?","What is this game played with?","What is the color of the plate?","Is the surfer that looks wet wearing a wetsuit?","What kind of temperature is provided in the area where the bottles are?","['fish', 'kite'] What kind of fish inspired the kite design?","Does this man need a haircut?","Are the land dinosaurs guarded byrail in both the Display Museum of Natural History in University of Michigan and the Museo Jurassic de Asturias?","What is the sculpted bust at the Baroque library, Prague wearing on its head?","How many years after the flight of the first jet airliner was the Boeing 727 released ?","Can you identify the type of flower depicted in the foreground?","Who is wearing brighter colored clothing, the man or the woman?","What time of day does this scene likely depict, morning or evening?","Which artwork in this image appears more abstract?","Based on the luggage and attire, where might the people in the image be heading?","What historical period might this painting represent?"]
    questions = ['What color is the shirt worn by the man standing on the left?']
    img_dir="/root/projects/code_lihan/threePeople.png"
    obj_dict={'man':[1,3],'shirt':[4,5,6]}
    main(questions,img_dir,obj_dict)