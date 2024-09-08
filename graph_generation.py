import string
import sys
import functools
import operator
import requests
import os
from bs4 import BeautifulSoup
from collections import defaultdict
import ast
import re
import json
# from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langgraph.prebuilt import ToolExecutor
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import FunctionMessage

# from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI 
from langchain_openai import ChatOpenAI
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence
from langchain.tools.render import format_tool_to_openai_function
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.llms import Tongyi

import os
os.environ["OPENAI_API_KEY"] = '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'
os.environ["DASHSCOPE_API_KEY"]= 'sk-ac7aca0206ae4da9a517628e5fa2170f'

def entity_generation(questions):
    print("========== Entity and Question ===========")
    #初步筛选可以用来detect的object
    # prompt=ChatPromptTemplate.from_messages([
    #     SystemMessage(content='''You are a smart AI assistant for selecting entity in the VQA question for object detection.

    #     The question is as follows:'''),
    #     # MessagesPlaceholder(variable_name="context"),
    #     MessagesPlaceholder(variable_name="question"),
    #      "Just show me entities you chose in the format of ['a','b'] or just empty []"
    # ])
    entity_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", '''You are a smart AI assistant for selecting entities worthy of object detection from a VQA question. Format the output in [], such as ['a','b']. Next are a few examples for you. '''),
    ("human", '''The VQA question: "Does the man sitting in the group of people look tired?". What are the entities worthy of object detection?'''),
    ("ai", '''['man','people']'''),
    ("human", '''The VQA question: "What is the animal besides the cat?". What are the entities worthy of object detection?'''),
    ("ai", '''['animal','cat']'''),
    ("human", '''The VQA question: "Is the man to the right of the hammer wearing eye glasses?". What are the entities worthy of object detection?'''),
    ("ai", '''['man','hammer','eye glasses']'''),
    ("human", '''The VQA question is "{question}". What are the entities worthy of object detection?'''),
    ])
    all_entities_question=[]
    nodes=[]
    # llm = ChatOpenAI(model_name="gpt-4o-2024-05-13", temperature=0)#gpt-3.5-turbol
    llm=Tongyi(model_name="qwen-max",temperature=0)
    def gene_entity_list(question):
        entities=(entity_generation_prompt|llm|StrOutputParser()).invoke( {
        "question": question
        })
        # print(question)
        # print(entities)
        try:
            entity_list=eval(entities)
        except:
        # if not isinstance(entity_list, list):
            return []

        # print(type(entity_list))
        # print((entity_list[0]))
        pattern = r'\b(photo|image|photograph|picture)s?\b'
        for entity in entity_list:
            if len(re.findall(pattern, entity, flags=re.IGNORECASE))>0:
                entity_list.remove(entity)
        del_words=['place','who','Who']
        for del_word in del_words:
            if del_word in entity_list:
                entity_list.remove(del_word)
        return entity_list
    for i in range(len(questions)):
        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
        question=questions[i]
        # print(question)
        entity_list=gene_entity_list(question)
        # print(entity_list)
        if len(entity_list)==0:
            pattern = r'\b(photo|image|photograph|picture)s?\b'
            question=re.sub(pattern, '', question, flags=re.IGNORECASE)
            entity_list=gene_entity_list(question)

        # arr=json.loads(ans.content)
    # ans=(prompt|llm).invoke({"question":["What is the worldwide gross of the movie Dune: part two?"]})
        # res="list: "+new_obj_dict.content+" Question: "+question
        # print(res)
        # all_entities_question.append(res)
        if i%100==0:
            print(i)
        nodes.append(entity_list)
    return nodes,all_entities_question
def relationship_dict_generation(entity_lists,questions):
    print("========== Dict Generation ===========")
    print(questions)

    #dictionary
    relationship_tuples_generation = ChatPromptTemplate.from_messages([
    ("system", '''You are a smart AI assistant and you should help me figure out the relationship tuples based on a VQA question. /
    You will be given a list of entities, and you should analyze these entities with the VQA question, and entities in relationship tuples result should comes from this list. Relationship includes but not limited to: 1. spatial relationship 2. affiliation. It is allowed to find out there is no relationship tuples. Next are a few examples for you.'''),
    ("human", '''The list of entities is: ['woman','sweater','man','door']. The question is: "Is the man who is to the right of the woman in red sweater to the left or to the right of the door?". What are the relationship tuples?'''),
    # ("ai", '''Thought: since the 'woman' is talking with the 'man', and both the 'woman' and the 'man' are in the given list E, the relationship tuple should be: ('woman','man'). Since the 'woman' is in the 'sweater', and both the 'woman' and 'sweater' are in the list E, the : ('woman','sweater'). Since it is asking if the man is next to the door: ('man','door'). Answer: [('woman','man'),('woman','sweater'),('man','door')]'''),
    ("ai", '''Thought: Since it says the 'man' is to the right of 'woman': ('man','woman'). Since it says the 'woman' is in the 'sweater': ('woman','sweater'). Since it says if the 'man' is to the left or to the right of the 'door': ('man','door'). Answer: [('man','woman'),('woman','sweater'),('man','door')]'''),
    ("human", '''The list of entities is: ['man','vehicle','chair']. The question is: "What type of vehicle is to the right of the man sitting on the chair?". What are the relationship tuples?'''),
    ("ai", '''Thought: Since it says if the 'vehicle' is to the right of the 'man': ('vehicle','man'). Since it says the 'man' is sitting on the 'chair': ('man','chair'). Answer: [('vehicle','man'),('man','chair')]'''),
    ("human", '''The list of entities is: ['apple','window']. The question is: "Do you see either any apple in the window?" What are the relationship tuples?'''),
    ("ai", '''Thought: Since it says if the 'apple' is in the 'window':('apple','window'). Answer: [('apple','window')]'''),
    ("human", '''The list of entities is: ['apple','banana']. The question is: "Do you see either any apple or banana?" What are the relationship tuples?'''),
    ("ai", '''Thought: There is 'or' between 'apple' and 'banana', so the question can be divided to "Do you see any apple?" and "Do you see any banana?". 'apple' and 'banana' are seperated: No tuple. Answer: []'''),
    # ("human", '''The list of entities is: ['apple','banana']. The question is: "Does the man like the dog lying on the ground?". What are the relationship tuples?'''),
    ("system",'''Notice there are some special cases if two entities are connected by 'and' or 'or', an example is as follows:'''),
    # ("ai", '''Thought: Since it is asking if the man like the dog, and after checking both 'man' and 'dog' are in the list E: ('man','dog'). Since the dog is lying on the ground, and after checking both 'dog' and 'ground' are in the list E: ('dog','ground'). Answer: [('man','dog'),('dog','ground')]'''),

    # ("human", '''The original dictionary is {{'person':[1,2],'shirt':[4,6]}}. The question is "Is the person [1,2] wearing a shirt [4,6]?" The answer is "Person [1] is wearing shirt [4]. Person [2] is not wearing shirt [6]." What is the update dictionary?'''),
    # ("ai", "Thought: Since the woman is talking with the man: ('woman','man'). Since the woman is in the sweater: ('woman','sweater'). Since it is asking if the man is next to the door: ('man','door'). Answer: [('woman','man'),('woman','sweater'),('man',door)]""),
    ("human", '''The list of entities is: {entity_list}. The question is "{question}". What are the relationship tuples??'''),
    ])

    llm=Tongyi(model_name="qwen-max",temperature=0)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
    # ans=(prompt|llm).invoke({"input":[entities_Qs[0]]})
    # print(ans)
    all_relationship_dict=[]
    extracted_lists=[]
    def gene_resd_tup(entity_list,question,cnt):
        if cnt==1: print('empty, try again.')
        if len(entity_list)<2:
            return []
        new_obj_dict=(relationship_tuples_generation|llm|StrOutputParser()).invoke( {
        "entity_list": entity_list,
        "question": question
        })
        print("new_obj_dict",new_obj_dict)
        matches = re.findall(r'\[.*?\]', new_obj_dict)
        for match in matches:
            print(match)
        if matches:
            extracted_content = matches[0]
        extracted_tuples = eval(extracted_content)
        # pairs = extracted_tuples
        # result_dict = defaultdict(list)
        for key, value in extracted_tuples:

            if key not in entity_list or value not in entity_list:
                extracted_tuples.remove((key,value))
        id_to_remove=[]
        for i,(key, value) in enumerate(extracted_tuples):
            if key not in entity_list or value not in entity_list:
                id_to_remove.append(i)
        if len(id_to_remove)>0:
            for index in sorted(id_to_remove, reverse=True):
                del extracted_tuples[index]
            
            if len(extracted_tuples)==0 and cnt==0:
                cnt=1
                gene_resd_tup(entity_list,question,cnt) 
        cnt=0
                # continue
        #     result_dict[key].append(value)
        print('extracted_tuples',extracted_tuples)
        return extracted_tuples
    for entity_list,question in zip(entity_lists,questions):
        if isinstance(question, str):
            extracted_tuples=gene_resd_tup(entity_list,question,0)
        else:
            sub_qs=(question[0],question[1])
            sub_lists=(entity_list[0],entity_list[1])
            extracted_tuples=[]
            for sub_q,sub_list in zip(sub_qs,sub_lists):
                # print('sub_list',sub_list)
                if len(sub_list)<2: extracted_tuples.append([])
                elif len(sub_list)==2: extracted_tuples.append([(sub_list[0],sub_list[1])])
                else:
                    extracted_tuples.append(gene_resd_tup(sub_list,sub_q,0))
            extracted_tuples=(extracted_tuples[0],extracted_tuples[1])
            # all_relationship_dict.append(result_dict)
        extracted_lists.append(extracted_tuples)
            # print('all_relationship_dict',all_relationship_dict)
        print('extracted_lists',extracted_lists)
    return extracted_lists
# def relationship_dict_generation(all_entities_question):
#     print("========== Dict Generation ===========")
#     #dictionary
#     prompt=ChatPromptTemplate.from_messages([
#             SystemMessage(content='''You are a clever AI assistant and you should help me analyze a VQA problem by figuring out the relationship between the entities and organize the relationship by a dictionary. I have a list of entities and an attached VQA question, and the entities are extracted from the question. 
#             Here I mainly consider two types of relationship among entities according to the question. The first one is affiliation, and the second one is parallel.
#             I will use a dictionary to represent the two types of relationship.

#             1-For the affiliation relationship: One entity is a part of the other one, or one entity belongs to the the other one.
#             For instance, all things on a human body belongs to this person.
#             2-For the parallel relationship: One entity does not belong to the other one, but there are some close relationship between the entities according to the question.
#             For instance, two entities are connect by words such as "and"
#             (Notice that the relationship between two entities cannot be both of them at the same time. It means if a pair of entities exists in "Parent_children", it cannot be in "parallel"; vice versa.
                        
#             Here are two examples:
                        
#             The list of entities and the VQA question is: list: ['woman','sweater','man','door'] Question: Is the man who is talking with the woman in red sweater next to the door?
#             The output should be like:
#             {"Parent_children":[{"woman":["sweater"]}],"Parallel":[{"woman":"man"},{"man":"door"}]}     

#             The list of entities and the VQA question is: list: ['man','dog','ground'] Question: Does the man like the dog lying on the ground?
#             The output should be like:
#             {"Parent_children":[]}],"Parallel":[{"man":"dog"},{"dog":"ground"}]]}  

#             Now, help me with the following list of entities and the VQA question:      '''),
#             # MessagesPlaceholder(variable_name="context"),
#             MessagesPlaceholder(variable_name="input"),
            
#     ])
#     entities_Qs=all_entities_question# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
#     # ans=(prompt|llm).invoke({"input":[entities_Qs[0]]})
#     # print(ans)
#     all_relationship_dict_=[]
#     for i in range(len(entities_Qs)):
#         entities_Q=entities_Qs[i]
#         llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
#         # print(entities_Q)

#         ans=(prompt|llm).invoke({"input":[entities_Q]})
#         # ans=json.load(ans.content)
#         res=ans.content
#         # print(ans)
#         # print(res)
#         all_relationship_dict_.append(res)
#     all_relationship_dict=[None]*len(all_relationship_dict_)
#     for i in range(len(all_relationship_dict_)):
#         all_relationship_dict[i]=remove_redundant_relationships(all_relationship_dict_[i])
#     # for relationship_dict in all_relationship_dict:
#     #     print(relationship_dict)
#     return all_relationship_dict


# def remove_redundant_relationships(content):

#     data_dict = json.loads(content)
#     # 获取Parent_children和Parallel列表
#     parent_children = data_dict.get('Parent_children')
#     # print(parent_children)
#     parallel = data_dict.get('Parallel')

#     # 创建一个集合来存储Parent_children中的所有关系
#     parent_children_relations = set()
#     for relation in parent_children:
#         for parent, children in relation.items():
#             for child in children:
#                 parent_children_relations.add((parent, child))
#     # 过滤掉Parallel中与Parent_children重复的条目
#     new_parallel = [] 
#     for relation in parallel:
#         for parent, child in relation.items():
#             if (parent, child) not in parent_children_relations:
#                 new_parallel.append({parent: child})
#     # 更新原始字典中的Parallel列表
#     data_dict['Parallel'] = new_parallel
#     # 将更新后的字典转换回JSON字符串
#     return data_dict
            # Besides the direct and clear characteristic of an entity, you should also consider one special type of subquestion: "Where is ...?". This "Where is ...?" type of question should only be asked when the question contains preposition of location, such as "left, right, front, back, up, down, next, besides, on" and so on. 
            # For example, for "A on B", both "Where is A?" and "Where is B?" are needed.
def nodequestion_generation(all_nodes,questions,obj_dicts):
    #归纳关系
    print("========== Node question Generation ===========")
    nodequestion_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", '''AI assistant should help only figure out the position information about each entity based on a list of entities and a VQA question, without considering other characteristics.
    You should analyze position adverb, such as 'left', 'right', 'under', 'above', 'front' and so on. If adverb connects two entities, omit it; if adverb connects one entity or it is asking the position on the photo (image,graph), include it in the question. Or it is asking the position of the entity on the image, include it in the question.
    Next are a few examples for AI assistant. '''),
    ("human", '''The list of entities is: ['car','child']. The VQA question is "What is the color of the car to the left of the child?". What are the position adverb question for each entity in the list?'''),
    ("ai", '''Thought: Since in the VQA question,'left' connects two entities 'car' and 'child', 'left' should not be in the characteristics question. Answer: {{"car":[], "child":[]}}'''),
    ("human", '''The list of entities is: ['sign']. The VQA question is "What is under the sign made of plastic?". What are the position adverb question for each entity in the list?'''),
    ("ai", '''Thought: Since in the VQA question, 'under' connects only one entity 'sign', 'under' can be in the characteristics question. Answer: {{"sign":["What is under the sign?"]}}'''),
    ("human", '''The list of entities is: ['building']. The VQA question is "Which side of the photo is the tall building on?". What are the position adverb question for each entity in the list?'''),
    ("ai", '''Thought: Since in the VQA question, it is asking the position of the building on the image, include it in the question. Answer: {{"building":["On which side of the image is the tall building?"]}}'''),
    ("human", '''The list of entities is: ['cat']. The VQA question is "Is the cat on the right side or on the left?". What are the position adverb question for each entity in the list?'''),
    ("ai", '''Thought: Since in the VQA question, it is asking the position of the cat on the image, include it in the question. Answer: {{"cat":["On which side of the image is the cat?"]}}'''),
    ("human", '''The list of entities is: {entity_list}. The VQA question is "{question}". What are the position adverb question for each entity in the list?'''),
    ])
    posi_qs=[]
    def gene_posi_q(all_node,question,obj_dict,copy):
        question_words=question[:-1].split(' ')
        if_there='there' in question_words or 'Do you see' in question
        if_bottom='bottom' in question_words and 'top' in question_words
        answer_dict={}
        print(if_bottom)
        # if if_there or if_bottom:
        #     if if_there:
        #         for node in all_node:
        #             answer_dict[node]=[f'Is {node} in the image?']
        #     if if_bottom:
        #         for node in all_node:
        #             if node in answer_dict:
        #                 answer_dict[node].append(f'Is {node} in the bottom part or top part of the image?')
        #             else:
        #                 answer_dict[node]=[f'Is {node} in the bottom part or top part of the image?']
        # else:
        if copy:
            answer_dict={all_node[0]:[question]}
        else:

            # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
            llm=Tongyi(model_name="qwen-max",temperature=0)
            posi=(nodequestion_generation_prompt|llm|StrOutputParser()).invoke( {
                "entity_list":all_node ,
                "question": question,
                })
            pattern = r"Answer:\s*(\{.*\})"
            match = re.search(pattern, posi)
            if match:
                json_str = match.group(1)
                # print('json_str',json_str)
                # print(question)
                answer_dict = json.loads(json_str)
            else:
                answer_dict={}
                for node in all_node:
                    answer_dict[node]=[]
        # if not if_there:
        #     for node in all_node:
        #         if len(obj_dict[node])<3:
        #             if node in answer_dict:
        #                 answer_dict[node][:0]=[f'Is {node} in the image?']
        #             else:
        #                 answer_dict[node]=[f'Is {node} in the image?']
        return answer_dict
    for all_node,question,obj_dict in zip(all_nodes,questions,obj_dicts):
        # print(all_node,question)
        if isinstance(question, str):
            posi_q_dict=gene_posi_q(all_node,question,obj_dict,False)
            nodes=all_node
        elif isinstance(question, tuple):
            sub_q1=question[0]
            sub_q2=question[1]
            sub_node1=all_node[0]
            sub_node2=all_node[1]
            nodes=sub_node1
            nodes.extend(sub_node2)
            print(nodes)
            if len(sub_node1)==1:
                sub_posi_q_dict1=gene_posi_q(sub_node1,sub_q1,obj_dict,True)
            else:
                sub_posi_q_dict1=gene_posi_q(sub_node1,sub_q1,obj_dict,False)
            if len(sub_node2)==1:
                sub_posi_q_dict2=gene_posi_q(sub_node2,sub_q2,obj_dict,True)
            else:
                sub_posi_q_dict2=gene_posi_q(sub_node2,sub_q2,obj_dict,False)
            # print('sub_posi_q_dict1',sub_posi_q_dict1)
            # print('sub_posi_q_dict2',sub_posi_q_dict2)
            for key, value in sub_posi_q_dict2.items():
                if key in sub_posi_q_dict1:
                    sub_posi_q_dict1[key].extend(value)
                else:
                    sub_posi_q_dict1[key] = value
            posi_q_dict=sub_posi_q_dict1
        else: 
            sys.exit()
        del_ele=[]
        for k in posi_q_dict:
            if not k in nodes: 
                del_ele.append(k)
        for k in del_ele: del posi_q_dict[k]
        posi_qs.append(posi_q_dict)
        #  In each subquestion, it should not contain other entities.

    def check_and_replace_questions(entity_dict, entities):

            for entity, questions in entity_dict.items():
                indices_to_delete = []
                for i, question in enumerate(questions):
                    question_words=question[:-1].split(' ')
                    if any(other_entity in question_words for other_entity in entities if other_entity != entity) or any(other_entity in question for other_entity in entities if (other_entity != entity and ' ' in other_entity)) or 'there' in question_words or'something' in question_words or 'someone' in question_words or 'Where' in question_words:
                        if len(questions) > len(indices_to_delete)+1:
                            indices_to_delete.append(i)
                        else:
                            entity_dict[entity][i] = f"None, because there is no characteristic about '{entity}' itself."
                for index in sorted(indices_to_delete, reverse=True):
                    del entity_dict[entity][index]
            return entity_dict
    nodequestion_generation_prompt2 = ChatPromptTemplate.from_messages([
    ("system", '''You are a clever AI assistant and you should help me figure out the subquestion of the entity according to the VQA question. 
        The subquestion may exist and it may not exist, which depends on the VQA question.
        Besides the direct and clear characteristic of an entity, you should also consider two special type of subquestion: 
        1- "Where is ...?". This "Where is ...?" type of question should only be asked when the question contains preposition of location, such as "left, right, front, back, up, down, next, besides, on" and so on. 
        For example, for "A on B", both "Where is A?" and "Where is B?" are needed. /
        2- "What type is ...?" This question should be asked only when the VQA question is asking the type of an entity, especially when the entity is a general category, such as vegetable and vehicle. It may ask like "What is ...", "What type is...", "What kind of ..." and "Which type is ...". '''),
    ("human", '''The list of entities is: ['woman','sweater','man','door'] The VQA question is "Is the old man who is talking with the woman in long sweater next to the door?".'''),
    # ("ai", '''Thought: In the list ['woman','cat','box']. For 'woman' , it says the 'woman' is old, so the subquestion for 'woman' should be: "Is the woman old?" For 'cat', it says the 'cat' is to the left of the 'woman', so this information is not considered. For 'cat', it says the 'cat' is in the 'box', however it considers another entity 'box', so this information is not considered. For 'cat', it says what color is the 'cat', so the subquestion for 'cat' should be: "What color is the cat?". For 'box', since there are no characteristics for the 'box' itself, the subquestion for the 'box' should be: "None, because there is no characteristic about 'box' itself.". Answer: {{"woman":["Is the woman old?"],"cat":["What color is the cat?"],'box':["None, because there is no characteristic about 'box' itself."]}}'''),
    ("ai", '''{{"woman":["None, because there is no characteristic about 'woman' itself."],"sweater":["Is the sweater long?"],"man":["Is the man old?","Where is the man talking?"],"door":["Where is the door?"]}}'''),
    ("human", '''The list of entities is: ['vegetable','plate']. The VQA question is "What is the vegetable on the round plate?".'''),
    # ("ai", '''Thought:In the list ['man','dog','ground']. For 'man', there are no characteristics for the 'man' itself, so the subquestion for the 'man' should be: "None, because there is no characteristic about 'man' itself.". For 'dog', it says the 'dog' is black, so the subquestion for 'dog' should be: "Is the dog black?". For 'dog', it says the 'dog' is lying on the ground', however it considers another entity 'ground', so this information is not considered. For 'ground', there are no characteristics about the 'ground' itself, so the subquestion for the 'ground' should be: "None, because there is no characteristic about 'ground' itself.". Answer: {{"man":["None, because there is no characteristic about 'man' itself."],"dog":["Is the dog black?"],"ground":["None, because there is no characteristic about 'ground' itself."]}}'''),
    ("ai", '''{{"vegetable":["What type is the vegetable?","Where is the vegetable?"],"plate":["Is the plate round?","Where is the plate?"]}}'''),
    ("human", '''The list of entities is: ['child','van']. The VQA question is "What van is to the left of the child appearing to be standing?".'''),
    ("ai", '''{{"child":["Does the child appear to be standing?","Where is the child standing?"],"van":["What type is the van?","Where is the van"]}}'''),
    ("human", '''The list of entities is: ['person','jacket']. The VQA question is "Which direction is the person wearing a red jacket looking at?".'''),
    ("ai", '''{{"person":["Which direction is the person looking at?"],"jacket":["Is the jacket red?"]}}'''),
    # ("ai", '''Thought: In the list ['child','van']. For 'child', it says the child appears to be standing, so the subquestion for 'child' should be: "Does the child appear to be standing?". For 'van', it says the 'van' is to the left of 'child', however it considers another entity 'child', so this information is not considered. For 'van', it says what is the 'van', the subquestion for the 'van' should be: "What is the van?". Answer: {{"child":["Is child appear to be standing?"],"van":["What is the van?"]}}'''),

    # ("human", '''The original dictionary is {{'person':[1,2],'shirt':[4,6]}}. The question is "Is the person [1,2] wearing a shirt [4,6]?" The answer is "Person [1] is wearing shirt [4]. Person [2] is not wearing shirt [6]." What is the update dictionary?'''),
    # ("ai", "Thought: Since the woman is talking with the man: ('woman','man'). Since the woman is in the sweater: ('woman','sweater'). Since it is asking if the man is next to the door: ('man','door'). Answer: [('woman','man'),('woman','sweater'),('man',door)]""),
    ("human", '''The list of entities is: {entity_list}. The VQA question is "{question}".'''),
    ])
    nodequestions=[]
    def gene_nodequestions(all_node,question):
        print('all_node',all_node)
        if len(all_node)==0:
            return {}
        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
        llm=Tongyi(model_name="qwen-max",temperature=0)
        node_question=(nodequestion_generation_prompt2|llm|StrOutputParser()).invoke( {
            "entity_list":all_node,
            "question": question,
            })    
        print('node_question',node_question)
        node_question=json.loads(node_question)
        
        node_question=check_and_replace_questions(node_question,all_node)
        return node_question
    #分成了position question 和node question
    for all_node,question,posi_q in zip(all_nodes,questions,posi_qs):
        if isinstance(question, str):
            node_question=gene_nodequestions(all_node,question)
        elif isinstance(question, tuple):
            sub_q1=question[0]
            sub_q2=question[1]
            sub_node1=all_node[0]
            sub_node2=all_node[1]
            print(sub_node1)
            print(sub_node2)
            if len(sub_node1)==1 and not sub_q1 in posi_q[sub_node1[0]]:
                sub_node_q_dict1=gene_nodequestions(sub_node1,sub_q1)
            else:
                sub_node_q_dict1={}
                for node in sub_node1:
                    sub_node_q_dict1[node]=[f"None, because there is no characteristic about {node} itself."]
            if len(sub_node2)==1 and not sub_q2 in posi_q[sub_node2[0]]: 
                # print('sub_q2',sub_q2)
                # print('posi_q[sub_node2[0]][0]',posi_q[sub_node2[0]][0])
                sub_node_q_dict2=gene_nodequestions(sub_node2,sub_q2)
            else:
                sub_node_q_dict2={}
                for node in sub_node2:
                    sub_node_q_dict2[node]=[f"None, because there is no characteristic about {node} itself."]
            # print('sub_node_q_dict1',sub_node_q_dict1)
            # print('sub_node_q_dict2',sub_node_q_dict2)
            for key, value in sub_node_q_dict2.items():
                if key in sub_node_q_dict1:
                    if sub_node_q_dict1[key][0].startswith('None'):
                        sub_node_q_dict1[key]=value
                    else:
                        if not sub_node_q_dict2[key][0].startswith('None'):
                            sub_node_q_dict1[key]=value
                else:
                    sub_node_q_dict1[key] = value
            node_question=sub_node_q_dict1
        for k in all_node:
            if isinstance(k,str):
                if not k in node_question: node_question[k]=[f"None, because there is no characteristic about {k} itself."]
            else:
                for k_ in k:
                    if not k_ in node_question: node_question[k_]=[f"None, because there is no characteristic about {k_} itself."]

        nodequestions.append(node_question)
    all_nodequestions=[]
    
    for nodequestion,posi_q in zip(nodequestions,posi_qs):
        print('node_question,posi_q',node_question,posi_q)
        for key, values in posi_q.items():
            if len(values)!=0:  
                if key in nodequestion and nodequestion[key][0].startswith('None'): 
                    nodequestion[key] = values  
                else:
                    if values not in nodequestion[key]:
                        nodequestion[key][:0]=values  
        # print('nodequestion',nodequestion)
        all_nodequestions.append(nodequestion)
    # print(all_nodequestions)
    return all_nodequestions
# def simplify_question(all_nodequestions,questions):

#     simplify_question_prompt = ChatPromptTemplate.from_messages([
#     ("system", '''AI assistant should help simplify the original question based on already exist sub-question dictionary. Since some sub-questions are already ask some information, the original question can be simplified without asking the already asked information, and it can even becomes a empty string. Next are a few examples for AI assistant. '''),
#     ("human", '''The original question is "Is the old man who is talking with the woman in a long sweater next to the black door?". The sub-question dictionary is {{"sweater":["Is the sweater long?"],"man":["Is the man old?"],"door":["What color is the door?"]}}'''),
#     ("ai", '''Is the man who is talking with the woman in a sweater next to the door?'''),
    # ("human", '''The original question is "What is the vegetable on the round plate?". The sub-question dictionary is {{"vegetable":["What type is the vegetable?"],"plate":["Is the plate round?"]}}'''),
    # # ("ai", '''Thought:In the list ['man','dog','ground']. For 'man', there are no characteristics for the 'man' itself, so the subquestion for the 'man' should be: "None, because there is no characteristic about 'man' itself.". For 'dog', it says the 'dog' is black, so the subquestion for 'dog' should be: "Is the dog black?". For 'dog', it says the 'dog' is lying on the ground', however it considers another entity 'ground', so this information is not considered. For 'ground', there are no characteristics about the 'ground' itself, so the subquestion for the 'ground' should be: "None, because there is no characteristic about 'ground' itself.". Answer: {{"man":["None, because there is no characteristic about 'man' itself."],"dog":["Is the dog black?"],"ground":["None, because there is no characteristic about 'ground' itself."]}}'''),
    # ("ai", '''Is the vegetable on the plate?'''),
    # ("human", '''The original question is "What van is to the left of the child appearing to be standing?". The sub-question dictionary is {{"child":["Does the child appear to be standing?"],"van":["What type is the van?"]}}'''),
    # ("ai", '''Is the van to the left of the child'''),
    # ("human", '''The original question is "Which direction is the person looking at?". The sub-question dictionary is {{"person":["Which direction is the person looking at?"]}}'''),
    # ("ai", ''''''),
    # ("human", '''The original question is "Is there an apple or a pear?". The sub-question dictionary is {{"apple":["Is there an apple?"],"pear":["Is there a pear?"]}}'''),
    # ("ai", ''''''),
    #  ("human", '''The original question is "{question}". The sub-question dictionary is {all_nodequestion}. '''),
    # ])
    # # posi_qs=[]
    # for all_nodequestion,question in zip(all_nodequestions,questions):
    #     print(all_nodequestion,question)
    #     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
    #     simplified_question=(simplify_question_prompt|llm|StrOutputParser()).invoke( {
    #         "question": question,
    #         "all_nodequestion":all_nodequestion ,
    #         })
    #     print('Original Qustion: ',question)
    #     print('simplified_question: ',simplified_question)
def edgequestion_generation(questions,extracted_list):
    print('=========generate edge============')
    # file_path = "all_relationship_dict.txt"
    # all_relationship_dict = []
    # with open(file_path, "r") as file:
    #     for line in file:
    #         all_relationship_dict.append(json.loads(line.strip()))

#整合Parent_children， Parallel
    # all_combined_relation=[]
    # for i in range(len(questions)):
    #     relationship_dict=all_relationship_dict[i]
    #     parent_children=relationship_dict['Parent_children']
    #     combined_relation={}
    #     # print(parent_children)
    #     for small_pc in parent_children:
    #         key,val=list(small_pc.items())[0]
    #         combined_relation[key]=val
    #     # print(combined_relation)
    #     parallel=relationship_dict['Parallel']
    #     # print(parallel)
    #     for small_parallel in parallel:
    #         key,val=list(small_parallel.items())[0]
    #         if key in combined_relation:
    #             combined_relation[key].append(val)
    #         elif val in combined_relation:
    #             combined_relation[val].append(key)
    #         else:
    #             combined_relation[key]=[val]
    #     all_combined_relation.append(combined_relation)

            #归纳关系
    # prompt=ChatPromptTemplate.from_messages([
    #     SystemMessage(content='''You are a clever AI assistant and you should help me analyze a VQA problem by figuring out the "edge_question" between two entities. I have a tuple of entities and an attached VQA question, and the entities are extracted from the question. 
    #     You should figure out the "edge_question" between the entities in the tuple, according to the question provided.

    #     Here are two examples:
    #     The tuple of entities and the VQA question is: 
    #     ('woman','sweater') Question: Is the man who is talking with the woman in red sweater next to the door?
    #     Output:
    #     <"Does the woman wear a sweater?">

    #     The tuple of entities and the VQA question is: 
    #     ('woman','man') Question: Is the man who is talking with the woman in red sweater next to the door?
    #     Output:
    #     <"Is the woman talking with a man?">

    #     Now, help me with the following tuple of entities and the VQA question,  '''),
    #     # MessagesPlaceholder(variable_name="context"),
    #     MessagesPlaceholder(variable_name="input"),
        
    # ])
    # # ans=(prompt|llm).invoke({"input":[entities_Qs[0]]})
    # # print(ans)
    # all_edgequestions=[]
    # for i in range(len(all_relationship_dict)):
    #     relationship_dict=all_relationship_dict[i]
    #     # print(relationship_dict)
    #     if(len(relationship_dict)==0):
    #         all_edgequestions.append('None')
    #         continue
    #     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
    #     # relationship_dict=json.dumps(relationship_dict)
    #     edge_question={}
    #     for k,vals in relationship_dict.items():
    #         for val in vals:
    #             # print((k,val))
    #             input="The tuple of entities: "+str((k,val)) +" Question: "+questions[i]
    #             ans=(prompt|llm).invoke({"input":[input]})
    #             res="Edge questions: "+ans.content
    #             # print(ans.content[1:-1])
    #             edge_question[(k,val)]=ans.content[1:-1]
    #     all_edgequestions.append(edge_question)
    relationship_checking_questions_prompt = ChatPromptTemplate.from_messages([
    ("system", '''You are a clever AI assistant and you should help me figuring out the relationship checking questions between two entities. I have a list of relationship tuples and an attached VQA question. Each relationship tuple has two entities, which are extracted from the question. 
            In a relationship tuple, I want to generate a relationship checking question according to the VQA question. You as a AI assistant must ensure your format as: "Thought: ... Answer: ...".Next are a few examples for you.'''),
    ("human", '''The list of relationship tuples is: [('woman','man'),('woman','sweater'),('man','ball')]. The VQA question is "Is the man who is talking with the woman in a red sweater to the left or to the right of the ball?". What are the relationship checking questions?'''),
    ("ai", '''Thought: I can break the VQA question into a few small questions to help me figure out the relationship checking questions: "Is the man talking with the woman?", "Is the woman in a red sweater?", "Is the man to the left or to the right of the ball?". For ('woman','man'), the relationship checking question should be: "Is the woman talking with the man?". For ('woman','sweater'), the relationship checking question should be: "Is the woman in a sweater?". For ('man','ball'), the relationship checking question should be: "Is the man to the left or to the right of the ball?". Answer: ('woman','man'):"Is the woman talking with the man?",('woman','sweater'):"Is the woman in a sweater?",('man','door'):"Is the man to the left or to the right of the ball?"'''),
    ("human", '''The list of relationship tuples is: [('child','van')]. The VQA question is "What is the van to the right of the child". What are the relationship checking questions?'''),
    ("ai", '''Thought: I can break the VQA question into a few small questions to help me figure out the relationship checking questions: "What is the van?", "Is the van to the right of the child?". For ('child','van'), the relationship checking question should be: "Is the van to the right of the child?" Answer: ('child','van'):"Is the van to the right of the child?"'''),
    # ("human", '''The list of relationship tuples is: [('man','dog'),('dog','ground')]. The VQA question is "Does the man like the dog lying on the ground?". What are the relationship checking questions?'''),
    # ("ai", '''Thought: Firstly, I can break the VQA question into a few small questions to help me understand: "Does the man like the dog?", "Is the dog lying on the ground?". Secondly, For ('man','dog'), since it is says the man like the dog, the relationship checking question should be: Does the man like the dog? For ('dog','ground'), since it says the dog is lying on the ground, the relationship checking question should be: Is the dog lying on the ground? Answer: ('man','dog'):"Does the man like the dog?",('dog','ground'):"Is the dog lying on the ground?"'''),
    # ("human", '''The original dictionary is {{'person':[1,2],'shirt':[4,6]}}. The question is "Is the person [1,2] wearing a shirt [4,6]?" The answer is "Person [1] is wearing shirt [4]. Person [2] is not wearing shirt [6]." What is the update dictionary?'''),
    # ("ai", "Thought: Since the woman is talking with the man: ('woman','man'). Since the woman is in the sweater: ('woman','sweater'). Since it is asking if the man is next to the door: ('man','door'). Answer: [('woman','man'),('woman','sweater'),('man',door)]""),
    ("human", '''The list of relationship tuples is: {relationship_tuples}. The question is "{question}". What are the relationship checking questions?'''),
    ])
    def remove_entry(dd, key, value):
        if key in dd:
            try:
                dd[key].remove(value)
                if not dd[key]:
                    del dd[key]
            except ValueError:
                pass
        return dd

    all_edgequestions=[]
    def gene_edge_rela( extracted_tuples,question):
        
        # print(extracted_list)
        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
        llm=Tongyi(model_name="qwen-max",temperature=0)
        relationship_checking_questions=(relationship_checking_questions_prompt|llm|StrOutputParser()).invoke( {
            "relationship_tuples": extracted_tuples,
            "question": question,
            })
        # print(relationship_checking_questions)
        answer_match = re.search(r'Answer:\s*(.*)', relationship_checking_questions)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            pairs = re.findall(r"\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)\s*:\s*\"([^\"]+)\"", answer_content)
            answer_dict = { (pair[0], pair[1]): [pair[2]] for pair in pairs }
            # print('answer_dict,',answer_dict)
            #del not both in
            keys_to_remove = []
            for key, value in answer_dict.items():
                entity1, entity2 = key
                if entity1 not in value[0] or entity2 not in value[0]:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del answer_dict[key]
            #if some dispear
            # print('answer_dict',answer_dict)
            return answer_dict
        else:
            print("未找到 'Answer' 的内容")
            return 'No edge_question'
    all_relationship_dict=[]
    def gene_relationship(dict1):
        relationship = {}

        for key1, key2 in dict1.keys():
            if key1 in relationship:
                relationship[key1].append(key2)
            else:
                if key2 in relationship or key2 in [k for t in dict1.keys() if t!=(key1,key2) for k in t ]:
                    key1, key2 = key2, key1
                relationship[key1] = [key2]
        return relationship
    for i, (extracted_tuples,question) in enumerate(zip(extracted_list,questions)):


        if isinstance(question, str):
            if len(extracted_tuples)==0:
                print("No edge_question for this question.")
                all_edgequestions.append(['No edge_question'])
                all_relationship_dict.append({})
                continue
            answer_dict=gene_edge_rela(extracted_tuples,question)
            all_edgequestions.append(answer_dict)
            if answer_dict=='No edge_question':
                all_relationship_dict.append({})
            else:
                all_relationship_dict.append(gene_relationship(answer_dict))
        elif isinstance(question, tuple):
            sub_qs=(question[0],question[1])
            print(sub_qs)
            sub_tuples=(extracted_tuples[0],extracted_tuples[1])
            sub_edgeQ=[]
            for idx, (sub_q,sub_tuple) in enumerate(zip(sub_qs,sub_tuples)):
                if len(sub_tuple)==0:
                    sub_edgeQ.append({})
                elif len(sub_tuple)==1:
                    sub_edgeQ.append({sub_tuple[0]:[sub_q]})
                else:
                    sub_answer_dict=gene_edge_rela(sub_tuple,sub_q)
                    print('sub_answer_dict',sub_answer_dict)
                    sub_edgeQ.append(sub_answer_dict)
            sub_edgeQ1,sub_edgeQ2=sub_edgeQ
            # print(sub_edgeQ1,sub_edgeQ2)
            for key, value in sub_edgeQ2.items():
                key2=(key[1],key[0])
                if key in sub_edgeQ1 :
                    sub_edgeQ1[key].extend(value)  
                elif key2 in sub_edgeQ1:
                    sub_edgeQ1[key2].extend(value)  
                else:
                    sub_edgeQ1[key] = value 
            all_edgequestions.append(sub_edgeQ1)
            all_relationship_dict.append(gene_relationship(sub_edgeQ1))
    print("all_edgequestions edge",all_edgequestions)
    print('all_relationship_dict: ',all_relationship_dict)
    return all_edgequestions,all_relationship_dict

def graph_generation(all_nodes,all_nodequestion,all_edgequestions,all_combined_relation):
    graphs=[]
    for i in range(len(all_nodequestion)):
        node_questions=all_nodequestion[i]
        edge_questions=all_edgequestions[i]
        all_node=all_nodes[i]
        if len(all_node)==0:
            print("No graph generation for this question.")
            graphs.append("No graph")
            continue
        # print("all_edgequestions gra",all_edgequestions)
        # print(edge_questions)
        combined_relation=all_combined_relation[i]
        print("combined_relation",combined_relation)
        keys={}
        nodes=[]
        # print(edge_questions)
        for key,vals in combined_relation.items():
            keys[key]={'node_question':node_questions[key],'edge':[]} if key not in nodes else {'node_question':['None, because it has been asked before.'],'edge':[]}
            nodes.append(key)
            for val in vals:
                if (key,val) in edge_questions:
                    edge_question=edge_questions[(key,val)]
                elif (val,key) in edge_questions:
                    edge_question=edge_questions[(val,key)]
                if val not in nodes:
                    keys[key]['edge'].append({val:{'edge_question':edge_question,'node_question':node_questions[val]}})  
                    nodes.append(val)
                else: keys[key]['edge'].append({val:{'edge_question':edge_question,'node_question':['None, because it has been asked before.']}})  
        # print(keys)

        def is_key_or_value(dd, item):

            if item in dd or any(item in sublist for sublist in dd.values()):
                return True
            else: return False
        for all_n in all_node:
            if not is_key_or_value(combined_relation,all_n):
                keys[all_n]={'node_question':node_questions[all_n]}
        graphs.append(keys)
    return graphs
def that_question_prompt(question):
    that_prompt = ChatPromptTemplate.from_messages([
    ("system", '''AI assistant should help me divide the sentence into two parts if needed. Specifically, if there is a verb after 'that', it means the part after that can form a question, and you should find the subject before 'that'. Also, form a question by the part before 'that'. Next are a few examples for you to follow the format.'''),
    ("human", '''The question is: "What is the walking man that is to the right of the woman carrying?"'''),
    ("ai", '''[What is the walking man?,Is the man to the right of the woman carrying?]'''),
    ("human", '''The question is: "Do you think that cap is black?"'''),
    ("ai", '''[Do you think that cap is black?]'''),
    ("human", '''The question is: "Are there zebras in the photo that are not staring?"'''),
    ("ai", '''[Are zebras staring?,Are there zebras in the photo?]'''),
    ("human", '''The question is: "What is that woman in?"'''),
    ("ai", '''[What is that woman in?]'''),
    ("human", '''The question is: "What type of fast food is on the grill that looks gray and black?"'''),
    ("ai", '''[What type of fast food is on the grill?,Does the grill look gray and black?]'''),
    ("human", '''The question is: "Is there a cat that is not white?"'''),
    ("ai", '''[Is there a cat?,Is the cat not white?]'''),
    ("human", f'''The question is: "{question}"'''),

    ])
    return that_prompt
def check_that_question(questions,all_nodes):
    check_that_questions=[]
    check_that_nodes=[]
    for question,entity_list in zip(questions,all_nodes):
        if not 'that' in question: 
            check_that_questions.append(question)
            check_that_nodes.append(entity_list)
            continue
        llm=Tongyi(model_name="qwen-max",temperature=0)
        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
        div_questions=(that_question_prompt(question)|llm|StrOutputParser()).invoke( {
        })
        div_questions=div_questions[1:-1].split(',')
        if len(div_questions)!=2 :
            check_that_questions.append(question)
            check_that_nodes.append(entity_list)
            continue
        cleaned_sentence1 = div_questions[0].lower().translate(str.maketrans('', '', string.punctuation))
        found_phrases1= [phrase for phrase in entity_list if phrase.lower() in cleaned_sentence1]
        cleaned_sentence2 = div_questions[1].lower().translate(str.maketrans('', '', string.punctuation))
        found_phrases2 = [phrase for phrase in entity_list if phrase.lower() in cleaned_sentence2]
        if len(found_phrases1)==0 or len(found_phrases2)==0:
            check_that_questions.append(question)
            check_that_nodes.append(entity_list)
            continue
        check_that_questions.append((div_questions[0],div_questions[1]))
        check_that_nodes.append((found_phrases1,found_phrases2))
        # if len(found_phrases)>2:
    return check_that_questions,check_that_nodes
def merge(all_nodes):
    for i in range(len(all_nodes)):
        if isinstance(all_nodes[i],tuple):
            nodes1=all_nodes[i][0]
            nodes2=all_nodes[i][1]
            for node in nodes1:
                if not node in nodes2:
                    nodes2.append(node)
            all_nodes[i]=nodes2
    return all_nodes
def main(questions,all_nodes,obj_dicts):
    # all_nodes,all_entities_question=entity_generation(questions)
    print("all_nodes",all_nodes)
    questions,all_nodes=check_that_question(questions,all_nodes)
    # print(all_nodes)

    #{'':''}   [('','')]
    all_nodequestions=nodequestion_generation(all_nodes,questions,obj_dicts)
    extracted_lists=relationship_dict_generation(all_nodes,questions)
    # print(all_relationship_dict)
    # # print('all_entity_question',all_entities_question)
    # # for relationship_dict in all_relationship_dict:
    # #     # import pdb; pdb.set_trace()
    # #     print(relationship_dict)
    # # print(all_nodequestions)
    all_edgequestions,all_relationship_dict=edgequestion_generation(questions,extracted_lists)
    # print("all_edgequestions",all_edgequestions)
    all_nodes=merge(all_nodes)
    # # import pdb; pdb.set_trace()
    graphs=graph_generation(all_nodes,all_nodequestions,all_edgequestions,all_relationship_dict)
    print("==========Generated graph==========")
    print(graphs)
    # print("====================\n")
    return graphs
if __name__=='__main__':
    # questions=["Does the man in a helmet and a shirt sit at the table with the cup?","What color is the hair of the man at the table?","Are there men to the left of the person that is holding the umbrella?","Of which color is the gate?","What is in front of the green fence?","Which place is this?","Are there any horses to the left of the man?","Is the person’s hair brown and long?","What kind of fish inspired the kite design?","What is this game played with?","What is the color of the plate?","Is the surfer that looks wet wearing a wetsuit?","What kind of temperature is provided in the area where the bottles are?","['fish', 'kite'] What kind of fish inspired the kite design?","Does this man need a haircut?","Are the land dinosaurs guarded byrail in both the Display Museum of Natural History in University of Michigan and the Museo Jurassic de Asturias?","What is the sculpted bust at the Baroque library, Prague wearing on its head?","How many years after the flight of the first jet airliner was the Boeing 727 released ?","Can you identify the type of flower depicted in the foreground?","Who is wearing brighter colored clothing, the man or the woman?","What time of day does this scene likely depict, morning or evening?","Which artwork in this image appears more abstract?","Based on the luggage and attire, where might the people in the image be heading?","What historical period might this painting represent?"]
    questions = [
        'The woman to the left of the man is wearing what?',
 'Are there any books near the device on the right?',
 'The man that is to the left of the horse is walking where?',
 'Do you see either any purple bags or umbrellas?',
 'Is the man that is to the left of the plate holding a frisbee?',
 'What is the vegetable on top of the pizza?',
 'Is there any short fence or surfboard?',
 'Is the purse pink and table white?',
 'Are there drawers to the left of the bed that is in front of the curtains?',
 'Which kind of animal is the child watching?'] # all_nodes,all_entities_question=entity_generation(questions)
    # print(all_nodes)
    # print(all_entities_question)
    # all_relationship_dict,extracted_lists=relationship_dict_generation(all_nodes,questions)
    # print(all_relationship_dict)

    # # print('all_entity_question',all_entities_question)
    # # for relationship_dict in all_relationship_dict:
    # #     # import pdb; pdb.set_trace()
    # #     print(relationship_dict)
    # all_nodequestions=nodequestion_generation(all_nodes,questions)
    # print(all_nodequestions)
    # all_edgequestions=edgequestion_generation(questions,extracted_lists)
    # print(all_edgequestions)
    # # print(all_combined_relation)
    # # import pdb; pdb.set_trace()
    # graphs=graph_generation(all_nodequestions,all_edgequestions,all_relationship_dict)
    # print(graphs)
    main(questions)