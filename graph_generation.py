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
import os
os.environ["OPENAI_API_KEY"] = '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'

def entity_generation(questions):
    print("========== Entity and Question ===========")
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
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)#gpt-3.5-turbol
    def gene_entity_list(question):
        entities=(entity_generation_prompt|llm|StrOutputParser()).invoke( {
        "question": question
        })
        entity_list=eval(entities)
        # print(type(entity_list))
        # print((entity_list[0]))
        pattern = r'\b(photo|image|photograph|picture)s?\b'
        for entity in entity_list:
            if len(re.findall(pattern, entity, flags=re.IGNORECASE))>0:
                entity_list.remove(entity)
        del_words=['place','who','Who']
        for del_word in del_words:
            if del_word in entity_list:
                entity_list.remove(del_words)
        return entity_list
    for i in range(len(questions)):
        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
        question=questions[i]
        entity_list=gene_entity_list(question)
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


    relationship_tuples_generation = ChatPromptTemplate.from_messages([
    ("system", '''You are a smart AI assistant and you should help me figure out the relationship tuples based on a VQA question. /
    You will be given a list of entities, and you should analyze these entities with the VQA question. Relationship includes but not limited to: 1. spatial relationship 2. affiliation However, you should not include tuples with parallel relationship. It is allowed to find out there is no relationship tuples. Next are a few examples for you.'''),
    ("human", '''The list of entities is: ['woman','sweater','man','door']. The question is: "Is the man who is to the right of the woman in red sweater to the left or to the right of the door?". What are the relationship tuples?'''),
    ("ai", '''Thought: Since it says the 'man' is to the right of 'woman': ('man','woman'). Since it says the 'woman' is in the 'sweater': ('woman','sweater'). Since it says if the 'man' is to the left or to the right of the 'door': ('man','door'). Answer: [('man','woman'),('woman','sweater'),('man','door')]'''),
    ("human", '''The list of entities is: ['man','vehicle','chair']. The question is: "What type of vehicle is to the right of the man sitting on the chair?". What are the relationship tuples?'''),
    ("ai", '''Thought: Since it says if the 'vehicle' is to the right of the 'man': ('vehicle','man'). Since it says the 'man' is sitting on the 'chair': ('man','chair'). Answer: [('vehicle','man'),('man','chair')]'''),
    ("human", '''The list of entities is: ['apple','window']. The question is: "Do you see either any apple in the window?" What are the relationship tuples?'''),
    ("ai", '''Thought: Since it says if the 'apple' is in the 'window':('apple','window'). Answer: [('apple','window')'''),
    ("human", '''The list of entities is: ['apple','banana']. The question is: "Do you see either any apple or banana?" What are the relationship tuples?'''),
    ("ai", '''Thought: There is 'or' between 'apple' and 'banana', so the question can be divided to "Do you see any apple?" and "Do you see any banana?". 'apple' and 'banana' are seperated: No tuple. Answer: []'''),
    ("system",'''Notice there are some special cases if two entities are connected by 'and' or 'or', an example is as follows:'''),
    ("human", '''The list of entities is: {entity_list}. The question is "{question}". What are the relationship tuples??'''),
    ])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
    all_relationship_dict=[]
    extracted_lists=[]
    def gene_resd_tup(entity_list,question):
        if len(entity_list)<2:
            return defaultdict(list),[]
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
                # continue
        #     result_dict[key].append(value)
        print('extracted_tuples',extracted_tuples)
        return extracted_tuples
    for entity_list,question in zip(entity_lists,questions):
        if isinstance(question, str):
            extracted_tuples=gene_resd_tup(entity_list,question)
        else:
            sub_qs=(question[0],question[1])
            sub_lists=(entity_list[0],entity_list[1])
            extracted_tuples=[]
            for sub_q,sub_list in zip(sub_qs,sub_lists):
                # print('sub_list',sub_list)
                if len(sub_list)<2: extracted_tuples.append([])
                elif len(sub_list)==2: extracted_tuples.append([(sub_list[0],sub_list[1])])
                else:
                    extracted_tuples.append(gene_resd_tup(sub_list,sub_q))
            extracted_tuples=(extracted_tuples[0],extracted_tuples[1])
            # all_relationship_dict.append(result_dict)
        extracted_lists.append(extracted_tuples)
            # print('all_relationship_dict',all_relationship_dict)
        # print('extracted_lists',extracted_lists)
    return extracted_lists


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
    ("human", '''The list of entities is: ['building']. The VQA question is "Is the cat on the right side or on the left?". What are the position adverb question for each entity in the list?'''),
    ("ai", '''Thought: Since in the VQA question, it is asking the position of the cat on the image, include it in the question. Answer: {{"building":["On which side of the image is the cat?"]}}'''),
    ("human", '''The list of entities is: {entity_list}. The VQA question is "{question}". What are the position adverb question for each entity in the list?'''),
    ])
    posi_qs=[]
    def gene_posi_q(all_node,question,obj_dict,copy):
        question_words=question[:-1].split(' ')
        if_there='there' in question_words or 'Do you see' in question
        if_bottom='bottom' in question_words and 'top' in question_words
        answer_dict={}
        
        if if_there or if_bottom:
            if if_there:
                for node in all_node:
                    answer_dict[node]=[f'Is {node} in the image?']
            if if_bottom:
                for node in all_node:
                    if node in answer_dict:
                        answer_dict[node].append(f'Is {node} in the bottom part or top part of the image?')
                    else:
                        answer_dict[node]=[f'Is {node} in the bottom part or top part of the image?']
        else:
            if copy:
                answer_dict={all_node[0]:[question]}
            else:
    
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
                posi=(nodequestion_generation_prompt|llm|StrOutputParser()).invoke( {
                    "entity_list":all_node ,
                    "question": question,
                    })
                pattern = r"Answer:\s*(\{.*\})"
                match = re.search(pattern, posi)
                if match:
                    json_str = match.group(1)
                    answer_dict = json.loads(json_str)
                else:
                    answer_dict={}
                    for node in all_node:
                        answer_dict[node]=[]
        if not if_there:
            for node in all_node:
                if len(obj_dict[node])<3:
                    if node in answer_dict:
                        answer_dict[node][:0]=[f'Is {node} in the image?']
                    else:
                        answer_dict[node]=[f'Is {node} in the image?']
        return answer_dict
    for all_node,question,obj_dict in zip(all_nodes,questions,obj_dicts):
        # print(all_node,question)
        if isinstance(question, str):
            posi_q_dict=gene_posi_q(all_node,question,obj_dict,False)
        elif isinstance(question, tuple):
            sub_q1=question[0]
            sub_q2=question[1]
            sub_node1=all_node[0]
            sub_node2=all_node[1]
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
    ("ai", '''{{"woman":["None, because there is no characteristic about 'woman' itself."],"sweater":["Is the sweater long?"],"man":["Is the man old?","Where is the man talking?"],"door":["Where is the door?"]}}'''),
    ("human", '''The list of entities is: ['vegetable','plate']. The VQA question is "What is the vegetable on the round plate?".'''),
    ("ai", '''{{"vegetable":["What type is the vegetable?","Where is the vegetable?"],"plate":["Is the plate round?","Where is the plate?"]}}'''),
    ("human", '''The list of entities is: ['child','van']. The VQA question is "What van is to the left of the child appearing to be standing?".'''),
    ("ai", '''{{"child":["Does the child appear to be standing?","Where is the child standing?"],"van":["What type is the van?","Where is the van"]}}'''),
    ("human", '''The list of entities is: ['person','jacket']. The VQA question is "Which direction is the person wearing a red jacket looking at?".'''),
    ("ai", '''{{"person":["Which direction is the person looking at?"],"jacket":["Is the jacket red?"]}}'''),

    # ("human", '''The original dictionary is {{'person':[1,2],'shirt':[4,6]}}. The question is "Is the person [1,2] wearing a shirt [4,6]?" The answer is "Person [1] is wearing shirt [4]. Person [2] is not wearing shirt [6]." What is the update dictionary?'''),
    # ("ai", "Thought: Since the woman is talking with the man: ('woman','man'). Since the woman is in the sweater: ('woman','sweater'). Since it is asking if the man is next to the door: ('man','door'). Answer: [('woman','man'),('woman','sweater'),('man',door)]""),
    ("human", '''The list of entities is: {entity_list}. The VQA question is "{question}". What are the subquestions for each entity in the list?'''),
    ])
    nodequestions=[]
    def gene_nodequestions(all_node,question):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
        node_question=(nodequestion_generation_prompt2|llm|StrOutputParser()).invoke( {
            "entity_list":all_node,
            "question": question,
            })    
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
    
        nodequestions.append(node_question)
    all_nodequestions=[]
    for nodequestion,posi_q in zip(nodequestions,posi_qs):
        print('node_question,posi_q',node_question,posi_q)
        for key, values in posi_q.items():
            if len(values)!=0:  
                if nodequestion[key] and nodequestion[key][0].startswith('None'): 
                    nodequestion[key] = values  
                else:
                    if values not in nodequestion[key]:
                        nodequestion[key][:0]=values  
        # print('nodequestion',nodequestion)
        all_nodequestions.append(nodequestion)
    # print(all_nodequestions)
    return all_nodequestions

def edgequestion_generation(questions,extracted_list):
    print('=========generate edge============')

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
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
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
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
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
 
    main(questions)
