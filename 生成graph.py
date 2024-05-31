import functools
import operator
import requests
import os
from bs4 import BeautifulSoup
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
    #初步筛选可以用来detect的object
    prompt=ChatPromptTemplate.from_messages([
            SystemMessage(content='''You are a clever AI assistance for selecting entity in the question for object detection using aobject detection tool.
            I'm solving a VQA problem. I don't want to do object detection for all of the things in the image, since some of them maybe a waste for answering the question.
            Thus, I want you to select entity for object detection in the question. 
            Please help me select the entities in the question that you think I should apply object detection on them. Before selecting, you should consider following points:
            1- Some entity is abstract concepts or intangible item, so you may not want to select them for object detection and do not leave them in the dictionary of your result, such as "weather","feeling".
            2- You should analyze the whole question sentence and do not let the comma in the question distract you, which means you can analyze the parts seperated by comma seperately. 
            3- It should be helpful for me to solve a VQA problem by applying object detection on the entity.
            The question is as follows:'''),
            # MessagesPlaceholder(variable_name="context"),
            MessagesPlaceholder(variable_name="question"),
            "Just show me entities you chose in the format of ['a','b'] or just empty []"
    ])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol

    all_entities_question=[]
    for question in questions:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol

        ans=(prompt|llm).invoke({"question":[question]})
        # arr=json.loads(ans.content)
    # ans=(prompt|llm).invoke({"question":["What is the worldwide gross of the movie Dune: part two?"]})
        res="list: "+ans.content+" Question: "+question
        # print(res)
        all_entities_question.append(res)
    return all_entities_question

def relationship_dict_generation(all_entities_question):
    #dictionary
    prompt=ChatPromptTemplate.from_messages([
            SystemMessage(content='''You are a clever AI assistance and you should help me analyze a VQA problem by figuring out the relationship between the entities and organize the relationship by a dictionary. I have a list of entities and an attached VQA question, and the entities are extracted from the question. 
            Here I mainly consider two types of relationship among entities according to the question. The first one is affiliation, and the second one is parallel.
            I will use a dictionary to represent the two types of relationship.

            1-For the affiliation relationship: One entity is a part of the other one, or one entity belongs to the the other one.
            For instance, all things on a human body belongs to this person.
            2-For the parallel relationship: One entity does not belong to the other one, but there are some close relationship between the entities according to the question.
            For instance, two entities are connect by words such as "and"
            (Notice that the relationship between two entities cannot be both of them at the same time. It means if a pair of entities exists in "Parent_children", it cannot be in "parallel"; vice versa.
                        
            Here are two examples:
                        
            The list of entities and the VQA question is: list: ['woman','sweater','man','door'] Question: Is the man who is talking with the woman in red sweater next to the door?
            The output should be like:
            {"Parent_children":[{"woman":["sweater"]}],"Parallel":[{"woman":"man"},{"man":"door"}]}     

            The list of entities and the VQA question is: list: ['man','dog','ground'] Question: Does the man like the dog lying on the ground?
            The output should be like:
            {"Parent_children":[]}],"Parallel":[{"man":"dog"},{"dog":"ground"}]]}  

            Now, help me with the following list of entities and the VQA question:      '''),
            # MessagesPlaceholder(variable_name="context"),
            MessagesPlaceholder(variable_name="input"),
            
    ])
    entities_Qs=all_entities_question# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
    # ans=(prompt|llm).invoke({"input":[entities_Qs[0]]})
    # print(ans)
    all_relationship_dict_=[]
    for i in range(len(entities_Qs)):
        entities_Q=entities_Qs[i]
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
        # print(entities_Q)

        ans=(prompt|llm).invoke({"input":[entities_Q]})
        # ans=json.load(ans.content)
        res=ans.content
        # print(ans)
        # print(res)
        all_relationship_dict_.append(res)
    all_relationship_dict=[None]*len(all_relationship_dict_)
    for i in range(len(all_relationship_dict_)):
        all_relationship_dict[i]=remove_redundant_relationships(all_relationship_dict_[i])
    # for relationship_dict in all_relationship_dict:
    #     print(relationship_dict)
    return all_relationship_dict


def remove_redundant_relationships(content):
    data_dict = json.loads(content)
    # 获取Parent_children和Parallel列表
    parent_children = data_dict.get('Parent_children')
    # print(parent_children)
    parallel = data_dict.get('Parallel')

    # 创建一个集合来存储Parent_children中的所有关系
    parent_children_relations = set()
    for relation in parent_children:
        for parent, children in relation.items():
            for child in children:
                parent_children_relations.add((parent, child))
    # 过滤掉Parallel中与Parent_children重复的条目
    new_parallel = [] 
    for relation in parallel:
        for parent, child in relation.items():
            if (parent, child) not in parent_children_relations:
                new_parallel.append({parent: child})
    # 更新原始字典中的Parallel列表
    data_dict['Parallel'] = new_parallel
    # 将更新后的字典转换回JSON字符串
    return data_dict

def nodequestion_generation(all_entities_question):
    #归纳关系
    prompt=ChatPromptTemplate.from_messages([
            SystemMessage(content='''You are a clever AI assistance and you should help me analyze a VQA problem by figuring out the subquestion about the entity. I have a list of entities and an attached VQA question, and the entities are extracted from the question. 
            The subquestion should be about the characteristic of an entity. It may exist and it may not exist, which depends on the question.
            Besides the direct and clear characteristic of an entity, you should also consider one special type of subquestion: "Where is ...?". This "Where is ...?" type of question should only be asked when the question contains preposition of location, such as "left, right, front, back, up, down, next, besides, on" and so on. 
            For example, for "A on B", both "Where is A?" and "Where is B?" are needed.
            In each subquestion, it should not contain other entities. 

            Here are two examples:

            The list of entities and the VQA question is: list: ['woman','sweater','man','door'] Question: Is the man who is talking with the woman in long sweater next to the door?
            The output should be like:
            {"woman":["None, because there is no characteristic about 'woman' itself."],"sweater":["Is the sweater long?"],"man":["Where is the man?"],"door":["Where is the door?"]}

            The list of entities and the VQA question is: list: ['dog','newspaper','man','ground'] Question: Does the man reading newspaper like the black dog lying on the ground?
            The output should be like:
            {"dog":["Where is the dog?","What color is the dog?"],"newspaper":["None, because there is no characteristic about 'newspaper' itself."],"man":["None, because there is no characteristic about 'man' itself."],"ground":["Where is the ground?"]}

            Now, help me with the following list of entities and the VQA question:      '''),
            # MessagesPlaceholder(variable_name="context"),
            MessagesPlaceholder(variable_name="input"),
            
    ])
    entities_Qs=all_entities_question
    all_nodequestions=[]
    for entities_Q in entities_Qs:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
        print("entities_Q",entities_Q)

        ans=(prompt|llm).invoke({"input":[entities_Q]})
        res="Node questions: "+ans.content
        node_question=json.loads(ans.content)
        all_nodequestions.append(node_question)
        # arr=json.loads(ans.content)
    # ans=(prompt|llm).invoke({"question":["What is the worldwide gross of the movie Dune: part two?"]})
        print(res)
    return all_nodequestions
def edgequestion_generation(questions,all_relationship_dict):
    # file_path = "all_relationship_dict.txt"
    # all_relationship_dict = []
    # with open(file_path, "r") as file:
    #     for line in file:
    #         all_relationship_dict.append(json.loads(line.strip()))

#整合Parent_children， Parallel
    all_combined_relation=[]
    for i in range(len(questions)):
        relationship_dict=all_relationship_dict[i]
        parent_children=relationship_dict['Parent_children']
        combined_relation={}
        print(parent_children)
        for small_pc in parent_children:
            key,val=list(small_pc.items())[0]
            combined_relation[key]=val
        print(combined_relation)
        parallel=relationship_dict['Parallel']
        # print(parallel)
        for small_parallel in parallel:
            print('a')
            key,val=list(small_parallel.items())[0]
            if key in combined_relation:
                combined_relation[key].append(val)
            elif val in combined_relation:
                combined_relation[val].append(key)
            else:
                combined_relation[key]=[val]
        all_combined_relation.append(combined_relation)

            #归纳关系
    prompt=ChatPromptTemplate.from_messages([
        SystemMessage(content='''You are a clever AI assistance and you should help me analyze a VQA problem by figuring out the "edge_question" between two entities. I have a tuple of entities and an attached VQA question, and the entities are extracted from the question. 
        You should figure out the "edge_question" between the entities in the tuple, according to the question provided.

        Here are two examples:
        The tuple of entities and the VQA question is: 
        ('woman','sweater') Question: Is the man who is talking with the woman in red sweater next to the door?
        Output:
        <"Does the woman wear a sweater?">

        The tuple of entities and the VQA question is: 
        ('woman','man') Question: Is the man who is talking with the woman in red sweater next to the door?
        Output:
        <"Is the woman talking with a man?">

        Now, help me with the following tuple of entities and the VQA question,  '''),
        # MessagesPlaceholder(variable_name="context"),
        MessagesPlaceholder(variable_name="input"),
        
    ])
    # ans=(prompt|llm).invoke({"input":[entities_Qs[0]]})
    # print(ans)
    all_edgequestions=[]
    for i in range(len(all_relationship_dict)):
        relationship_dict=all_combined_relation[i]
        print(relationship_dict)
        if(len(relationship_dict)==0):
            all_edgequestions.append('None')
            continue
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
        # relationship_dict=json.dumps(relationship_dict)
        edge_question={}
        for k,vals in relationship_dict.items():
            for val in vals:
                # print((k,val))
                input="The tuple of entities: "+str((k,val)) +" Question: "+questions[i]
                ans=(prompt|llm).invoke({"input":[input]})
                res="Edge questions: "+ans.content
                # print(ans.content[1:-1])
                edge_question[(k,val)]=ans.content[1:-1]
        all_edgequestions.append(edge_question)

    return all_edgequestions,all_combined_relation

def graph_generation(all_nodequestion,all_edgequestions,all_combined_relation):
    for i in range(len(all_nodequestion)):
        node_questions=all_nodequestion[i]
        edge_questions=all_edgequestions[i]
        combined_relation=all_combined_relation[i]

        keys={}
        # print(edge_questions)
        for key,vals in combined_relation.items():
            keys[key]={'node_question':node_questions[key],'edge':[]}
            for val in vals:
                if (key,val) in edge_questions:
                    edge_question=edge_questions[(key,val)]
                elif (val,key) in edge_questions:
                    edge_question=edge_questions[(val,key)]
                keys[key]['edge'].append({val:{'edge_question':edge_question,'node_question':node_questions[val]}})
        # print(keys)
    return keys

if __name__=='__main__':
    questions=["Does the man in a helmet and a shirt sit at the table with the cup?","What color is the hair of the man at the table?","Are there men to the left of the person that is holding the umbrella?","Of which color is the gate?","What is in front of the green fence?","Which place is this?","Are there any horses to the left of the man?","Is the person’s hair brown and long?","What kind of fish inspired the kite design?","What is this game played with?","What is the color of the plate?","Is the surfer that looks wet wearing a wetsuit?","What kind of temperature is provided in the area where the bottles are?","['fish', 'kite'] What kind of fish inspired the kite design?","Does this man need a haircut?","Are the land dinosaurs guarded byrail in both the Display Museum of Natural History in University of Michigan and the Museo Jurassic de Asturias?","What is the sculpted bust at the Baroque library, Prague wearing on its head?","How many years after the flight of the first jet airliner was the Boeing 727 released ?","Can you identify the type of flower depicted in the foreground?","Who is wearing brighter colored clothing, the man or the woman?","What time of day does this scene likely depict, morning or evening?","Which artwork in this image appears more abstract?","Based on the luggage and attire, where might the people in the image be heading?","What historical period might this painting represent?"]
    all_entities_question=entity_generation(questions)
    all_relationship_dict=relationship_dict_generation(all_entities_question)
    # print('all_entity_question',all_entities_question)
    for relationship_dict in all_relationship_dict:
        print(relationship_dict)
    all_nodequestions=nodequestion_generation(all_entities_question)
    all_edgequestions,all_combined_relation=edgequestion_generation(questions,all_relationship_dict)
    graph=graph_generation(all_nodequestions,all_edgequestions,all_combined_relation)
    print(graph)
