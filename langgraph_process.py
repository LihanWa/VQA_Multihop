#lang graph
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence
from tools import *
# tools= ["VQA_tool"]

class AgentState(TypedDict):
    Image: str
    Entity: str
    Subquestions: str
    Context: str
    next: str
    Tool_return: str
    question_type: str
    label_sentence:str
    axis_dict:str
    obj_dict:str
    obj_filt_dict:str
workflow = StateGraph(AgentState)
workflow.add_node("agent", create_agents)
workflow.add_node("VQA_tool", call_VQA_tool)
workflow.add_node("left_right_tool", call_left_right_tool)
workflow.add_node("judge_obj_tool", call_judge_obj_tool)
workflow.add_node("bottom_top_tool", call_bottom_top_tool)
workflow.add_node("size_tool", call_size_tool)
workflow.add_node("identity_tool", call_identity_tool)
workflow.set_entry_point("agent")
tools=["VQA_tool","left_right_tool","judge_obj_tool","bottom_top_tool","size_tool","identity_tool"]
conditional_map = {k: k for k in tools}
# conditional_map['FINISH'] = END
# print(conditional_map)
workflow.add_conditional_edges(
    "agent", lambda x: x["next"], conditional_map)

for tool in tools:
    workflow.add_edge(start_key=tool, end_key=END)
# workflow.add_edge(start_key="agent", end_key="VQA_tool")
# workflow.add_edge(start_key="VQA_tool", end_key=END)

graph = workflow.compile()

def lang_graph(Image,node,question,context,question_type,label_sentence,axis_dict,obj_dict,obj_filt_dict):
    print("label_sentence",label_sentence)
    # if context=="None":
    #      input={"Entity": [HumanMessage(content=node)],"Subquestions": [HumanMessage(content=question)]},
    # else:
    # input={"Entity": [HumanMessage(content=node)],"Subquestions": [HumanMessage(content=question)],"Context": [HumanMessage(content=context)]},
    for s in graph.stream(
            
            {"Image": [HumanMessage(content=Image)],"Entity": [HumanMessage(content=node)],"Subquestions": [HumanMessage(content=question)],"Context": [HumanMessage(content=context)],"question_type":[HumanMessage(content=question_type)],"label_sentence":[HumanMessage(content=label_sentence)],"axis_dict":[HumanMessage(content=axis_dict)],"obj_dict":[HumanMessage(content=obj_dict)],"obj_filt_dict":[HumanMessage(content=obj_filt_dict)]},
            {"recursion_limit": 150}
            ):
                if not "__end__" in s:
                    # print((llm))
                    print(s, end="\n-----------------\n")
                    if 'VQA_tool' in s:
                        # print("Answer from Langgraph",s['VQA_tool']['Tool_return'][0].content)
                        # print(s['VQA_tool']['Tool_return'][0].content,'VQA_tool')
                        return s['VQA_tool']['Tool_return'][0].content,'VQA_tool'
                    if 'left_right_tool' in s:
                        # print("Answer from Langgraph",s['VQA_tool']['Tool_return'][0].content)
                        return s['left_right_tool']['Tool_return'][0].content,'left_right_tool'
                    if 'bottom_top_tool' in s:
                        # print("Answer from Langgraph",s['VQA_tool']['Tool_return'][0].content)
                        return s['bottom_top_tool']['Tool_return'][0].content,'bottom_top_tool'
                    if 'judge_obj_tool' in s:
                        # print("Answer from Langgraph",s['VQA_tool']['Tool_return'][0].content)
                        return s['judge_obj_tool']['Tool_return'][0].content,'judge_obj_tool'
                    if 'size_tool' in s:
                        return s['size_tool']['Tool_return'][0].content,'size_tool'
                    if 'identity_tool' in s:
                        return s['identity_tool']['Tool_return'][0].content,'identity_tool'
