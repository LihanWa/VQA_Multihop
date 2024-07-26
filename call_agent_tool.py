from langchain_core.messages import FunctionMessage
from langgraph.prebuilt import ToolInvocation
from tools import tool_executor,tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent

import ast
import json
def call_Face_recognition_tool(state):
    action = ToolInvocation(
        tool='e',
        # tool_input=json.loads('{"image_input":"/root/projects/Einstein2.png"}'),
        tool_input=json.loads('{"image_input":"a.png"}'),
    )
    response = tool_executor.invoke(action)#函数返回值
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"Tool_return": [function_message]}


def call_KB_tool(state):
    print("state in kb tool: ",state)
    print((state['Question']))
    print((state['Question'][0].content))
    print(type(str(state['Question'].content[0])))
    action = ToolInvocation(
        tool='a',
        # tool_input=json.loads('{"obj":"Einstein"}'),
        tool_input={'question':str(state['Question'][0])},
    )
    print("come to call kb tool")
    response = tool_executor.invoke(action)#函数返回值
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"Tool_return": [function_message]}
    # return {"messages": [function_message]}
def call_Count_text_len_tool(state):
    action = ToolInvocation(
        tool='b',
        tool_input=json.loads('{"text":"balala"}'),
    )
    response = tool_executor.invoke(action)#函数返回值
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"Tool_return": [function_message]}
def call_Identify_text_tool(state):
    action = ToolInvocation(
        tool='c',
        tool_input=json.loads('{"image":"adfs.png"}'),
    )
    response = tool_executor.invoke(action)#函数返回值
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"Tool_return": [function_message]}
def call_Identify_object_tool(state):
    action = ToolInvocation(
        tool='d',
        tool_input=json.loads('{"image":"ok.png"}'),
    )
    response = tool_executor.invoke(action)#函数返回值
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"Tool_return": [function_message]}

def call_Identify_color_tool(state):
    action = ToolInvocation(
        tool='f',
        tool_input=json.loads('{"image_part":"a part"}'),
    )
    response = tool_executor.invoke(action)#函数返回值
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"Tool_return": [function_message]}

def call_Identify_space_relation_tool(state):
    action = ToolInvocation(
        tool='g',
        tool_input=json.loads('{"image_part":"a part"}'),
    )
    response = tool_executor.invoke(action)#函数返回值
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"Tool_return": [function_message]}


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
system_prompt = (
    "You are a helpful AI assistant, and you should mainly depend on the Question, Previously used tools and The corresponded return value of the tools to make decisions."
    " Use the provided tools to progress towards answering the question."
    " If you don't want to use any tools, it's OK and you should choose FINISH. And it is better than selecting useless tools."
)
# options=["a","b","c","d","FINISH"]
options=["a","b","c","d","e","f","g","FINISH"]
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt),
    MessagesPlaceholder(variable_name="Question"),
    MessagesPlaceholder(variable_name="Previously used tools and The corresponded return value of the tools"),
    # MessagesPlaceholder(variable_name="Question"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    # MessagesPlaceholder(
    #         variable_name="chat_history"
    #     ),
    # ("assistant",[HumanMessage(content='Does the man in this picture do research about the cat'), 'Face_recognition_tool', FunctionMessage(content='Bruce Lee', name='Face_recognition_tool')]),
    ("system",
    # " If you have used a tool in this task, you should not use the same tool again, because the tool will always gives the same answer for the same question."
    " If you have used a tool in this task, you should not use the same tool again."
    " So even you can't solve the promblem or can't get a satisfying answer (For example, the returned answer of tool is 'do not know, not clear'), you should not use the same tool again."
    " Given the conversation above, who should act next? Or should we FINISH? Select one of: {options} (The output should be one of the elements in the options)"),
]).partial(options=str(options))

agent = create_openai_tools_agent(llm, tools, prompt)
# executor = AgentExecutor(agent=agent, tools=toool,memory=memory,verbose=True)
executor = AgentExecutor(agent=agent, tools=tools)
def create_agents(state):
    if('next' in state): state.pop('next')
    print('state',state)
    new_state = {
        'Question': [],
        'Previously used tools and The corresponded return value of the tools': []
    }
    new_state['Question'].append(f"Question: {state['Question'][0]}")
    if 'Tool_return' in state and state['Tool_return'] is not None:
        for tool_return in state['Tool_return']:
            new_state['Previously used tools and The corresponded return value of the tools'].append(
                f"Previously used tool: {tool_return.name}. The corresponded return value of the tools: {tool_return.content}"
            )
    # print(new_state)
    print('new state:',new_state)
    res=executor.invoke(new_state)
    # res=executor.invoke(state)
    print(res)
    return {"next":res['output']}
