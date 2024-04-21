from langchain.tools import tool
from langchain.tools.render import format_tool_to_openai_function
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.prebuilt import ToolExecutor


@tool("a", return_direct=False)
def a(obj:str) -> str:#imag 类别修改
    """Knowledge base tool. This tool is used to provide background information from wikipedia, which the picture itself does not include the answer (such as history, the job of a person, the habits of animals)."""
    # query = obj
    
    # a=search_wikipedia(query)
    # rag_chain=search_in_rag(a)
    # ans=rag_chain.invoke("Does Einstein do research about cat")
    return "I can't give a good answer."
    # return ans
@tool("b", return_direct=False)
def b(text:str) -> int:#imag 类别修改
    """This tool should not be used on human, animals or anything else except texts. It is used to count the text length of the text."""
    return "I can't give a good answer."
@tool("c", return_direct=False)
def c(image:str) -> int:#imag 类别修改
    """This tool should not be used on human, animals or anything else except texts. It is used to identify the text content in the image"""
    return 5
@tool("d", return_direct=False)
def d(image:str) -> str:#imag 类别修改
    """This tool should not be used on people in the image. It is used to identify a object in the image, such as a pencil, a window and so on."""
    return "We don't know."
@tool("e", return_direct=False)
def e(image_input: str) -> str:#imag 类别修改
    """Human face recognition. This tool is only used for human. It is used to figure out the name of the person."""

    return "We don't know."
tools=[a,b,c,d,e]

# functions = [format_tool_to_openai_function(t) for t in tools]
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol
# llm = llm.bind_functions(functions)
tool_executor = ToolExecutor(tools)