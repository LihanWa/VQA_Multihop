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

from langgraph.prebuilt import ToolExecutor
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
def search_wikipedia(query):
    URL = "https://en.wikipedia.org/w/api.php"
    
    PARAMS = {
        "action": "opensearch",
        "search": query,
        "limit": 5,
        "namespace": 0,
        "format": "json"
    }
    
    response = requests.get(URL, params=PARAMS)
    data = response.json()
    
    print(data)
    if len(data) > 1 and isinstance(data[1], list):
        for title in data[1]:
            print(title)
    return data
def search_in_rag(a):
    loader = WebBaseLoader(
        web_paths=(a[3][0],),
        bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(id="bodyContent")
    ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

#llm can't answer
prompt_a=ChatPromptTemplate.from_messages([
        SystemMessage(content='''You are an assistant for question-answering tasks. 
        Consider if you can answer the following question.
        If you have 90 percent confidence about giving a correct answer, then give me the answer.
        Otherwise, just say "I don't know". '''),
        # MessagesPlaceholder(variable_name="context"),
        MessagesPlaceholder(variable_name="question"),

])
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)#gpt-3.5-turbol

@tool("a", return_direct=False)
def a(obj:str) -> str:#imag 类别修改
    """Knowledge base tool. This tool is used to provide background information from wikipedia, which the picture itself does not include the answer (such as history, the job of a person, the habits of animals)."""
        
    ans=(prompt_a|llm).invoke({"question":["Does Dune: Part Two has a higher gross than Kung Fu Panda 4?"]})
    if (ans=="I don't know."):
   
        query='Dune: part two' #之后可以考虑用prompt或者直接通过parameter传入
        a=search_wikipedia(query)
        rag_chain=search_in_rag(a)
        ans=rag_chain.invoke("What is the gross of Dune: Part Two?")

    return ans
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