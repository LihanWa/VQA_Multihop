import base64
import json
import mimetypes
import os
import requests
import sys
from dashscope import MultiModalConversation

import re
os.environ["DASHSCOPE_API_KEY"]= 'sk-ac7aca0206ae4da9a517628e5fa2170f'
def call_with_local_file():
    local_file_path = 'file:///root/projects/code_lihan/experiment_questions_data/test_0-200/0_n161313_new.jpg'
    local_file_path2 = 'file://The_local_absolute_file_path2'
    messages = [{
        'role': 'user',
        'content': [
            {
                'image': local_file_path
            },
            # {
            #     'image': local_file_path2
            # },
            {
                'text': '图片里有什么东西?'
            },
        ]
    }]
    response = MultiModalConversation.call(model='qwen-vl-max', messages=messages)
    print(response)
def create_payload_edge(image: str, prompt: str, label_sentence: str,model="gpt-4o-2024-05-13", max_tokens=200, detail="high"):
    """Creates the payload for the API request."""
    spe_words = ['photo', 'image', 'photograph', 'Photo', 'Image', 'Photograph', 'place']
    prompt_words = prompt.split(' ')
    for i, prompt_word in enumerate(prompt_words):
        if prompt_word in spe_words:
            break
        if i == len(prompt_words) - 1:
            prompt = prompt[:-1] + " in the image?"
    print(prompt)
    local_file_path = f'file://{image}'

    
    messages = [
        {
            "role": "system",
            "content": [
                {

                    "text": '''AI assistant helps answer a question based on the image. AI assistant should see the image from the perspective of image viewer when considering left and right. When considering 'front' and 'behind', the behind one is the deeper one in the image. Please just answer the question itself, and do not say anything irrelevant about the question. /
                   If there is a label on the object, it means the label is only for the object and they correspond. Answer the quesiton with the label of an object. /
                   For the "Thought" and "Answer", AI assistant should consider objects by permutation./
                   Next is an example, you can learn from the conversation between the assistant and the user. /
                   In the image, there are three cups marked by 1,2,3 and the liquid in the cup is marked by 4,5,6. The color of the cup mark is pink, and the color of the liquid mark is blue. Although the marks are close, AI assistant should divide them./'''
                },
                
            ]
            
        },
        
        {
            "role": "user",
            "content": [
                {
                    "text": "My question is: Does cup [1] contain liquid [4,5,6]? ",
                },
            ],
        },
        {
            "role": "assistant",
            "content":"Thought: For the question, I only have to consider liquid marked by 4 or 5 or 6, and cup marked by 1. Since the liquid marked by 4 is in the cup marked by 1, the liquid marked by 5 is not in the cup marked by 1, and the liquid marked by 6 is not in the cup marked by 1. Answer: Cup [1] contains liquid [5] and does not contain liquid[4,6]",

        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": "My question is: Does cup [3] contains liquid [4,5]?",
                },
            ],
        },
        {
            "role": "assistant",
            "content":"Thought: For the question, I only have to consider liquid marked by 4 or 5, and cup marked by 3. Since the liquid marked by 4 is not in the cup marked by 3, the liquid marked by 5 is not in the cup marked by 3. Answer: Cup [3] does not contain liquid [4,5].",

        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": "My question is: Does cup [2,3] contains liquid [4,5]?",
                },
            ],
        },
        {
            'role': 'assistant',
            'content': 'Thought: For the question, I only have to consider liquid marked by 4 or 5, and cup marked by 2 or 3. Since the liquid marked by 4 is not in the cup marked by 3, the liquid marked by 5 is not in the cup marked by 3, the cup marked by 2 contains liquid marked by 4, and the the cup marked by 2 does not contain liquid marked by 5. Answer: Cup [3] does not contain liquid [4,5]. Cup [2] contains liquid [4] and does not contain liquid [5].'
        },
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''Next is another image and the question from the user, AI assistant should see the image from the perspective of image viewer when considering left and right. When considering 'front' and 'behind', the behind one is the deeper one in the image. You should learn from the conversation above to answer. The format of the answer should be: "Thought: ... Answer:...".'''
                      + label_sentence+'''Although the marks are close, you should divide them. The image is as follows given by the user./'''
                },
                
            ]
            
        },
        
        {
            "role": "user",
            "content": [
                
                {
                    
                    "text": "My question is: "+prompt
                },
                {
                    'image': local_file_path
                },
            ],
        },

    ]
    # print("Edge question label_sentence: ",label_sentence)
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }

def create_payload_final_ans(image: str, prompt: str, label_sentence: str, context: str, model="gpt-4o-2024-05-13", max_tokens=200, detail="high", temperature=0.0):
    """Creates the payload for the API request."""
    # print(context)
    print('image_dir',image)
    spe_words = ['photo', 'image', 'photograph', 'Photo', 'Image', 'Photograph', 'place']
    prompt_words = prompt.split(' ')
    for i, prompt_word in enumerate(prompt_words):
        if prompt_word in spe_words:
            break
        if i == len(prompt_words) - 1:
            prompt = prompt[:-1] + " in the image?"
    print(prompt)
    local_file_path = f'file://{image}'
    
    if context== '':
        text=f'My VQA question is: {prompt}'
    else:
        text=f'The context information is as follows: <{context}>. My VQA question is: {prompt}'
    print('Contexto_to_gpt4:',text)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''GPT-4 (AI assistant) is tasked with answering a VQA question based on an image and provided context. For 'left' and 'right' in the user's question, the AI should consider the image strictly from the pixel-based left and right perspective as seen by an image viewer. The term 'behind' refers to elements that are deeper within the image frame.
The context provides essential information about the image, and the AI should adhere to the following key guidelines:
- Responses should be concise, typically "Yes" or "No." For questions about "What" or "Where," the answer should be no more than two words, indicating either the object or the place.
- The AI must provide a definite answer and should avoid responses like "Not clear" or "I do not know."
- The task may require reasoning based on the visual and contextual information provided.
In the context, brackets '[]' are used to mark different instances of an object. For example, A[1] indicates the first instance of object A, and A[2] indicates a second, distinct instance.
However, AI assistant should not answer with the mark. For example, you should not say "Animal 2", instead, you should see what the animal is and then your answer should be like horse or cat which is a species of Animal 2.

The context and prompt are as follows:'''
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": text
                },
                {
                    'image': local_file_path
                },
            ],
        },
    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature  # Adding temperature to the API request
    }
def create_payload_size_tool(image: str, question: str, label_sentence: str, model="gpt-4o-2024-05-13", max_tokens=200, detail="high", temperature=0.0):
    """Creates the payload for the API request."""

    local_file_path = f'file://{image}'
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''AI assistant is tasked with answering a VQA question based on an image and provided context. If the VQA question ask the size of an object, AI assistant should answer the specific object is either 'large' or 'small' instead of "medium". /
For example, if the question asks "Is the plate[1] large and ..?" AI assistant should say "The plate[1] is large and ..." or "The plate[1] is small and ..." /
In the question, brackets '[]' are used to mark different instances of an object.'''+label_sentence+'''The question is as follows:'''
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": question
                },
                {
                    'image': local_file_path
                },
            ],
        },
    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature  # Adding temperature to the API request
    }
def create_payload_identity_tool(image: str, question: str, label_sentence: str, model="gpt-4o-2024-05-13", max_tokens=200, detail="high", temperature=0.0):
    """Creates the payload for the API request."""

    local_file_path = f'file://{image}'
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''AI assistant is tasked with answering a VQA question based on an image and provided context. The question is always "Who ...?" Since usually it has a few ways to describe a person, the answer choices are limited here. /
If do not have a specific identity, you can choose from 'man', 'woman', 'boy', 'girl' or 'people'. If it seems the person has an identity, you can choose from 'skateboarder', 'player', 'batter', 'umpire', 'skier','snowboarder', 'catcher','surfer'./
You should answer a full sentence. For example, "Woman is watching TV"./
In the question, brackets '[]' are used to mark different instances of an object.'''+label_sentence+'''The question is as follows:'''
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": question
                },
                {
                    'image': local_file_path
                },
            ],
        },
    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature  # Adding temperature to the API request
    }




def mmbench_payload(image: str, question: str, options: str, model="gpt-4o-2024-05-13", max_tokens=200, detail="high", temperature=0.0):
    """Creates the payload for the API request."""
    # print(context)
    local_file_path = f'file://{image}'
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''AI assistant will be given a question, an image and four options, choose the best one, and choose one from the options including a letter such as 'A' and the content of the answer.'''
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": f'The question is {question}. The four options are {options}. The image is as follows:'
                },
                {
                    'image': local_file_path
                },
            ],
        },
    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature  # Adding temperature to the API request
    }

def judge_few_objs(image: str, prompt: str, entity_str,model="gpt-4o-2024-05-13", max_tokens=200, detail="high", temperature=0.0):
    """Creates the payload for the API request."""
    # print(context)
    # spe_words = ['photo', 'image', 'photograph', 'Photo', 'Image', 'Photograph', 'place']
    # prompt_words = prompt.split(' ')
    # for i, prompt_word in enumerate(prompt_words):
    #     if prompt_word in spe_words:
    #         break
    #     if i == len(prompt_words) - 1:
    #         prompt = prompt[:-1] + " in the image?"
    # print(prompt)
    local_file_path = f'file://{image}'
    

    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": f'''AI assistant should help find the object in sub-images. Specifically are as follows: AI assistant will be given a image, and this image contains one or a few blurry sub-images which are pasted. Although they are blurry, try your best to identify it./
AI assistant should analyze each sub-image one by one, find if the {entity_str} are in the sub-image and answer the question. Although sometimes the object is small or is behind something, you should find it. AI assistant should be format output by a dictionary: {{'mark':ans}}, (mark is a number, ans should be 'Yes' if found and 'No' if not found. Next are two examples about the format of AI assistant answer:'''
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": "Can you find banana in the sub-image marked by [0,1]?",
                },
            ],
        },
        {
            "role": "assistant",
            "content": '{"0":"Yes","1":"No"}'
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": "Can you find apple in the sub-image marked by [0,1,2]?",
                },
            ],
        },
        {
            "role": "assistant",
            "content": '{"0":"No","1":"Yes","2":"Yes"}'
        },
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''Okay, now you know the format.'''   }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": f'{prompt}'
                },
                {
                    'image': local_file_path
                },
            ],
        },
    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature  # Adding temperature to the API request
    }


def create_payload_node(image: str, prompt: str, label_sentence:str,model="gpt-4o-2024-05-13", max_tokens=200, detail="high"):
    """Creates the payload for the API request."""
    spe_words = ['photo', 'image', 'photograph', 'Photo', 'Image', 'Photograph', 'place']
    prompt_words = prompt.split(' ')
    for i, prompt_word in enumerate(prompt_words):
        if prompt_word in spe_words:
            break
        if i == len(prompt_words) - 1:
            prompt = prompt[:-1] + " in the image?"
    print(prompt)
    local_file_path = f'file://{image}'

    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''AI assistant help answer a question based on the image. When considering left and right, AI assistant should see the image from the perspective of image viewer. When considering 'front' and 'behind', the behind one is the deeper one in the image. Please just answer the question itself, and do not say anything irrelevant about the question. /
                   If there is a label on the object, it means the label is only for the object and they correspond. Answer the quesiton with the label of an object. /
                   Next is an example, you can learn from the conversation between the assistant and the user. /
                   In the image, there are three cups marked by 1,2,3 and the liquid in the cup is marked by 4,5,6. The color of the cup mark is pink, and the color of the liquid mark is blue. Although the marks are close, you should divide them. 
                     /'''
                },
                # image0_info,
            ]
            
        },
        
        {
            "role": "user",
            "content": [
                {
                    
                    "text": "My question is: Is the liquid [4] light brown? ",
                },
            ],
        },
        {
            "role": "assistant",
            "content":"Thought: Based on the question, I only have to consider liquid marked by 4. Since the liquid marked by 4 is white. Answer: Liquid [4] is not light brown.",

        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": "My question is: Is liquid [4,5] light brown?",
                },
            ],
        },
        {
            "role": "assistant",
            "content":"Thought: Based on the question, I only have to consider liquid marked by 4 or 5. Since the liquid marked by 4 is not light brown, the liquid marked by 5 is light brown. Answer: Liquid [4] is not light brown, liquid [5] is light brown.",

        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": "My question is: Is liquid [4,5,6] dark brown?",
                },
            ],
        },
        {
            "role": "assistant",
            "content":"Thought: For the question, I only have to consider liquid marked by 4 or 5 or 6. Since the liquid marked by 4 is not dark brown, the liquid marked by 5 is not dark brown, the liquid marked by 6 is dark brown. Answer: Liquid [4] is not dark brown, liquid [5] is not dark brown, liquid [6] is dark brown.",
        },
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''Next is another image and the question from the user. AI assistant should learn from the conversation above to give the answer. The format of the answer should be: "Thought: ... Answer:...".'''
                    +label_sentence+''' Although the marks are close, you should divide them. The image is given by the user as follows:/'''
                },
              
            ]
            
        },
        
        {
            "role": "user",
            "content": [
                {
                    
                    "text": "My question is: "+prompt
                },
                {
                    'image': local_file_path
                },
            ],
        },

    ]
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }


def create_payload_normal(image: str, prompt: str,model="gpt-4o-2024-05-13", max_tokens=200, detail="high"):
    """Creates the payload for the API request."""

    local_file_path = f'file://{image}'    
    spe_words = ['photo', 'image', 'photograph', 'Photo', 'Image', 'Photograph', 'place']
    prompt_words = prompt.split(' ')
    for i, prompt_word in enumerate(prompt_words):
        if prompt_word in spe_words:
            break
        if i == len(prompt_words) - 1:
            prompt = prompt[:-1] + " in the image?"
    print(prompt)
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''You are a smart AI assistant for helping me answer a VQA question based on the image and the prompt. /
                    Make your answer as short as possible.
                    The image is as follows:'''
                },
                
            ]
            
        },
        
        
        {
            "role": "user",
            "content": [
                {
                    
                    "text": "My VQA question is: "+prompt
                },
                {
                    'image': local_file_path
                },
            ],
        },

    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }
def find_other_obj(image: str, prompt: str,entity:str,model="gpt-4o-2024-05-13", max_tokens=200, detail="high"):
    """Creates the payload for the API request."""

    local_file_path = f'file://{image}'
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": f'''AI assistant should help figure out if there are other {entity} except {entity} in a yellow bounding box. Do not worry about the black part, they are just masks, so just ignore it./
AI assistant do not have to reason, just say what you see. If AI assistant make sure there are other {entity}, describe their location. For example, AI assistant can output 'There are (or is) other ... without mark or bounding box, and they (or it) is (location).' in less than 30 words.'''


                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": prompt
                },
                {
                    'image': local_file_path
                },
            ],
        },

    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }

def find_not_found_obj(image: str, prompt: str,entity:list,model="gpt-4o-2024-05-13", max_tokens=200, detail="high"):

    local_file_path = f'file://{image}'
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": f'''AI assistant should help figure out if the position of {entity} in the image if the entity exists. For example, AI assistant can output 'The {entity} is (location)' in less than 20 words. If AI assistant does not find {entity}, output "There is no {entity} in the image." '''


                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": prompt
                },
                {
                    'image': local_file_path
                },
            ],
        },

    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }
def conclude_answer(image: str, question: str,answer:str,model="gpt-4o-2024-05-13", max_tokens=200, detail="high"):

    local_file_path = f'file://{image}'
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": f'''AI assistant should help conclude answers if there are more than one element, which are usually combined by 'and' or ','. AI assistant should process the conclude the answer based on an image and a question. Generally AI assistant has two ways to do it, it can either choose the most special or obvious one according to the image or summarize all of the answers. /
    Next are three examples for you.'''
                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": f"The question is 'What is the person wearing?'. The answer is 'a cap, a navy blue jersey with \"Brewers\" on it, light gray pants, black socks, black shoes, and a baseball glove.' (The image mainly shows a baseball player wearing a baseball uniform.)"
                },
            ],
        },
        {
            "role": "assistant",
            "content": "baseball uniform" 
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": f"The question is 'What is the person wearing?'. The answer is 'helmet, jeans, shirt.' (The image shows the person is a normal person, who does not seem to have an identity such as player. And the helmet is the most special and obvious one.)"
                },
            ],
        },
        {
            "role": "assistant",
            "content": "helmet" 
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": f"The question is 'What are the animals on the shore?'. The answer is 'Elephant, lion, capybara.' (The image shows only elephat is on the shore.)"
                },
            ],
        },
        {
            "role": "assistant",
            "content": "elephant"
        },
        {
            "role": "system",
            "content": [
                {
                    
                    "text": f'''Okay, now you know how to solve the following one.'''
                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": f"The question is '{question}'. The choices are '{answer}'. The image is as follows:"
                },
                {
                    'image': local_file_path
                },
            ],
        }
    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }
def judge_obj(image: str, prompt: str,entity:list,model="gpt-4o-2024-05-13", max_tokens=200, detail="high"):

    local_file_path = f'file://{image}'
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": f'''AI assistant should help judge if the text mark and yellow box of {entity} in the image are correct. AI assistant can output 'The text mark of yellow box of {entity} is incorrect', or 'All correct'. '''


                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    
                    "text": prompt
                },
                {
                    'image': local_file_path
                },
            ],
        },

    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }

def check_mark(image: str,model="gpt-4o-2024-05-13", max_tokens=200, detail="high"):
    """Creates the payload for the API request."""

    local_file_path = f'file://{image}'
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''AI assistant should check if the text mark on the top left corner are correct. The mea'''

#                     If the object of the red text mark on the upper left corner cannot be found in the yellow bounding box, it means the text mark is not correct. /
# Just output the red text mark without saying other irrelavant things. For example, if the red text mark is "cat 4", and you can't find a cat in this corresponding bounding box, ouput "cat 4".'''
                },
                
            ]
            
        },
        
        
        {
            "role": "user",
            "content": [
                # {
                #     
                #     "text": f"{check_sentence} If you think any mark is very inappropriate, figure it out. "
                # },
                {
                    'image': local_file_path
                },
            ],
        },

    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }


def vcr_ans_q(image: str, question: str,options:str,model="gpt-4o-2024-05-13", max_tokens=200, detail="high"):
    """Creates the payload for the API request."""

    local_file_path = f'file://{image}'
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''AI assistant should answer choose from "A","B",C","D" based on an image and options. The image is from a movie plot. Usually AI assistant should takes reasoning to select the correct one from options. Some entities in the image are labeled, which is surrounded by a yellow box and with red texts on the top left corner. In the question, the number after an entity (usually a person) is corresponded with the label in the image. /
AI assistant should choose from "A","B",C","D". For instance AI assistant should say "A". '''
                },
                
            ]
            
        },
        
        
        {
            "role": "user",
            "content": [
                {
                    
                    "text": f"The question is: '{question}'. The options are '{options}'. The image is as follows:"
                },
                {
                    'image': local_file_path
                },
            ],
        },

    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }
def vcr_des(image: str,question: str,objects:str,model="gpt-4o-2024-05-13", max_tokens=200, detail="high"):
    """Creates the payload for the API request."""

    local_file_path = f'file://{image}'
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''AI assistant should describe the objects, which is surrounded by a yellow box and with red name on the top left corner. AI assistant will be given some questions, targeted objects' name and an image. /
AI assistant should just describe the objects respectively, withouting answering the question or saying anything useless or irrelevant. The description of the object should be useful for answering the question. /
The format of AI assistant should be like: <Description of ..: .. Description of ..: ..> '''
                },   
            ]
            
        },
        
        {
            "role": "user",
            "content": [
                {
                    
                    "text": f'''The question is "{question}". The objects are "{objects}". The image is as follows:'''
                },
                {
                    'image': local_file_path
                },
            ],
        },

    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }
def vcr_des_ans(image: str,question: str,multi_choice:str,objects:str,model="gpt-4o-2024-05-13", max_tokens=200, detail="high"):
    """Creates the payload for the API request."""

    local_file_path = f'file://{image}'
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    
                    "text": '''AI assistant should answer the question by choosing the most appropriate one from the multiple choices. AI assistant will be given an image and a brief description about objects in the image, which are both important for answer the question. Just answer 'A' or 'B' or 'C' or 'D', the information are given by the ussers as follows.'''                },   
            ]
            
        },
        
        {
            "role": "user",
            "content": [
                {
                    
                    "text": f'''The question is "{question}". The multiple choices are :"{multi_choice} Just answer 'A' or 'B' or 'C' or 'D' The objects are "{objects}". The image is as follows:'''
                },
                {
                    'image': local_file_path
                },
            ],
        },

    ]

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }
def query_openai(payload):
    """Sends a request to the OpenAI API and prints the response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()
