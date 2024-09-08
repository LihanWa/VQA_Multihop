# from graph_generation import *
from objects_graph import ObjModel
# from tools import *

# 'what color is the helmet wearing by the man with blue shirt'
# '[man: shirt, helmet],
# multiple man
objmodel = ObjModel()

def ans_question(nodequestion_or_edgequestion): 
    '''
    Original: {'man':[1,2,3],'shirt':[4,5,6]}
    Edge question:
    Q: Is the man [1,2,3] wearing a shirt [4,5,6]?
    A: man 1 is wearing shirt 4. man 2 is wearing shirt 5. man 3 does not wear a shirt.
    Then, A is sent to filt_instance, return editted: {'man':[1,2],'shirt':[4,5]}

    Node question:
    Q: Is the shirt blue?
    A: The color of shirt 4 is red. The color of shirt 5 is blue.
    Then, A is sent to filt_instance, return editted: {'man':[1,2],'shirt':[4]}
    '''
def filt_instance(A):
    '''
    Original: {'man':[1,2,3],'shirt':[4,5,6]}
    ---------
    可以考虑把问题(总的)放进来，来提示哪些应该筛选。
    A: man 1 is wearing shirt 4. man 2 is wearing shirt 5. man 3 does not wear a shirt.
    Edit: {'man':[1,2],'shirt':[4,5]} (return)

    A: The color of shirt 4 is red. The color of shirt 5 is blue.
    Edit: {'man':[1,2],'shirt':[4]} (return)
    '''
# def filt_instance(instances, edge_question, node_question, parent_instance):
#     '''
#     instances: [2, 4, 5]

#     'what number is labled for the red helmet(node_question) wearing by the man labeled as 3, is 2, 4 or 5'
#     '''
    # 两种情况 第一筛掉部分重复的instance， 第二筛不掉重复instance 那就汇总

def dynamic_graph_gen(static_graph=None, objects_dict=None):
    '''
    static_graph: a semantic graph extracted from question-side with gpt-3.5
                format: {}
    
    objects_dict: object level info dict extracted from the img-side with gdino:
                format: { obj1:{ 
                                number: # of recgonized objects
                                instance: list of recgonized objects info dict
                                    [instance1: {
                                        x, y, w, h: x,y coordinateds and width and height;
                                        label: description of label (number or box color),
                                    },
                                     instance2: {
                                        ......
                                     }
                                    ]
                                 },
                            obj2:{
                                ......
                            }
                
                        }

    '''
    # for node, .. in static_graph.items():
    #     # node is an object
    #     node_dict = obj_graph_gen(node)
    #     filt_instance() 
    tt = objmodel.find_obj(['bottle', 'laptop', 'book'], 'computer_twoBottles.jpg')
    print(tt)

dynamic_graph_gen()