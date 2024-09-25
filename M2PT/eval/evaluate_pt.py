import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from M2PT.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from M2PT.conversation import conv_templates, SeparatorStyle
from M2PT.model.builder_PTTrained import load_pretrained_model
from M2PT.utils import disable_torch_init
from M2PT.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import pdb
from pathlib import Path
import random
import torch.nn.functional as F
from transformers import StoppingCriteria

#EVAL_TASKS = ['text_vqa', 'visual_spatial_reasoning', 'pope_rand', 'cifar_100', 'cifar_10', 'mnist', 'snli_ve_classification', 'pope_adv', 'pope_pop']

EVAL_TASKS = ['text_vqa', 'visual_spatial_reasoning', 'pope_rand', 'pope_adv', 'pope_pop', 'cifar_10']
DIST_TOKEN = {'aok_vqa':'Provide a very short and abstract response.', 'science_qa':"Provide a very short and abstract response.", 'visit':'Provide a very long response.', 'text_vqa':"Provide a very short and abstract response.", 'visual_spatial_reasoning': "Provide a very short and abstract response.", 'winoground':"Provide a short response.", 'medic_damage_severity':"Provide a very short and abstract response.",'medic_informative':"Provide a very short and abstract response.", 'medic_disaster_types':"Provide a very short and abstract response.",'medic_humanitarian':"Provide a very short and abstract response.", 'aokvqa_rational':"Provide a long response.", 'cifar_10':"Provide a very short and abstract response.", 'cifar_100':"Provide a very short and abstract response.",'miniImagenet':"Provide a very short and abstract response.",'mnist':"Provide a very short and abstract response.",'pope_adv':"Provide a very short and abstract response.",'pope_pop':"Provide a very short and abstract response.",'pope_rand':"Provide a very short and abstract response.",'snli_ve_answer_choice':'Provide a very short and abstract response.','snli_ve_classification':'Provide a short response.'}

def get_task_def(task):
    if task == 'image_caption':
        instructs=[
            f"""Generate textual description of the given image."""
        ]
    elif task == 'GQA':
        instructs = [
            f"""Answer the compositional question in natural language based on the content of the image. The questions involve multiple reasoning skills, spatial understanding and multi-step inference""",
        ]
    elif task == 'VQAv2':
        instructs = [
            f"""Answer the question in natural language based on the content of the image.""",
        ]
    elif task == 'visualgenome_vqa':
        instructs = [
            f"""Answer the question in natural language based on the content of the image.""",
        ]
    elif task == 'ok_vqa':
        instructs = [
            f"""Answer the question in natural language based on the content of the image. The questions require external knowledge to answer.""",
        ]
    elif task == 'VQA':
        instructs = [
            f"In this task, we ask you a question about an image and provide you with some options containing the correct answer. You should select the best answer from the options based on the content of the image."]
    elif task == 'GC': # same as region_caption
        instructs = [
            f"""The goal of this task is to generate description for part of the image within a bounding box. We provide you with the coordinate of top-left corner of the bounding box denoted as x1 y1 and the coordinate of bottom-right corner of the bouding box denoted as x2 y2. The format of the input bounding box is x1 y1 x2 y2.""",
        ]
    elif task == 'GC_selection': # same as region_caption
        instructs = [
            f"""In this task, you are given some natual langugae sentences in the options and you are required to select the sentence that works best as a caption for part of the image within a bounding box. A bounding box is an imaginary rectangle that outlines an object or part of the scene in an image. We provide you with the coordinate of top-left corner of the bounding box denoted as x1 y1 and the coordinate of bottom-right corner of the bouding box denoted as x2 y2. The format of the input bounding box is "x1 y1 x2 y2".""",
        ]

    elif task == 'VG':
        instructs = [
            f"""In this task, you are asked to localize the region in an image that is described by the given text. The region should be specified via a bounding box which is an imaginary rectangle that outlines part of the scene in an image. Your output should contain the coordinate of top-left corner x1 y1 and the coordinate of bottom-right corner x2 y2 of the bouding box. Specifically, the output should look like "x1 y1 x2 y2"."""
        ]

    elif task == 'VG_selection': # same as region_caption
        instructs = [
            f"""We give you a caption for part of the image specified by a bounding box. A bounding box is an imaginary rectangle that outlines part of the scene in an image. We provide you with some candidate bounding boxes in the options. The format of the bounding boxes is x1 y1 x2 y2. x1 y1 denotes the coordinate of top-left corner and x2 y2 denotes the bottom-right corner of a bounding box. You need to select the bounding box that aligns best with the given caption.""",
        ]

    elif task == 'object_grounding':
        instructs = [
            f"""In this task, we ask you to identify the object in a region of an image. The region is specified via the coordinates of a rectangle. The input format of the region is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner of the rectangle. You need to first localize the rectangular region in the image and then identify what is the main object in that region."""
        ]
    elif task == 'object_region_match':
        instructs = [
            f"""In this task, we will provide you with an object name and a bounding box, and you will decide if the object that we give you is the same object in the bounding box. We specify the bounding box via two coordinates x1 y1 and x2 y2 which denotes the position of top-left corner and the bottom-right corner of the rectangular bounding box, respectively. Instead of answering the question by using your own words, select the best answer from options.""",
        ]
    elif task == 'object_match':
        instructs = [
            f"""In this task, we provide you with two rectangular regions (i.e., region 1 and region 2) in an image and you will decide if the object in region 1 is the same as the object in region 2. Each region is a imaginary rectangular box in the image that outlines an object. The region format is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the rectangle. You should select the best answer from options."""
        ]
    elif task == 'question_image_match':
        instructs = [
            f"""In this task, you need to decide if the image contains enough information to answer a visual question. You will select your answer from options.""",
        ]
    elif task == 'object_region_selection':
        instructs = [
            f"""In this task, we provide you with an object name and you need to select the region from options that contains the object. We define a region as a rectangular box and the format is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the rectangle.""",
        ]
    # modify
    elif task == 'missing_object_selection':
        instructs = [
            f"""In this task, we provide you with some regions and you need to select an object from options that do not appear in any of the region. A region is a rectangular box and the format is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the rectangle. Select "None" if all the objects appear in all regions.""",
        ]
    elif task == 'ITM':
        instructs = [
                     f"""In this task you are given some text and you need to decide if the text describe the image.""",
        ]
    # modify
    elif task == 'region_object_selection':
        instructs = [
            f"""Select objects from the options that appear in at least one of the regions. Select "None" if you can't find any object that appears in any region.""",
        ]
    # modify
    elif task == 'region_generation': # mscoco
        instructs = [
            f"""In this task, you are asked to identify all the regions in the image that contain the given object. The region is defined as a rectangular box and the format is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner the rectangle. In your output you should use a single space to separate the regions.""",
        ]

    elif task == 'region_caption_match':
        instructs = [
            f"""We provide you with a natural langugae sentence and the coordinates of a region in the image. You need to decide if the sentence matches the content in the region. Do not consider the content of the image outside of the region. The region is defined as a rectangular box and the format is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the rectangle.""",
        ]
    elif task == 'object_relationship':
        instructs = [
            f"""In this task, you will decide the relationship between two objects in the image. The relationship can be about the relative position of two objects such as "on", "in", "behind" ... and the action of one object performing to another object such as "taking", "researching for", "feeding"... One of the objects is the "subject" and the other object is the "object" in the relationship. The two objects are specified via their regions. A region is an imaginary rectangular box in the image. The format of the box is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner of the box."""
        ]
    elif task == 'visual_object_identification':
        instructs = [
            f"""In this task, you need to predict the name of the "object" given a "subject" in a visual relationship. The subject is specified via its bounding box which is a rectangualr region in the image. The rectangle is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner."""
        ]
    elif task == 'visual_subject_identification':
        instructs = [
            f"""In this task, you need to predict the name of the "subject" given an "object" in a visual relationship. The object is specified via its bounding box which is a rectangualr region in the image. The rectangle is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner.""",
        ]
    elif task == 'visual_object_region':
        instructs = [
            f"""In this task, you need to predict the region of the "object" given the name of a "subject" in a visual relationship. The subject is specified via its bounding box which is a rectangualr region in the image. The rectangle is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner.""",
        ]
    elif task == 'visual_subject_region':
        instructs = [
            f"""In this task, you need to predict the region of the "subject" given an "object" in a visual relationship. The object is specified via its bounding box which is a rectangualr region in the image. The rectangle is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner."""
        ]
    elif task == 'descriptive_object_region_generate':
        instructs = [
            f"""In this task, we provide you a description of an object in the image. The description is about distinct features such as "shape", "color", and "position" of the object. You need to identify a single object in the image that satisfies the description. Once you identify the object, output its region in the format of x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner of the bounding box.""",
        ]
    elif task == 'descriptive_object_region_select':
        instructs = [
            f"""In this task, we provide you a description of an object in the image. The description is about distinct features such as "shape", "color", and "position" of the object. You need to identify a single object in the image that satisfies the description. Once you identify the object, output its region in the format of x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner of the bounding box.""",
        ]
    elif task == 'object_description_generate':
        instructs = [
            f"""In this task, we ask you to generate a referring expression about an object in the image. The referring expression is a natual language sentence that describes the distinct properties of an object so that people can identify the object by reading it. To aviod confusion, we specify the object via its bounding box. The bounding box has the format x1 y1 x2 y2 where x1 y1 specifies its coordinate of the top-left corner and x2 y2 specifies its coordinate of the bottom-right corner of the rectangular box.""",
        ]
    elif task == 'image_quality':
        instructs = [
            f"""You are given an image and you need to decide the quality issue of the image. If the image is not clear or out of focus, select "blur". If the main object is at a cornern of the image or most part of the main object is not in the image, select "bad framing". If the object of interest in the image is blocked by another object that is very closed to the camera, select "obscured". If the scene or objects in the image is upside down, or in a bad rotated view, select "rotation". If there is no quality issue with the image, select "no flaws". If the quality issue is not listed in the options, select "other"."""
                     ]
    elif task == 'text_localization':
        instructs = [
            f"""There is some text on the image and we want you to localize some letters in the text. We provide you with the letters (can be a single letter or a sequence of letters) and you need to select a bounding box from options for the provided letters. The format of the bounding box is x1 y1 x2 y2 where x1 y1 specifies its coordinate of the top-left corner and x2 y2 specifies its coordinate of the bottom-right corner of the rectangular box."""
        ]
    elif task == 'text_legibility':
        instructs = [
            f"""Decide if the text in the given region is legible (clear and complete) or illegible (not clear and not complete). The region is a imaginary rectangle in the image and the format of the region is x1 y1 x2 y2 where x1 y1 specifies its coordinate of the top-left corner and x2 y2 specifies its coordinate of the bottom-right corner.""",
        ]
    elif task == 'text_type':
        instructs = [
            f"""Look at the text in the given region and select the text type from options. The text types are "handwritten" and "machine printed". The region is a imaginary rectangle on the image and the format of the region is x1 y1 x2 y2 where x1 y1 specifies its coordinate of the top-left corner and x2 y2 specifies its coordinate of the bottom-right corner."""
        ]
    elif task == 'region_text_match':
        instructs = [
            f"""In this task, we provide you with some text and an image. You need to decide if the text on part of the image is the same as the text we provided.  We specify the part of the image via a bounding box. A bounding box is a imaginary rectangle and its format is x1 y1 x2 y2. x1 and y1 denotes the coordinate of the top-lefp corner and the x2 y2 denotes the coordinate of the right-bottom corner."""
        ]
    elif task == 'multimodal_factual_checking':
        instructs = [
            f"Determine if the given claim is factually supported by both the image and the context Choose your answer from the provided options."
        ]
    elif task == 'wikihow_next_step':
        instructs = [
            f"Given the task, the history of completed steps, and the current step with its corresponding image, determine the next step for this task. Consider the task's goal and the context provided to generate the appropriate subsequent step.",
        ]
    elif task == 'wikihow_text_image_step_order':
        instructs = [
            f"Given the task and the current step, determine if the content of the image represents the next or previous step in the process. Choose your answer from the provided options.",
        ]
    elif task == 'wikihow_image_text_step_order':
        instructs = [
            f"Given the task and the current step represented by the image, determine if the provided text describes the next or previous step in the process. Consider the overall goal and the context of the step shown in the image to make your decision. Choose your answer from the provided options.",
        ]
    elif task == 'wikihow_immediate_next_step_selection':
        instructs = [
            f"Given the task and the current step represented by the image, identify the immediate next step in the process. Consider the overall goal and the context of the step shown in the image, and select the correct next step from the provided options.",
        ]
    elif task == 'image_text_selection':
        instructs = [f"""Examine the image provided and choose the text from the options that best describes it. Consider the content and context of the image, and select the most accurate caption from the given options.""",
                     ]
    elif task == 'visual_attribute':
        instructs = [
            f"""Examine the image and the specified region within it. A bounding box is an imaginary rectangle defined by coordinates x1 y1 x2 y2, where x1 y1 represents the top-left corner and x2 y2 represents the bottom-right corner. Consider the object's properties and characteristics within the specified region, identify the attribute of the object, and select the correct option from the given choices.""",
        ]
    # image generation tasks
    elif task == 'infilling':
        instructs = [
            f"Examine the image containing a filled black box representing the missing portion. Your task is to generate only the content of the missing part, considering the context and content of the visible area. Do not recreate the entire image with the missing part filled in; focus solely on generating the content for the missing region itself.",
        ]
    elif task == 'im_region_extraction':
        instructs = [
            f"Examine the image and concentrate on the specified region. The region is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner. Extract the portion of the original image defined by the given region to create a new image. The extracted image should appear as if it was directly taken from the original image, maintaining the same visual quality and content as the designated area.",
        ]
    elif task == 'im_descriptive_infilling':
        instructs = [
            f"Examine the image containing a filled black box representing the missing or obscured portion. Use the provided description to generate the content of the missing region. Focus solely on generating the content for the missing region, taking into account the context provided by the description. Ensure that the generated content integrates seamlessly with the visible parts of the image.",
        ]
    elif task == 'image_completion':
        instructs = [
            f"Examine the image containing a filled black box, which represents the missing or obscured portion. Using the provided description, generate a complete version of the image that includes the content described by the text in the previously missing area. Make sure the generated content integrates seamlessly with the visible parts of the image, replacing the black box and forming a cohesive and complete image, without altering any existing visible areas.",
        ]
    elif task == 'image_completion_w_region_caption':
        instructs = [
            f"Examine the image with a filled black box representing the missing or obscured portion, along with the provided caption describing the content of the missing area. Based on the caption, generate the content for the missing region and integrate it seamlessly into the image, creating a complete version of the image with the missing area filled in. Ensure that the generated content reflects the caption accurately and integrates well with the visible parts of the image, without creating or modifying any existing visible areas.",
        ]
    elif task == 'image_completion_w_image_caption':
        instructs = [
            f"Examine the image with a missing area represented by a black box. Use the provided image caption to generate a complete version of the image, including the content described by the text. Ensure that the generated content integrates seamlessly with the visible parts of the image, creating a cohesive and complete image without modifying any existing visible areas.",
        ]
    elif task == 'VQA_activity_recognition':
        instructs = [
            f"""Examine the image and identify the activity being performed by the animals or people present in the image, based on the given question. Select the most appropriate answer from the provided options."""
        ]
    elif task == 'VQA_attribute':
        instructs = [
            f"""In this task, you will be asked a question about the attribute of an object in the image. Look at the image and answer the question by identifying the object and selecting its attribute from the given options. The question will ask about a specific attribute of the object, and you must choose the best answer from the options provided."""
        ]
    elif task == 'VQA_color':
        instructs = [
            f"""Given an image, answer the question about the color of a specific object in the image by selecting the best answer from the given options. The question will ask about the color of the object in the image, and you must identify the object first before selecting the correct color option."""
        ]

    elif task == 'VQA_counting':
        instructs = [
            f"""Examine the image and count the number of specific objects as asked in the given question. Your task is to select the correct answer from the given options based on your count."""
        ]

    elif task == 'VQA_object_presence':
        instructs = [
            f"""Given an image and a question asking about the presence of a specific object in the image, select the answer from the given options. The question will include a reference to the object of interest.""",
        ]

    elif task == 'VQA_object_recognition':
        instructs = [
            f"""Examine the image and answer a question about the type or subclass of an object in the image. Choose the best answer from the given options. """

        ]

    elif task == 'VQA_positional_reasoning':
        instructs = [
            f"""Examine the image and analyze the spatial relationships between objects. Based on this analysis, answer the given question about the position of objects within the image. Consider the relative locations of objects in the image and select the best answer from the given options."""
        ]

    elif task == 'VQA_scene_recognition':
        instructs = [
            f"""In this task, you are presented with an image depicting a certain environment or scene. Your goal is to understand the scene in the image and select the correct answer to the provided question, which is related to the overall environment or scene. You should carefully analyze the scene and choose the answer that best fits the provided question from the given options."""
        ]

    elif task == 'VQA_sentiment_understanding':
        instructs = [
            f"""Examine the image and interpret the sentiment depicted within it. Answer the provided question regarding the emotion conveyed in the image, and select the best answer from the given options."""
        ]

    elif task == 'VQA_sport_recognition':
        instructs = [
            f"""Examine the image and answer the question about the sport depicted in it. Choose the correct answer from the given options based on the sports that are taking place in the image."""
        ]

    elif task == 'VQA_utility_affordance':
        instructs = [
            f"""You will answer a question about the utility affordance of an object in the image. Utility affordance refers to the potential usefulness or practical value of an object for achieving a particular goal or fulfilling a specific need.""",
        ]

    elif task == 'select_overlap_most_region':
        instructs = [
            f"""Given the a region, you need to decide which region in the options overlaps most with given region. The region is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner. In order to find the region that overlaps most with the given region, you need to compute their overlapped area.""",
        ]
    elif task == 'select_overlap_least_region':
        instructs = [
            f"""Given the a region, you need to decide which region in the options overlaps least with given region. The region is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner. In order to find the region that overlaps least with the given region, you need to compute their overlapped area.""",
        ]
    elif task == 'region_area':
        instructs = [
            f"""You are given a bounding box and you need to find the area of the bounding box. The bounding box is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner. The region can be compute by the equation: (x2-x1) multiply (y2-y1).""",
        ]
    elif task == 'select_overlaped_region':
        instructs = [
            f"""Given the a region, you need to select a region in the options that overlaps with given region. The region is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner. Two regions are overlapped if their overlapped area is not zero.""",

        ]
    elif task == 'select_nonoverlaped_region':
        instructs = [
            f"""Given the a region, you need to select a region in the options that does not overlap with given region. The region is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner. Two regions are overlapped if their overlapped area is zero."""
        ]
    elif task == 'if_region_overlap':
        instructs = [
            f"""Given two regions, you need to decide if two regions are overlapped. Two regions are overlapped if their overlapped area is not zero."""
        ]
    # --------------- llava tasks -----------------
    elif task == 'llava_conversation':
        instructs = [
            f"""Given the history of a conversation between a User and an AI Assistant, generate the next response."""
        ]
    elif task == 'llava_detailed_caption':
        instructs = [
            f"""Generate a detailed caption of the image."""
        ]
    elif task == 'llava_complex_reasoning':
        instructs = [
            f"""Answer a complex question about the image. The question can be related to the background knowledge of the objects in the image, or about events happening in the image."""
        ]

    elif task == 'textcaps':
        instructs = [
            f"""Generate a caption involving the text in the image."""
        ]
    elif task == 'scienceqa_exp':
        instructs = [
            f"""given the question and the answer, generate an explanation."""
        ]
    elif task == 'aok_vqa':
        instructs = [
            f"""select the correct answer for the given question."""
        ]
    elif task == 'science_qa':
        instructs = [
            f"""select the correct answer for the given question. the question is about sicence."""
        ]
    elif task == 'visit':
        instructs = [
            f"""generate a detailed answer for the given question."""
        ]
    elif task == 'text_vqa':
        instructs = [
            f"""answer a question about some text in the image."""
        ]
    elif task == 'visual_spatial_reasoning':
        instructs = [
            f"""answer a question about the spatial relationship between objects in the image."""
        ]
    elif task == 'natural_language_visual_reasoning':
        instructs = [
            f"""answer a question about the image."""
        ]
    elif task == 'winoground':
        instructs = [
            f"""select the more accurate caption from two similar captions for the image."""
        ]
    elif task == 'medic_damage_severity':
        instructs = [
            f"""classify the damage severity in the image."""
        ]
    elif task == 'medic_informative':
        instructs = [
            f"""classify if the image is informative about a disaster."""
        ]
    elif task == 'medic_disaster_types':
        instructs = [
            f"""identify the type of disaster happenining in the image."""
        ]
    elif task == 'medic_humanitarian':
        instructs = [
            f"""decide the humanitarian of the image."""
        ]
    elif task == 'aokvqa_rational':
        instructs = [
            f"""given a questions and an answer, explain why it is the answer to the question."""
        ]
    elif task == 'cifar_10':
        instructs = [
            f"""classify the main object in the image."""
        ]
    elif task == 'cifar_100':
        instructs = [
            f"""classify the main object in the image."""
        ]
    elif task == 'miniImagenet':
        instructs = [
            f"""classify the main object in the image."""
        ]
    elif task == 'mnist':
        instructs = [
            f"""classify the number in the image."""
        ]
    elif task == 'pope_adv':
        instructs = [
            f"""decide if the mentioned object is in the image."""
        ]
    elif task == 'pope_pop':
        instructs = [
            f"""decide if the mentioned object is in the image."""
        ]
    elif task == 'pope_rand':
        instructs = [
            f"""decide if the mentioned object is in the image."""
        ]
    elif task == 'snli_ve_answer_choice':
        instructs = [
            f"""select the caption fits the sentence best."""
        ]
    elif task == 'snli_ve_classification':
        instructs = [
            f"""decide if the content of the image support the sentence."""
        ]
    elif task in VISION_FLAN_DEF:
        instruct = VISION_FLAN_DEF.get(task, 'Answer the following question based on the image.')
        instructs = [
            instruct
        ]

    elif task == 'object_localization.jsonl':
        instructs = [
            f"""Answer a question about the location of an object in the image."""
        ]
    elif task == 'image_style.jsonl':
        instructs = [
            f"""Identify the art style of this image."""
        ]
    elif task == 'celebrity_recognition.jsonl':
        instructs = [
            f"""recognize the celebrity in the image."""
        ]
    elif task == 'physical_property_reasoning.jsonl':
        instructs = [
            f"""Read a passage and answer a question about the physical property of an object in the image."""
        ]
    elif task == 'image_quality.jsonl':
        instructs = [
            f"""select the image with the quality mentioned in the text."""
        ]
    elif task == 'function_reasoning.jsonl':
        instructs = [
            f"""identify the function of the demonstrated object."""
        ]
    elif task == 'attribute_comparison.jsonl':
        instructs = [
            f"""Comparing attributes of two objects."""
        ]
    elif task == 'nature_relation.jsonl':
        instructs = [
            f"""Decide the nature relations of these animals or humans in the image."""
        ]
    elif task == 'identity_reasoning.jsonl':
        instructs = [
            f"""Identify the best options based on the image and input text."""
        ]
    elif task == 'image_emotion.jsonl':
        instructs = [
            f"""Identify the emotion in the image."""
        ]
    elif task == 'image_topic.jsonl':
        instructs = [
            f"""select the best caption describing the image."""
        ]
    elif task == 'future_prediction.jsonl':
        instructs = [
            f"""predict a future event based on the image."""
        ]
    elif task == 'ocr.jsonl':
        instructs = [
            f"""Answer a question about the text in the image."""
        ]
    elif task == 'structuralized_imagetext_understanding.jsonl':
        instructs = [
            f"""Answer a question about a chart of a table which has structured text on it."""
        ]
    elif task == 'physical_relation.jsonl':
        instructs = [
            f"""answer a question about the physical relationship between objects in the image."""
        ]
    elif task == 'image_scene.jsonl':
        instructs = [
            f"""select the best caption describing the image."""
        ]
    elif task == 'attribute_recognition.jsonl':
        instructs = [
            f"""recognize the attributes of an object in the image."""
        ]
    elif task == 'spatial_relationship.jsonl':
        instructs = [
            f"""answer a question about the saptial relationship in the image."""
        ]
    elif task == 'social_relation.jsonl':
        instructs = [
            f"""decide the social relationship between the two persons in this image."""
        ]
    elif task == 'action_recognition.jsonl':
        instructs = [
            f"""Decide what kind of human behavior does this picture describe."""
        ]
    elif task == 'mm_vet.jsonl':
        instructs = [
            f"""Perform the task based on the instruction. Some of the questions can be answered with short phrases and some other more open-ended questions require you to generate detailed respones."""
        ]

    elif task == 'landmark.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about the landmark in the image."""
        ]
    elif task == 'text_translation.jsonl':
        instructs = [
            f"""Decide if the translation of some text in the image is correct or not."""
        ]
    elif task == 'color.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about the color of an object in the image."""
        ]
    elif task == 'celebrity.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about the celebrity in the image."""
        ]
    elif task == 'scene.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about the scene in the image."""
        ]
    elif task == 'numerical_calculation.jsonl':
        instructs = [
            f"""Decide if the answer to a arithmetic question shown in the image is correct or not."""
        ]
    elif task == 'commonsense_reasoning.jsonl':
        instructs = [
            f"""Answer a yes-or-no question related to commonsense reasoning about the image."""
        ]
    elif task == 'code_reasoning.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about some computer code in the image."""
        ]
    elif task == 'count.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about counting objects in the image."""
        ]
    elif task == 'OCR.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about the text in the image."""
        ]
    elif task == 'existence.jsonl':
        instructs = [
            f"""Decide the presence of an object in the image."""
        ]
    elif task == 'artwork.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about an artwork."""
        ]
    elif task == 'posters.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about a poster."""
        ]
    elif task == 'position.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about the position of an object in the image."""
        ]


    # elif task == 'Scene_Understanding':
    # elif task == 'Visual_Reasoning':
    # elif task == 'Instances_Counting':
    # elif task == 'Instance_Interaction':
    # elif task == 'Instance_Attributes':
    # elif task == 'Text_Understanding':
    # elif task == 'Instance_Identity':
    # elif task == 'Instance_Location':
    # elif task == 'Spatial_Relation':

    else:
        instructs = [
            f"""None"""
        ]
        print(f'warning: {task} does not have a valid definition. plz write it!!!!')

    return instructs[0]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(args.model_path)

    answers_dir = Path(args.model_path) / 'results' if not args.dist_token else Path(
        args.model_path) / 'dist_token_results'
    if args.full:
        answers_dir = answers_dir / "full"
    answers_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,PT_len=args.PT_len,VIT_PT_len=args.VIT_PT_len)
    if args.VIT_PT_len > 0:
        modules = torch.load(os.path.join(model_path, "pytorch_model-00002-of-00002.bin"))
        vit_prompts = torch.nn.ParameterList([modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.0"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.1"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.2"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.3"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.4"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.5"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.6"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.7"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.8"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.9"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.10"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.11"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.12"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.13"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.14"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.15"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.16"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.17"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.18"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.19"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.20"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.21"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.22"],
                                           modules["model.vision_tower.vision_tower.vision_model.encoder.prompts.23"]])
        # model.vision_tower.vision_tower.vision_model.encoder.prompts = vit_prompts
        model.get_model().vision_tower.vision_tower.vision_model.encoder.prompts = vit_prompts

    file_list = EVAL_TASKS
    # not all 7 tasks will be computed once,
    # file_list = file_list[3:]
    #print("file_list: ", args.chunk_idx*3, (args.chunk_idx+1)*3)
    question_dir = args.question_dir
    for question_file in file_list:
        task_def = get_task_def(question_file)
        args.question_file = question_file
        answers_file = Path(answers_dir) / (args.question_file + '.jsonl')
        args.question_file = os.path.join(question_dir, args.question_file)
        try:
            questions = [json.loads(q) for q in open(
                f"{args.question_file}.jsonl", "r")]
        except:
            print(
                f'fail to open {args.question_file}.jsonl . makes sure it exist')
            continue
        if os.path.exists(answers_file):
            try:
                answers = [json.loads(q) for q in open(f"{answers_file}", "r")]
                if len(answers) == len(questions):
                    print(
                        f"Predictions at {answers_file} exists and has the same length as number of questions. skip it.")
                    continue
            except:
                print(f'regenerate predictions at {answers_file}')
            # spdb.set_trace()
        print(f"Testing {args.question_file}.jsonl")
        print(f"Save predictions at {answers_file}")

        print(f'Totally {len(questions)} testing instances')

        print(
            f"a sample instance look like this:\n\n{questions[0]['prompt']}\n\nAnswer: {questions[0]['target']}")
        print(
            f"\nIt's image is at {os.path.join(args.image_folder, questions[0]['image_path'])}")
        # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")
        if args.in_context:
            in_context_ex = random.choice(questions)
            in_context_q = in_context_ex['prompt']
            in_context_a = in_context_ex['target']

        total_gen_tokens = 0
        for line in tqdm(questions):
            idx = line["unique_id"]

            image_file = os.path.join(args.image_folder, line["image_path"])
            assert os.path.exists(image_file)
            image = Image.open(image_file)
            qs = line["prompt"]
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                    DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.in_context:
                qs = f"[Example]: [Input]: {in_context_q} [Output]: {in_context_a}||||{qs}"

            conv = conv_templates[args.conv_mode].copy()
            if args.dist_token:
                # pdb.set_trace()
                if question_file in DIST_TOKEN:
                    # use distribution token
                    qs = f"{qs}\n{DIST_TOKEN[question_file]}"
                    # pdb.set_trace()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt().replace('[Options]', 'Options')

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_tensor = image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(
                keywords, tokenizer, input_ids)

            image_tensor = image_tensor.unsqueeze(0).half().cuda()

            task_def_id = None
            if getattr(model.config, "cond_type", None):
                if "task_def" in model.config.cond_type:
                    task_def_id = tokenizer_image_token(
                        task_def, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    if input_ids.shape[1] > task_def_id.shape[1]:
                        task_def_id = F.pad(
                            task_def_id, (0, input_ids.shape[1] - task_def_id.shape[1]), "constant", tokenizer.pad_token_id)
                    else:
                        input_ids = F.pad(
                            input_ids, (0, task_def_id.shape[1] - input_ids.shape[1]), "constant", tokenizer.pad_token_id)
                    input_ids = torch.cat([input_ids, task_def_id], dim=0)
                elif getattr(model.config, "mix_mm_projector", False) or model.config.cond_type == 'task_embed':
                    task_def_id = tokenizer_image_token(
                        task_def, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    task_def_ids=task_def_id,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=50,
                    use_cache=True)

            if getattr(model.config, "cond_type", None):
                if "task_def" in model.config.cond_type:
                    output_ids, _ = torch.chunk(output_ids, 2, dim=0)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(
                    f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            total_gen_tokens += output_ids[:, input_token_len:].shape[1]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": prompt,
                                       "predict": outputs,
                                       "target": line['target'],
                                       "image_path": image_file,
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
            # pdb.set_trace()
            ans_file.flush()
        ans_file.close()
        print(f"Total generated tokens: {total_gen_tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--PT_len", type=int, default=1)
    parser.add_argument("--VIT_PT_len", type=int, default=1)
    parser.add_argument("--image-folder", type=str,
                        default="/path/to/mixlora_data/eval_images/full_data")
    parser.add_argument("--question-file", type=str,
                        default="tables/question.jsonl")
    parser.add_argument("--question-dir", type=str,
                        default="/path/to/mixlora_data/eval_images/full_data")
    parser.add_argument("--eval-file", type=str, default="answer.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--in-context", action='store_true', default=False)
    parser.add_argument("--training-set", action='store_true', default=False)
    parser.add_argument("--dist-token", action='store_true', default=False)
    parser.add_argument("--full", action='store_true', default=False)
    args = parser.parse_args()

    eval_model(args)
