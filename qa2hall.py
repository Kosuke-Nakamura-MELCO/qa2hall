"""QA2HALL: A Framework for Generating Non-trivial Hallucination Detection Datasets from KGQA Datasets

    This source code is relased under CC-BY NC SA 4.0 License.
    (https://creativecommons.org/licenses/by-nc-sa/4.0/)
    
    by Kosuke Nakamura
"""

import random
import unicodedata

import tqdm
import torch

# ====================================================================
# LLMsDefinition
# ====================================================================
class LLMsDefinition():
    """ Class for LLMs args

        modify this class to add models to use or to change args for each model
    """

    MODELS = {
        "vicuna_13B_v1.5":{
            "model_id":"lmsys/vicuna-13b-v1.5",
            "model_kwargs":{}
        },
        "llama31_8B_inst":{
            "model_id":"meta-llama/Meta-Llama-3.1-8B-Instruct",
            "model_kwargs":{"torch_dtype": torch.bfloat16},
        },     
        "gemma2_9B_inst":{
            "model_id":"google/gemma-2-9b-it",
            "model_kwargs":{"torch_dtype": torch.bfloat16},
        }
    }

    def __init__(self, model_name:str):
        
        self.model_name = model_name

    def get_model_id(self)->str:
        return self.MODELS[self.model_name]["model_id"]
    
    def get_model_kwargs(self)->dict:
        return self.MODELS[self.model_name]["model_kwargs"]

# ====================================================================
# QAPromptTemplate
# ====================================================================
class QAPromptTemplate():

    PROMPT_WITH_EXAMPLE_0 = "\n".join([
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        "Provide context and examples to the model.", 
        "The output should follow the exact format as the example output, which does not contain extra information about the answer."
        "Return only one most plausible answer."
    ])
    PROMPT_WITH_EXAMPLE_1 = "\n".join([
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        "Provide context and examples to the model.", 
        "The output should follow the exact format as the example output."
    ])
    PROMPT_WITH_EXAMPLE_2 = "\n".join([
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        "Provide context and examples to the model.", 
        "The output should follow the exact format, which contains only a noun as the most plausible answer."
    ])
    PROMPT_WITH_EXAMPLE_3 = "\n".join([
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives answers to the user's questions.",
        "Provide context and examples to the model.", 
        "The output should follow the exact format as the example output, which does not contain extra information about the answer.",
        "Return only one most plausible answer."
    ])

    PROMPT_WITH_EXAMPLE_LIST = [
        PROMPT_WITH_EXAMPLE_0, PROMPT_WITH_EXAMPLE_1, PROMPT_WITH_EXAMPLE_2, PROMPT_WITH_EXAMPLE_3
    ]

    def __init__(self, template_id=0) -> None:

        self._template_id = 0 if template_id + 1 > len(self.PROMPT_WITH_EXAMPLE_LIST) else template_id

        self._template = self.PROMPT_WITH_EXAMPLE_LIST[self._template_id]

        self._example_str = ""
        self._content = ""


    def set_example(self, examples:list) -> None:
            
        exm_str = ""
        
        for e in examples:
            exm_str = exm_str + "USER:" + e["question"] + "\n" + "ASSISTANT:" + e["answer"] + "\n"
            
        self._example_str = exm_str

    def set_content(self, content:str) -> None:

        self._content = "USER:" + content

    def get_prompt(self) -> str:

        return  self._template + "\n\n" + self._example_str + "\n" + self._content + "\n" + "ASSISTANT:"

# ====================================================================
# DecSentTemplate
# ====================================================================
class DecSentTemplate():

    TEMPLATES = [
        "\n".join(["Rewrite the following question sentence into declarative sentence by using the given answer of it.",
                    "# Question sentence",
                    "{question}",
                    "# Answer",
                    "{answer}",
                    "# Output"
        ])               
    ]

    def __init__(self, template_id=0) -> None:
    
        self.template_id = template_id
        self.template = self.TEMPLATES[self.template_id]
        self.question = ""
        self.answer   = ""

    def set_content(self, question:str, answer:str):
        
        self.question = question
        self.answer   = answer

    def get_prompt(self) -> str:

        return self.template.format(question = self.question, answer=self.answer) + "\n"

# ====================================================================
# IncorrectAnswerFilter
# ====================================================================
class IncorrectAnswerFilter():

    def __init__(self, ignore_capitalize=True, do_strip=True, length_thresh=50, normalize_type="NFKC"):
        
        self.ignore_capitalize = ignore_capitalize
        self.do_strip          = do_strip
        self.length_thresh     = length_thresh
        self.normalize_type    = normalize_type

    def filter(self, answered_sample:dict) -> bool:
        """ Determine wether given sample's generated answer is acceptable as inccorect answer
        
        """

        if self.do_strip:
            gt_answer = answered_sample["gt_answer"].strip()
            gen_answer = answered_sample["gen_answer"].strip()
        else:
            gt_answer = answered_sample["gt_answer"]
            gen_answer = answered_sample["gen_answer"]
        
        if self.normalize_type != "":
            gt_answer = unicodedata.normalize(self.normalize_type, gt_answer)
            gen_answer = unicodedata.normalize(self.normalize_type, gen_answer)

        if len(gen_answer) > self.length_thresh:
            return False

        if self.ignore_capitalize:
            gt_answer = gt_answer.lower()
            gen_answer = gen_answer.lower()
        
        if gt_answer == gen_answer:
            return False
        
        return True
    
# ====================================================================
# EntExistFilter
# ====================================================================
class EntExistFilter():
    
    def __init__(self, entities:list, ignore_capitalize=True, do_strip=True, normalize_type="NFKC"):

        self.entities = entities        
        self.normalized_entities = set(self.entities)
        self.ignore_capitalize = ignore_capitalize
        self.do_strip          = do_strip
        self.normalize_type    = normalize_type

        self.normalize_entities()

    def normalize_entities(self):

        self.normalized_entities = set()
        for ent in self.entities:

            norm_ent = ent
            if self.do_strip:
                norm_ent = norm_ent.strip()
            
            if self.normalize_type != "":
                norm_ent = unicodedata.normalize(self.normalize_type, norm_ent)

            if self.ignore_capitalize:
                norm_ent = norm_ent.lower()

            self.normalized_entities.add(norm_ent)

    def filter(self, answered_sample:dict) -> bool:
        """ Check whether given sample's generated answer exist in the KG
        
        """
        gen_ans = answered_sample["gen_answer"]
        if self.do_strip:
            gen_ans = gen_ans.strip()
        
        if self.normalize_type != "":
            gen_ans = unicodedata.normalize(self.normalize_type, gen_ans)

        if self.ignore_capitalize:
            gen_ans = gen_ans.lower()

        if gen_ans in self.normalized_entities:
            return True
        
        return False


# ====================================================================
# CorrectAnsInDecFilter
# ====================================================================
class CorrectAnsInDecFilter():
    
    def __init__(self, ignore_capitalize=True, do_strip=True, normalize_type="NFKC"):

        self.ignore_capitalize = ignore_capitalize
        self.do_strip          = do_strip
        self.normalize_type    = normalize_type

    def filter(self, converted_sample:dict) -> bool:
        """ Check whether given sample's generated declarative sentence contains original correct answer as str

        """

        gt_answer = converted_sample["gt_answer"]
        dec_sent = converted_sample["declarative"]

        if self.do_strip:
            gt_answer = gt_answer.strip()
            dec_sent = dec_sent.strip()
        
        if self.normalize_type != "":
            gt_answer = unicodedata.normalize(self.normalize_type, gt_answer)
            dec_sent = unicodedata.normalize(self.normalize_type, dec_sent)

        if self.ignore_capitalize:
            gt_answer = gt_answer.lower()
            dec_sent  = dec_sent.lower()
        
        if gt_answer in dec_sent:
            return True
        
        return False
    


# ====================================================================
# get_example
# ====================================================================
def get_examples(qa_data:list, num:int, shuffle=True, seed=0) -> tuple[list, list]:

    if num >= len(qa_data):
        num = len(qa_data) - 1

    qa_data_copy = qa_data[:]
    examples = []

    if shuffle:
        random.seed(seed)
        targets = sorted(random.sample(range(len(qa_data)), k=num), reverse=True)
    else:
        targets = sorted(list(range(num)), reverse=True)

    for ind in targets:

        examples.append(qa_data_copy.pop(ind))

    return examples, qa_data_copy

# ====================================================================
# generate_answers
# ====================================================================
def generate_answers(qa_data:list, pipe, template:QAPromptTemplate, examples=[], max_length=512):

    answered_samples = []

    for trg in tqdm.tqdm(qa_data):
                
        query = trg["question"]

        template.set_content(query)
        template.set_example(examples)
        input_txt = template.get_prompt()

        # generate answers
        output = pipe(input_txt, max_length=max_length) 

        gen_text = output[0]["generated_text"]
        # strip
        gen_answer = gen_text.replace(input_txt, "").strip()

        answered_samples.append({
            "id":trg["id"],
            "query":query,
            "input":input_txt,
            "gt_answer":trg["answer"],
            "output":output,
            "gen_text":gen_text,
            "gen_answer":gen_answer
        })

    return answered_samples

# ====================================================================
# convert_question_into_dec_sent
# ====================================================================
def convert_question_into_dec_sent(qa_data:list, pipe, template:DecSentTemplate, max_length=512):
    
    dec_sentences = []

    for trg in tqdm.tqdm(qa_data):
    
        template.set_content(question=trg["question"], answer=trg["answer"])
        input_txt = template.get_prompt()

        output = pipe(input_txt, max_length=max_length)

        gen_text = output[0]["generated_text"]
        dec_sent = gen_text.replace(input_txt, "").strip()

        dec_sentences.append({
            "id":trg["id"],
            "query":trg["question"],
            "input":input_txt,
            "output":output,
            "gen_text":gen_text,
            "declarative":dec_sent,
            "gt_answer":trg["answer"]
        })

    return dec_sentences

