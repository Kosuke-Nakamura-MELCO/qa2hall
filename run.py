"""QA2HALL: A Framework for Generating Non-trivial Hallucination Detection Datasets from KGQA Datasets

    This source code is relased under CC BY-NC-SA 4.0 License.
    (https://creativecommons.org/licenses/by-nc-sa/4.0/)
    
    by Kosuke Nakamura
"""

import os
import json
import argparse

from transformers import pipeline

from qa2hall import QAPromptTemplate, DecSentTemplate, LLMsDefinition
from qa2hall import IncorrectAnswerFilter, EntExistFilter, CorrectAnsInDecFilter
from qa2hall import generate_answers, convert_question_into_dec_sent, get_examples
from utils import load_kqapro_entities, dict_list_to_tsv, tsv_to_dict_list

## set proxies if needed
# os.environ['HTTP_PROXY'] = ''
# os.environ['HTTPS_PROXY'] = ''

## set visible devices if needed
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

## CONSTANTS
# output file names
INCORRECT_ANS_TSV_NAME = "incorrect_answers.tsv"
ALL_ANS_JSON_NAME = "all_incorrect_answers.json"
DEC_SENT_TSV_NAME = "declarative_sentences.tsv"
ALL_DEC_SENT_JSON_NAME = "all_declarative_sentences.json"
DATASET_FNAME = "dataset.json"
# quality threshold
ANS_QUALITY_THRESH = 0
DEC_SENT_QUALITY_THRESH = 0

# ====================================================================
# Step 1 ~ 3
# ====================================================================
def step_1_3(args):

    ## Load KGQA dataset
    print("Load KGQA dataset: " + args.qa_data_path)
    with open(args.qa_data_path, "r") as f:
        qa_data = json.load(f)

    ## Load model
    print("Load model: " + args.model_name)
    # Choose LLM to use
    model_def = LLMsDefinition(args.model_name)
    model_id = model_def.get_model_id()
    model_kwargs = model_def.get_model_kwargs()        

    # load model as pipeline
    pipe = pipeline("text-generation",
                    model=model_id, 
                    model_kwargs=model_kwargs,
                    device_map=args.device_map
                    )


    # ====================================================================
    # Step 1. Collect Incorrect Answers
    # ====================================================================

    print("")
    print("="*40)
    print("Step 1. Collect Incorrect Answers")
    print("="*40)

    ## Prepare template
    qa_template = QAPromptTemplate(args.qa_template_id)

    ## Get examples for prompt
    examples, qa_data = get_examples(qa_data, num=args.num_examples, shuffle=True, seed=0)

    ## Generate answers
    answered_samples = generate_answers(qa_data, pipe, qa_template, examples=examples)

    if args.save_all_answers:
        with open(os.path.join(args.out_dir, ALL_ANS_JSON_NAME), "w", encoding="utf-8") as f:
            json.dump(answered_samples, f, indent=4)

    ## Extract reasonable INCORRECT answers
    ans_filter = IncorrectAnswerFilter(ignore_capitalize=True, do_strip=True, 
                                       length_thresh=args.ans_len_thresh)
    answered_samples = [s for s in answered_samples if ans_filter.filter(s)]

    ## Extract incorrect answers which exist in the KG
    entities = load_kqapro_entities(args.kg_path)
    ent_filter = EntExistFilter(entities, ignore_capitalize=True, do_strip=True)
    answered_samples = [s for s in answered_samples if ent_filter.filter(s)]

    # ====================================================================
    # Step 2. Convert questions into declarative sentences
    # ====================================================================

    print("")
    print("="*40)
    print("Step 2. Convert questions into declarative sentences")
    print("="*40)

    ## prepate template
    dec_template = DecSentTemplate(args.dec_template_id)

    ## Convert questions into declarative sentences
    converted_samples = convert_question_into_dec_sent(qa_data, pipe, dec_template)
    
    if args.save_all_dec_sents:
        with open(os.path.join(args.out_dir, ALL_DEC_SENT_JSON_NAME), "w", encoding="utf-8") as f:
            json.dump(converted_samples, f, indent=4)

    ## Extract samples where the generated dec sent contains original correct answers as string
    ans_in_dec_filter = CorrectAnsInDecFilter(ignore_capitalize=True, do_strip=True)
    converted_samples = [s for s in converted_samples if ans_in_dec_filter.filter(s)]

    # ====================================================================
    # Step 3. Manual Filtering
    # ====================================================================

    ## generate TSV files for manual filtering using spread sheet software
    # incorrect answers
    ans_for_tsv = [{
            "id":sample["id"],
            "query":sample["query"],
            "gt_ans":sample["gt_answer"], 
            "gen_ans":sample["gen_answer"],
            "quality":""
            } for sample in answered_samples
        ]

    if ans_for_tsv:
        dict_list_to_tsv(ans_for_tsv, os.path.join(args.out_dir, INCORRECT_ANS_TSV_NAME))
        print("save generated answers: " + os.path.join(args.out_dir, INCORRECT_ANS_TSV_NAME))
    else:
        print("No generated answers preserved")

    # declarative sentences
    dec_sent_for_tsv = [{
            "id":sample["id"],
            "query":sample["query"],
            "gt_ans":sample["gt_answer"],
            "declarative":sample["declarative"],
            "quality":""
            } for sample in converted_samples
        ]
    
    if dec_sent_for_tsv:
        dict_list_to_tsv(dec_sent_for_tsv, os.path.join(args.out_dir, DEC_SENT_TSV_NAME))
        print("save generated declarative sentences: " + os.path.join(args.out_dir, DEC_SENT_TSV_NAME))
    else:
        print("No generated declarative sentences preserved")


# ====================================================================
# Step 4. Replace Correct Answers with Incorrect Answers
# ====================================================================
def step_4(args):

    ## Load manually checked incorrect answers and declarative sentences
    incrct_ans = tsv_to_dict_list(args.ans_path)
    dec_sents  = tsv_to_dict_list(args.dec_sent_path)

    ## IDs of samples which has accepted incorrect answers
    id2ans = {s["id"]:s for s in incrct_ans if int(s["quality"]) > ANS_QUALITY_THRESH}
    
    ## list of dict to output
    output_json = []

    ## Replace correct answers in declarative sentences with incorrect answers
    for sample in dec_sents:

        id = sample["id"]

        if int(sample["quality"]) <= DEC_SENT_QUALITY_THRESH:
            continue

        # use sample with incorrect ans as hallucinated sample
        if id in id2ans:
            hallucinated = 1
            gen_ans = id2ans[id]["gen_ans"]
            dec_sent = sample["declarative"].replace(sample["gt_ans"], gen_ans)
        # use sample without incorrect ans as not-hallucinated sample
        else:
            hallucinated = 0
            gen_ans = ""
            dec_sent = sample["declarative"]

        output_json.append({            
            "id": id,
            "hallucinated": hallucinated, 
            "query": sample["query"],
            "declarative": dec_sent,
            "gt_ans": sample["gt_ans"],
            "gen_ans": gen_ans
        })
        
    ## Save json
    with open(os.path.join(args.out_dir, DATASET_FNAME), "w", encoding='utf-8') as f:
        json.dump(output_json, f, indent=4)

# ====================================================================
# MAIN
# ====================================================================
def main():
    
    ## Parse arguments        
    parser = argparse.ArgumentParser(description='QA2HALL framework')

    # Main args
    parser.add_argument('-q', '--qa_data_path', type=str, default="",
                        help="path to QA dataset file (json). required for step 1 and 2")
    parser.add_argument('-m', '--model_name', type=str, default="",
                        help="name of LLM to use. See LLMsDefinition in qa2hall.py. required for step 1 and 2")
    parser.add_argument('--ans_path', type=str, default="",
                        help="path to filtered incorret answer file (). required for step 4")
    parser.add_argument('--dec_sent_path', type=str, default="",
                        help="path to filtered declarative sentence file(). required for step 4")
    parser.add_argument('--kg_path', type=str, default="",
                        help="path to KG file (json). required for step 1")
    parser.add_argument('-o', "--out_dir", type=str, required=True,
                        help="output directory path")
    # Steps to run
    parser.add_argument('--replace_step', action='store_true', default=False,
                        help="Run Step 4. Replace correct answers with incorrect answers")
    # Parameters
    parser.add_argument('--qa_template_id', type=int, default=0,
                        help="id for prompt template to use for question answering") 
    parser.add_argument('--dec_template_id', type=int, default=0,
                        help="id for prompt template to use for converting questions into declarative sentences")
    parser.add_argument('--num_examples', type=int, default=5, 
                        help="number of example QA pairs to input LLM")
    parser.add_argument('--ans_len_thresh', type=int, default=50, 
                        help="answers with length than this will be filtered out")
    # Other options for processing
    parser.add_argument('--save_all_answers', action="store_true", default=False,
                        help="save all generated answers regardless of their quality as extra json file")
    parser.add_argument('--save_all_dec_sents', action="store_true", default=False,
                        help="save all generated declarative sentences regardless of their quality as extra json file")
    parser.add_argument('--device_map', type=str, default='auto')
 
    args = parser.parse_args()
    
    ## Run steps
    if not args.replace_step:
        step_1_3(args)

    else:
        step_4(args)

if __name__ == "__main__":
    main()