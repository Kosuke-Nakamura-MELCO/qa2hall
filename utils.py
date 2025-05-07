# ====================================================================
# IMPORT
# ====================================================================
import json
import csv

# ===================================================================
# load_kqapro_entities
# ====================================================================
def load_kqapro_entities(path:str) -> list:

    with open(path, "r") as f:
        kqa_pro_kb = json.load(f)

    entities = kqa_pro_kb["entities"]
    concepts = kqa_pro_kb["concepts"]

    entity_names  = [val["name"] for val in entities.values()]
    concept_names = [val["name"] for val in concepts.values()]

    return entity_names + concept_names

# ===================================================================
# dict_list_to_tsv
# ====================================================================
def dict_list_to_tsv(dict_list, file_path, delimiter='\t'):
    
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_STRINGS)
        
        keys = dict_list[0].keys()
        writer.writerow(keys)
        
        for dictionary in dict_list:
            writer.writerow(dictionary[key] for key in keys)

# ===================================================================
# tsv_to_dict_list
# ====================================================================
def tsv_to_dict_list(file_path, delimiter='\t'):
    
    dict_list = []
    
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=delimiter, quotechar='"')
        
        for row in reader:
            dict_list.append(dict(row))
    
    return dict_list