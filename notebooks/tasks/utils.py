import hashlib
import importlib
from IPython.display import clear_output, display, HTML
import json
import logging
import os
import polars as pl
import requests
import regex
import subprocess
import sys
from termcolor import colored
import time
from typing import List, Dict, Any, Tuple, Optional
try:
    from google.colab import files
except:
    pass


CHAR_PACKAGE = "📦"
CHAR_SUCCESS = "✅"
CHAR_FAILURE = "❌"
COLORS = {"PERSON": "red", "LOCATION": "green", "OTHER": "blue"}
LOCATION_ALTERNATIVES = ["PLACE"]


def safe_import(package_name):
    """Import a package;. If it missing, download it first"""
    try:
        return importlib.import_module(package_name)
    except ImportError:
        print(f"{CHAR_PACKAGE} {package_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Finished installing {package_name}")
        return importlib.import_module(package_name)


dotenv = safe_import("dotenv")
langid = safe_import("langid")
openai = safe_import("openai")
regex = safe_import("regex")
tabulate = safe_import("tabulate")


def squeal(text=None):
    """Clear the output buffer of the current cell and print the given text"""
    clear_output(wait=True)
    if not text is None: 
        print(text)


def set_css():
    """Fix line wrapping of output texts for Google Colab"""
    display(HTML("<style> pre { white-space: pre-wrap; </style>"))


def check_google_colab():
    """Check if the notebook is running on Google COlab and return a flag indicating the result of the check"""
    try:
        from google.colab import files
        get_ipython().events.register('pre_run_cell', set_css)
        return True
    except:
        return False


def mark_entities_in_text(texts_input, entities, linking_model=""):
    """Convert the text to HTML with colored antities and return these"""
    for entity in reversed(entities):
        entity["label"] = "LOCATION" if entity["label"] in LOCATION_ALTERNATIVES else entity["label"]
        entity_label = entity["label"] if entity["label"] in COLORS.keys() else "OTHER"
        if "wikidata_id" in entity:
            label_text = entity['wikidata_id'][list(entity['wikidata_id'].keys())[0]]
            if "link" in entity and linking_model in entity["link"]:
                label_text += "," + regex.sub(r"^(\d+).*$", r"\1", entity["link"][linking_model])
            texts_input = texts_input[:entity["end_char"]] + f"<sup>{label_text}</sup>" + texts_input[entity["end_char"]:]
        texts_input = texts_input[:entity["end_char"]] + "</span>" + texts_input[entity["end_char"]:]
        texts_input = (texts_input[:entity["start_char"]] + 
                      f"<span style=\"border: 1px solid black; color: {COLORS[entity_label]};\">" + 
                      texts_input[entity["start_char"]:])
    return texts_input


def save_data_to_json_file(json_data, file_name, in_colab):
    """Save json data in a json file; download it when working in Google Colab"""
    json_string = json.dumps(json_data, ensure_ascii=False, indent=2)
    hash = hashlib.sha1(json_string.encode("utf-8")).hexdigest()
    file_name = regex.sub(".json$", f"_{hash}.json", file_name)
    with open(file_name, "w", encoding="utf-8") as output_file:
        print(json_string, end="", file=output_file)
        output_file.close()
        print(f"️{CHAR_SUCCESS} Saved data to file {file_name}")
    if in_colab:
        try:
            files.download(file_name)
            print(f"️{CHAR_SUCCESS} Downloaded data file {file_name}")
        except:
            print(f"️{CHAR_FAILURE} Downloading data file {file_name} failed!")


def extract_entities_from_ner_input(texts_input):
    """For each entity in the input text return the entity text, context text and context text id""" 
    return [{"entity_text": entity["text"],
             "text_id": index,
             "text": text["text_cleaned"]} 
            for index, text in enumerate(texts_input) for entity in text["entities"]]


def detect_text_language(text: str) -> Dict[str, Any]:
    """Detect language text is written in and return id of the language"""
    return langid.classify(text)[0]
    
    
def read_json_file(file_name):
    """Read a json file and return its contents"""
    with open(file_name, "r") as infile:
        json_data = json.load(infile)
        infile.close()
    return json_data


def write_json_file(file_name, json_data):
    """Write json data to a file with the specified name"""
    json_string = json.dumps(json_data, ensure_ascii=False, indent=2)
    with open(file_name, "w") as outfile:
        print(json_string, end="", file=outfile)
        outfile.close()


def get_openai_api_key():
    """Extract OpenAI API key from environment or file and return it"""
    dotenv.load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        try:
            with open("OPENAI_API_KEY", "r") as infile:
                openai_api_key = infile.read().strip()
                infile.close()
        except:
            pass
    if not openai_api_key:
        try:
            from google.colab import userdata
            openai_api_key = userdata.get('OPENAI_API_KEY')
        except:
            pass
    if not openai_api_key:
        raise(Exception(f"{CHAR_FAILURE} no openai_api_key found!"))
    return openai_api_key


def connect_to_openai(openai_api_key):
    """Connect to OpenAI and return processing space"""
    return openai.OpenAI(api_key=openai_api_key)


def process_text_with_gpt(openai_client, model, prompt):
    """Send text to OpenAI via prompt and return results"""
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def save_entities_as_table(file_name, texts_output):
    entities_table = []
    for text_id, text in enumerate(texts_output):
        for entity in text["entities"]:
            entities_table.append({key: entity[key] for key in ["text", "label", "start_char", "end_char"]})
            entities_table[-1]["text_id"] = text_id
            if "wikidata_id" in entity:
                entities_table[-1]["wikidata_id"] = entity["wikidata_id"][list(entity["wikidata_id"].keys())[0]]
            if "link" in entity:
                entities_table[-1]["link"] = entity["link"][list(entity["link"].keys())[0]]
    pl.DataFrame(entities_table).write_csv(file_name)
    print(f"️{CHAR_SUCCESS} Saved data to file {file_name}")


def has_gpu() -> bool:
    """check if there is a gpu available, otherwise runs will take a lot of time"""
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=True)
        return True
    except:
        return False


def install_ollama():
    """install Ollama, start it as a server and check if it is running"""
    print(f"{CHAR_PACKAGE} Installing ollama")
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    os.environ["OLLAMA_MODELS"] = "/content/.ollama/models"
    server = subprocess.Popen(["ollama", "serve"], env=os.environ.copy())
    for _ in range(60):
        try:
            requests.get("http://127.0.0.1:11434/api/tags", timeout=1)
            print(f"{CHAR_SUCCESS} Ollama server is up")
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError(f"{CHAR_FAILURE} Ollama server did not start")
    time.sleep(3)


def import_ollama_module():
    """import Ollama module in Python"""
    try:
        if not has_gpu():
            print(f"{CHAR_FAILURE} Warning: no GPU found! On Colab you may want to switch Runtime to: T4 GPU")
        return importlib.import_module("ollama")
    except Exception as e:
        install_ollama()
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
        importlib.invalidate_caches()
        if not has_gpu():
            print(f"{CHAR_FAILURE} Warning: no GPU found! On Colab you may want to switch Runtime to: T4 GPU")
        return importlib.import_module("ollama")


def install_ollama_model(model, ollama):
    """install a Ollama model, if it is not installed already"""
    if model not in [m["model"] for m in ollama.list().get("models")]:
        prefix = f"Downloading model {model}: "
        counter = 0
        for chunk in ollama.pull(model=model, stream=True):
            if 'status' in chunk:
                counter += 1
                squeal(prefix + chunk['status'] + f" {counter}")


def process_text_with_ollama(model, prompt, ollama):
    response = ollama.generate(
        model=model,
        prompt=prompt
    )
    return response["response"]


target_labels=["PERSON", "LOCATION"]
NER_CACHE_FILE = "ner_cache.json"

def ollama_run(model, texts_input, make_prompt, in_colab):
    ner_cache = read_json_file(NER_CACHE_FILE)
    texts_output = []
    for index, text in enumerate(texts_input):
        if text in ner_cache and model in ner_cache[text]:
            squeal(f"Retrieving entities for text {index + 1} from cache for model {model}")
            texts_output.append(ner_cache[text][model])
        else:
            if "ollama" in sys.modules:
                ollama = importlib.import_module("ollama")
            else:
                ollama = import_ollama_module()
            install_ollama_model(model, ollama)
            squeal(f"Processing text {index + 1} with model {model}")
            prompt = make_prompt(text, target_labels)
            ollama_response = process_text_with_ollama(model, prompt, ollama)
            texts_output.append(ollama_response)
            if text not in ner_cache:
                ner_cache[text] = {}
            ner_cache[text][model] = ollama_response
            write_json_file(NER_CACHE_FILE, ner_cache)
    print("Finished processing")
    save_data_to_json_file(ner_cache, file_name=NER_CACHE_FILE, in_colab=in_colab)
    return texts_output


def get_char_offsets_of_entity(text_cleaned, entity_text):
    """lookup entity in original text and return list of character offsets"""
    matches = regex.finditer(entity_text, text_cleaned)
    char_offsets = []
    for match in matches:
        char_offsets.append({"start_char": match.start(), "end_char": match.end()})
    return char_offsets


def get_char_offsets_of_entities(text_id, text, raw_entities):
    """lookup list of entities in original text and return list with character offsets"""
    cleaned_entities = []
    for raw_entity in raw_entities:
        char_offsets = get_char_offsets_of_entity(text, raw_entity["text"])
        if not char_offsets:
            print(f"raw entity not found in text! text_id: {text_id}; entity: {raw_entity['text']}")
            continue
        cleaned_entities.extend([char_offset | {"text": raw_entity["text"], "label": raw_entity["label"]} 
                               for char_offset in char_offsets])
    return cleaned_entities


def remove_overlapping_entities(entities):
    """remove overlapping entities from list of entities sorted by start offset"""
    indices_to_be_deleted = []
    for entity_index, entity in enumerate(entities):
        if entity_index > 0:
            if entities[entity_index - 1]["end_char"] > entities[entity_index]["start_char"]:
                indices_to_be_deleted.append(entity_index)
    for entity_index in reversed(indices_to_be_deleted):
        entities.pop(entity_index)
    return entities


def read_lines_from_file(file_name):
    """Read lines from the file file_name, return as list of strings"""
    if type(file_name) != str:
        return read_lines_from_stdin()
    else:
        with open(file_name, "r") as file_handle:
            lines = [line.strip() for line in file_handle]
        file_handle.close()
        return lines


def add_cleaned_entities_to_tokens(text_tokens, cleaned_entities, text_id):
    """Add entity information to token information and return the combination"""
    token_id = 0
    tokens_with_machine_tags = []
    for entity in cleaned_entities:
        if entity["label"] == "PLACE":
            entity["label"] = "LOCATION"
        if entity["label"] not in ["LOCATION", "PERSON"]:
            continue
        while token_id < len(text_tokens) and text_tokens[token_id]["start"] < entity["start_char"]:
            tokens_with_machine_tags.append(text_tokens[token_id] | {"machine_tag": "O"})
            token_id += 1
        if token_id < len(text_tokens):
            if text_tokens[token_id]["start"] != entity["start_char"]:
                print(f"clean entity not found in text! text_id: {text_id}; entity: {entity}; token: {text_tokens[token_id]}")
            else:
                iob = "B"
                while token_id < len(text_tokens) and text_tokens[token_id]["end"] <= entity["end_char"]:
                    tokens_with_machine_tags.append(text_tokens[token_id] | {"machine_tag": f"{iob}-{entity['label']}"})
                    token_id += 1
                    iob = "I"
                if abs(text_tokens[token_id - 1]["end"] - entity["end_char"]) > 1:
                    print(f"warning: machine token boundary error: text_id = {text_id}; entity: {entity}; token: {text_tokens[token_id - 1]}")
    while token_id < len(text_tokens):
        tokens_with_machine_tags.append(text_tokens[token_id] | {"machine_tag": "O"})
        token_id += 1
    return tokens_with_machine_tags


def read_machine_data(file_name):
    """Read the output file of the NER task and return the contents as a list of text results"""
    ner_json = read_json_file(file_name)
    text_tokens = []
    for text_id, text in enumerate(ner_json):
        cleaned_entities = get_char_offsets_of_entities(text_id, text["text_cleaned"], text["entities"])
        tokens_with_machine_tags = add_cleaned_entities_to_tokens(text["tokens"],
                                                                  remove_overlapping_entities(sorted(cleaned_entities, 
                                                                                                     key=lambda entity: (entity["start_char"], 
                                                                                                                         -entity["end_char"]))),
                                                                  text_id)
        text_tokens.append(tokens_with_machine_tags)
    return text_tokens


DOC_SEPARATOR = "-DOCSTART-"
label_names = {"p": "PERSON", "l": "LOCATION"}


def add_gold_entities_to_tokens(lines, text_tokens):
    """Add the golden standard entities to the text tokens"""
    text_tokens_text_index = 0
    text_tokens_token_index = 0
    last_label = "O"
    for line in lines:
        if line == "" or line == f"{DOC_SEPARATOR} {DOC_SEPARATOR}":
            continue
        label, token = line.split()
        if token != text_tokens[text_tokens_text_index][text_tokens_token_index]["text"]:
            print(f"token mismatch!: annotation: {token}; machine: {text_tokens[text_tokens_text_index][text_tokens_token_index]['text']}")
            break
        elif label not in ["p", "l"]:
            text_tokens[text_tokens_text_index][text_tokens_token_index]["gold_tag"] = "O"
            last_label = label
        else:
            iob = "I" if last_label == label else "B"
            text_tokens[text_tokens_text_index][text_tokens_token_index]["gold_tag"] = f"{iob}-{label_names[label]}"
            last_label = label
        text_tokens_token_index += 1
        if text_tokens_token_index >= len(text_tokens[text_tokens_text_index]):
            text_tokens_text_index += 1
            text_tokens_token_index = 0
    return text_tokens


HIPE_MACHINE_FILE = "HIPE_machine.txt"
HIPE_GOLD_FILE = "HIPE_gold.txt"


def save_data_for_evaluation(text_tokens):
    """Save the data for evaluation to two files: on for the machine analysis and one for the golden standard"""
    save_data_to_file(text_tokens, "machine_tag", HIPE_MACHINE_FILE)
    save_data_to_file(text_tokens, "gold_tag", HIPE_GOLD_FILE)


def save_data_to_file(text_tokens, data_column_name, file_name):
    """Save the data for evaluation to a single specified file"""
    with open(file_name, "w") as hipe_file:
        print("TOKEN\tNE-COARSE-LIT\tNE-COARSE-METO\tMISC", file=hipe_file)
        for text in text_tokens:
            sent_id = 0
            for token in text:
                if token["sent_id"] != sent_id:
                    print("-\tO\tO\tO", file=hipe_file)
                    sent_id = token["sent_id"]
                print(token["text"], token[data_column_name], token[data_column_name], "O", sep="\t", file=hipe_file)
            print("-\tO\tO\tO", file=hipe_file)
        hipe_file.close()


ARGS_VALUES = {'--glue': None, '--help': False, '--hipe_edition': 'hipe-2020',
 '--log': 'log.txt', '--n_best': '1', '--noise-level': None,
 '--original_nel': False, '--skip-check': False, '--suffix': None,
 '--tagset': None, '--time-period': None}


def run_scorer(clef_evaluation, args):
    """Run the HIPE-scorer to perform the evaluation"""
    tasks = ("nerc_coarse", "nel")
    if args["--task"] not in tasks:
        msg = "Please restrict to one of the available evaluation tasks: " + ", ".join(tasks)
        logging.error(msg)
        sys.exit(1)
    logging.debug(f"ARGUMENTS {args}")
    clef_evaluation.main(args=args | ARGS_VALUES)


SCORER_DIR = "HIPE-scorer"

def evaluate(text_tokens, clef_evaluation):
    """Evaluate the data present in text tokens: save them, run the scorer and show the results"""
    BASE_DIR = os.getcwd()
    save_data_for_evaluation(text_tokens)
    os.chdir(os.path.join(BASE_DIR, SCORER_DIR))
    run_scorer(clef_evaluation, 
               args={"--ref": os.path.join("..", HIPE_GOLD_FILE),
                     "--pred": os.path.join("..", HIPE_MACHINE_FILE),
                     "--task": "nerc_coarse",
                     "--outdir": "."})
    os.chdir(BASE_DIR)
    show_scores()


LABELS = ["PERSON", "LOCATION"]

def show_scores(labels=LABELS):
    """Extract relevant evaluation scores from the output file of the scorer and show them"""
    hipe_scores = read_json_file(os.path.join(SCORER_DIR, HIPE_MACHINE_FILE))
    print(colored("HIPE Analysis", attrs=["bold"]))
    data = [[round(100*value, 1)
             for key, value in hipe_scores['NE-COARSE-LIT']['TIME-ALL']['LED-ALL'][label]['exact'].items() 
             if regex.search("micro", key)] for label in labels]
    data = [[labels[row_index]] + row for row_index, row in enumerate(data)]
    headers = ["Label"] + [key
                           for key, value in hipe_scores['NE-COARSE-LIT']['TIME-ALL']['LED-ALL'][labels[0]]['exact'].items()
                           if regex.search("micro", key)]
    print(tabulate.tabulate(data, headers=headers, tablefmt="fancy_grid"))


def load_hipe_scorer():
    """Import the HIPE-scorer for evaluation. If it is not yet installed, install it"""
    BASE_DIR = os.getcwd()
    if not os.path.exists("HIPE-scorer/clef_evaluation.py"):
        subprocess.run(["git", "clone", "https://github.com/enriching-digital-heritage/HIPE-scorer.git"])
        os.chdir(os.path.join(BASE_DIR, "HIPE-scorer"))
        subprocess.run(["pip", "install", "-r", "requirements.txt"])
        subprocess.run(["pip", "install", "."])
        os.chdir(BASE_DIR)
    os.chdir(os.path.join(BASE_DIR, "HIPE-scorer"))
    clef_evaluation = importlib.import_module("clef_evaluation")
    os.chdir(BASE_DIR)
    return clef_evaluation


def insert_machine_entities_in_tokens(input_data):
    """Insert machine entities in tokens and return as text_tokens data structure"""
    text_tokens = []
    for text_id, text in enumerate(input_data):
        cleaned_entities = get_char_offsets_of_entities(text_id, text["text_cleaned"], text["entities"])
        cleaned_entities = remove_overlapping_entities(sorted(cleaned_entities,
                                                       key=lambda entity: (entity["start_char"],
                                                                           -entity["end_char"])))
        text_tokens.append(add_cleaned_entities_to_tokens(text["tokens"], cleaned_entities, text_id))
    return text_tokens
    

def add_gold_linking_entities_to_text_tokens(text_tokens, gold_entities):
    text_tokens_text_id = 0
    text_tokens_token_id = 0
    for gold_entities_text in gold_entities:
        for gold_entity in gold_entities_text:
            while text_tokens[text_tokens_text_id][text_tokens_token_id]["start"] < gold_entity["start"]:
                text_tokens[text_tokens_text_id][text_tokens_token_id]["wikidata_id"] = "_"
                text_tokens_token_id += 1
                if text_tokens_token_id >= len(text_tokens[text_tokens_text_id]):
                    break
            while text_tokens_token_id < len(text_tokens[text_tokens_text_id]) and text_tokens[text_tokens_text_id][text_tokens_token_id]["end"] <= gold_entity["end"]:
                if "wikidata_id" not in gold_entity:
                    text_tokens[text_tokens_text_id][text_tokens_token_id]["wikidata_id"] = "_"
                else:
                   text_tokens[text_tokens_text_id][text_tokens_token_id]["wikidata_id"] = gold_entity["wikidata_id"]["id"]
                text_tokens_token_id += 1
        while text_tokens_token_id < len(text_tokens[text_tokens_text_id]):
            text_tokens[text_tokens_text_id][text_tokens_token_id]["wikidata_id"] = "_"
            text_tokens_token_id += 1
        text_tokens_text_id += 1
        text_tokens_token_id = 0
    return text_tokens


def get_char_offsets_of_entities_linking(text_id, text, raw_entities):
    """lookup list of entities in original text and return list with character offsets"""
    cleaned_entities = []
    for raw_entity in raw_entities:
        char_offsets = get_char_offsets_of_entity(text, raw_entity["text"])
        if not char_offsets:
            print(f"raw entity not found in text! text_id: {text_id}; entity: {raw_entity['text']}")
            continue
        cleaned_entities.extend([char_offset | {"text": raw_entity["text"], "wikidata_id": raw_entity["wikidata_id"]} 
                               for char_offset in char_offsets])
    return cleaned_entities


def get_machine_entities(file_name):
    linking_json = read_json_file(file_name)
    machine_entities = []
    for text_id, text in enumerate(linking_json):
        cleaned_entities = get_char_offsets_of_entities_linking(text_id, text["text_cleaned"], text["entities"])
        cleaned_entities = remove_overlapping_entities(sorted(cleaned_entities, 
                                                              key=lambda entity: (entity["start_char"], 
                                                                                  -entity["end_char"])))
        for cleaned_entity in cleaned_entities:
            cleaned_entity["start"] = cleaned_entity.pop("start_char")
            cleaned_entity["end"] = cleaned_entity.pop("end_char")
        machine_entities.append(cleaned_entities)
    return machine_entities


HIPE_MACHINE_FILE = "HIPE_machine.txt"
HIPE_GOLD_FILE = "HIPE_gold.txt"


def save_linking_data_to_file(text_tokens, data_column_name, file_name):
    """Save the data for evaluation to a single specified file"""
    with open(file_name, "w") as hipe_file:
        print("TOKEN\tNE-COARSE-LIT\tNE-COARSE-METO\tNEL-LIT\tNEL-METO\tMISC", file=hipe_file)
        for text in text_tokens:
            sent_id = 0
            for token in text:
                if token["sent_id"] != sent_id:
                    print("-\tO\tO\t_\t_\tO", file=hipe_file)
                    sent_id = token["sent_id"]
                if "wikidata_id" not in token:
                    token["wikidata_id"] = {"id": "_", "description": ""}
                print(token["text"], token[data_column_name], token[data_column_name], 
                                     token["wikidata_id"], token["wikidata_id"], "O", sep="\t", file=hipe_file)
            print("-\tO\tO\t_\t_\tO", file=hipe_file)
        hipe_file.close()


SCORER_DIR = "HIPE-scorer"


def evaluate_linking(text_tokens_gold, text_tokens_machine, clef_evaluation):
    """Evaluate the data present in text tokens: save them, run the scorer and show the results"""
    BASE_DIR = os.getcwd()
    save_linking_data_to_file(text_tokens_gold, "gold_tag", HIPE_GOLD_FILE)
    save_linking_data_to_file(text_tokens_machine, "machine_tag", HIPE_MACHINE_FILE)
    os.chdir(os.path.join(BASE_DIR, SCORER_DIR))
    run_scorer(clef_evaluation,
               args={"--ref": os.path.join("..", HIPE_GOLD_FILE),
                     "--pred": os.path.join("..", HIPE_MACHINE_FILE),
                     "--task": "nel",
                     "--outdir": "."})
    os.chdir(BASE_DIR)
    show_scores_linking()


def show_scores_linking(labels=["ALL"]):
    """Extract relevant evaluation scores from the output file of the scorer and show them"""
    hipe_scores = read_json_file(os.path.join(SCORER_DIR, HIPE_MACHINE_FILE))
    print(colored("HIPE Analysis", attrs=["bold"]))
    data = [[round(100*value, 1) 
             for key, value in hipe_scores['1']['NEL-LIT']['TIME-ALL']['LED-ALL'][label]['exact'].items() 
             if regex.search("micro", key)] for label in labels]
    data = [[labels[row_index]] + row for row_index, row in enumerate(data)]
    headers = ["Label"] + [key
                           for key, value in hipe_scores['1']['NEL-LIT']['TIME-ALL']['LED-ALL'][labels[0]]['exact'].items()
                           if regex.search("micro", key)]
    print(tabulate.tabulate(data, headers=headers, tablefmt="fancy_grid"))


def add_char_offsets_to_processed_entities(processed_entities):
    """Add missing character offsets to machine entities and return in datastructure per text"""
    machine_entities = []
    for entity in processed_entities:
        char_offsets = get_char_offsets_of_entities(entity["text_id"], 
                                                    entity["text"], 
                                                    [{"text": entity["entity_text"], 
                                                            "label": "PERSON"}])
        while len(machine_entities) <= entity["text_id"]:
            machine_entities.append([])
        for char_offset in char_offsets:
            machine_entities[entity["text_id"]].append(entity | {"start_char": char_offset["start_char"],
                                                                 "end_char": char_offset["end_char"]})
    cleaned_entities = [remove_overlapping_entities(sorted(entities,
                                                    key=lambda entity: (entity["start_char"],
                                                                        -entity["end_char"])))
                        for entities in machine_entities]
    for entities in cleaned_entities:
        for entity in entities:
            entity["start"] = entity.pop("start_char")
            entity["end"] = entity.pop("end_char")
    return cleaned_entities
