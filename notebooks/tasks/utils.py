from dotenv import load_dotenv
import hashlib
import importlib
from IPython.display import clear_output, display, HTML
import json
import os
import subprocess
import sys
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


langid = safe_import("langid")
openai = safe_import("openai")


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


def mark_entities_in_text(texts_input, entities):
    """Convert the text to HTML with colored antities and return these"""
    for entity in reversed(entities):
        entity["label"] = "LOCATION" if entity["label"] in LOCATION_ALTERNATIVES else entity["label"]
        entity_label = entity["label"] if entity["label"] in COLORS.keys() else "OTHER"
        if "wikidata_id" in entity:
            label_text = entity['wikidata_id']['id']
            if "link" in entity and "gpt-4o-mini" in entity["link"]:
                label_text += "," + entity["link"]["gpt-4o-mini"][0]
            texts_input = texts_input[:entity["end_char"]] + f"<sup>{label_text}</sup>" + texts_input[entity["end_char"]:]
        texts_input = texts_input[:entity["end_char"]] + "</span>" + texts_input[entity["end_char"]:]
        texts_input = (texts_input[:entity["start_char"]] + 
                      f"<span style=\"border: 1px solid black; color: {COLORS[entity_label]};\">" + 
                      texts_input[entity["start_char"]:])
    return texts_input


def save_results(texts, key, in_colab):
    """Save preprocessed texts in a json file"""
    json_string = json.dumps(texts, ensure_ascii=False, indent=2)
    hash = hashlib.sha1(json_string.encode("utf-8")).hexdigest()
    output_file_name = f"output_{key}{hash}.json"
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        print(json_string, end="", file=output_file)
        output_file.close()
        print(f"️{CHAR_SUCCESS} Saved preprocessed texts to file {output_file_name}")
    if in_colab:
        try:
            files.download(output_file_name)
            print(f"️{CHAR_SUCCESS} Downloaded preprocessed texts to file {output_file_name}")
        except:
            print(f"️{CHAR_FAILURE} Downloading preprocessed texts failed!")


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
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        try:
            with open("OPENAI_API_KEY", "r") as infile:
                openai_api_key = infile.read().strip()
                infile.close()
        except:
            pass
    if not openai_api_key:
        print(f"{CHAR_FAILURE} no openai_api_key found!")
        return ""
    return openai_api_key


def connect_to_openai(openai_api_key):
    """Connect to OpenAI and return processing space"""
    return openai.OpenAI(api_key=openai_api_key)


def process_text_with_gpt(openai_client, model, prompt):
    """Send text to OpenAI via prompt and return results"""
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except:
        print(f"{CHAR_FAILURE} GPT call failed")
        return []
