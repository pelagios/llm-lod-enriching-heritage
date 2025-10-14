import hashlib
import importlib
from IPython.display import clear_output, display
import json
import subprocess
import sys
try:
    from google.colab import files
except:
    pass


CHAR_PACKAGE = "📦"
CHAR_SUCCESS = "✅"
CHAR_FAILURE = "❌"
COLORS = {"PERSON": "red", "LOCATION": "green", "OTHER": "blue"}


def safe_import(package_name):
    """Import a package;. If it missing, download it first"""
    try:
        return importlib.import_module(package_name)
    except ImportError:
        print(f"{CHAR_PACKAGE} {package_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Finished installing {package_name}")
        return importlib.import_module(package_name)


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
        entity_label = entity["label"] if entity["label"] in COLORS.keys() else "OTHER"
        if "wikidata_id" in entity:
            texts_input = texts_input[:entity["end_char"]] + f"<sup>{entity['wikidata_id']['id']}</sup>" + texts_input[entity["end_char"]:]
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
        if in_colab:
            try:
                files.download(output_file_name)
                print(f"️{CHAR_SUCCESS} Downloaded preprocessed texts to file {output_file_name}")
            except:
                print(f"️{CHAR_FAILURE} Downloading preprocessed texts failed!")
        else:
            print(f"️{CHAR_SUCCESS} Saved preprocessed texts to file {output_file_name}")