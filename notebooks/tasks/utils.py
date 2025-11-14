import hashlib
import importlib
from IPython.display import clear_output, display, HTML
import json
import os
import polars as pl
import requests
import regex
import shutil
import subprocess
import sys
import time
import torch
from typing import List, Dict, Any, Tuple, Optional
import yaml

with open('../config.yml') as yaml_file:
    data = yaml.safe_load(yaml_file)


try:
    from google.colab import files
except:
    pass


CHAR_PACKAGE = "📦"
CHAR_SUCCESS = "✅"
CHAR_FAILURE = "❌"
COLORS = data['utils']['label_colors']
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
    """Return True if CUDA (NVIDIA) or MPS (Apple Silicon) is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()

'''def has_gpu() -> bool:
    """check if there is a gpu available, otherwise runs will take a lot of time"""
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=True)
        return True
    except:
        return False
'''

def install_ollama_linux():
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


def import_ollama_module(arch='mac'):
    """import Ollama module in Python"""
    try:
        if not has_gpu():
            print(f"{CHAR_FAILURE} Warning: no GPU found! On Colab you may want to switch Runtime to: T4 GPU")
        return importlib.import_module("ollama")
    except Exception as e:
        if arch=='mac':
            install_ollama_mac()
        elif arch=='linux':
            install_ollama_linux()
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
        importlib.invalidate_caches()
        if not has_gpu():
            print(f"{CHAR_FAILURE} Warning: no GPU found! On Colab you may want to switch Runtime to: T4 GPU")
        return importlib.import_module("ollama")


def install_ollama_mac():
    """Install Ollama on macOS, start the server, and verify it is running."""

    print("📦 Installing Ollama (macOS)")

    # Check if Homebrew exists (best way to install on mac)
    brew_path = shutil.which("brew")

    try:
        if brew_path:
            # Install via Homebrew
            subprocess.run(["brew", "install", "ollama"], check=True)
        else:
            # Fallback: use the official mac installer script
            subprocess.run(
                ["curl", "-fsSL", "https://ollama.com/install.sh"],
                check=True,
                stdout=subprocess.PIPE
            )
            subprocess.run("sh install.sh", shell=True, check=True)
    except Exception as e:
        print(f"{CHAR_FAILURE} Failed to install Ollama:", e)
        raise

    # Start the server
    print("▶️ Starting ollama server…")
    server = subprocess.Popen(["ollama", "serve"])

    # Wait for the API to come up
    for _ in range(60):
        try:
            requests.get("http://127.0.0.1:11434/api/tags", timeout=1)
            print(f"{CHAR_SUCCESS} Ollama server is running")
            break
        except Exception:
            time.sleep(1)
    else:
        server.terminate()
        raise RuntimeError(f"{CHAR_FAILURE} Ollama server did not start")

    time.sleep(1)
    return server

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


target_labels= data['museum']['labels']
NER_CACHE_FILE = data['utils']['ner_cache_file']

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
