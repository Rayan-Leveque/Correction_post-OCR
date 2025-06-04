# -*- coding: utf-8 -*-
from torch import nn
import openai
import yaml
import os
import argparse
from tqdm import tqdm
import importlib
import jsonlines
import logging
from const import Const
from retrying import retry

logger = logging.getLogger("gpt-experiments")
logger.setLevel(logging.INFO)

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_dict(list_of_dicts: list):
    return_list = {}
    for list_entry in list_of_dicts:
        return_list |= list_entry
    return return_list


def generate(
    input_dir: str = "../data/datasets",
    output_dir: str = "../data/output",
    prompt_dir: str = "../data/prompts",
    config_file: str = "../data/config.yml",
    prompt_name: str = "prompt_basic_02.txt",
    few_shot: bool = True,
    lang_specific: bool = False,
    temperature: float = None,
    model_class_filter: str = None,

) -> None:
    """
    Génère des textes via plusieurs modèles définis dans le fichier de configuration
    et sauvegarde les prédictions dans des fichiers de sortie.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if model_class_filter:
        filtered = []
        for entry in config["models"]:
            # each entry is {model_name: [ {class:…}, {prompt:…}, … ]}
            (model_name, details_list), = entry.items()
            details = get_dict(details_list)
            if details.get("class") == model_class_filter:
                filtered.append(entry)
        logger.info(f"Keeping only models with class={model_class_filter}: {len(filtered)} left")
        config["models"] = filtered

    # Construction du chemin vers le prompt de base en utilisant le paramètre prompt_name
    prompt_path = os.path.join(prompt_dir, prompt_name)
    print(prompt_dir, prompt_name)
    if os.path.exists(prompt_path):
        logger.info(f"Loading prompt from {prompt_path}.")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
            print(prompt)
    else:
        logger.info(f"Model prompt missing: {prompt_path}. The prompt will be loaded dynamically.")

    for model in config["models"]:
        (model_name, experiment_details), = model.items()
        logger.info("Experimenting with {}".format(model_name))
        experiment_details = get_dict(experiment_details)
        model_class = experiment_details["class"]

        if few_shot and lang_specific:
            results_dir = os.path.join(
                output_dir,
                "few_shot/prompt_complex_lang/",
                prompt_name.replace(".txt", "")
            )
        elif few_shot:
            results_dir = os.path.join(
                output_dir,
                "few_shot",
                prompt_name.replace(".txt", "")
            )
            print(results_dir)
        else:
            results_dir = os.path.join(
                output_dir,
                prompt_name.replace(".txt", "")
            )
            print(results_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        module = importlib.import_module("prompt")
        class_ = getattr(module, model_class)

        instance = class_(
            model=model_name,
            temperature=temperature,   # ← propagate CLI value
        )


        print("WHA", input_dir)
        for root, dirs, files in os.walk(input_dir, topdown=False):
            for name in files:
                if name.endswith("jsonl"):
                    input_file = os.path.join(root, name)
                    logger.info(f"Post-correcting {input_file}")
                    dataset_name = name.replace(".jsonl", "")
                    print("-----", input_file, dataset_name)

                    dataset_model_results_dir = os.path.join(results_dir, dataset_name)
                    if not os.path.exists(dataset_model_results_dir):
                        os.makedirs(dataset_model_results_dir)

                    # build the “base” file name
                    if few_shot:
                        base_fname = f"results-3few-shot-{dataset_name}-{model_name}"
                    else:
                        base_fname = f"results-{dataset_name}-{model_name}"
                    
                    # append the temperature tag if given
                    if temperature is not None:
                        # format so you get, e.g., “-temp0.3”
                        base_fname += f"-temp{temperature}"
                    
                    # sanitize and add extension
                    fname = base_fname.replace("/", "-") + ".jsonl"
                    
                    # final path
                    output_file = os.path.join(dataset_model_results_dir, fname)

                    logger.info("Predictions for {} are saved in {}".format(input_file, output_file))

                    with open(input_file, "r") as g:
                        total_files = len(g.readlines())

                    progress_bar = tqdm(total=total_files, desc="Processing files", unit="file")

                    mode = "w"
                    count = None
                    if os.path.exists(output_file):
                        mode = "a"
                        logger.info("We found the results file {}. We will continue predicting from where it was left off.".format(output_file))
                        lines = []
                        count = 0
                        with jsonlines.open(output_file, "r") as f:
                            for json_line in f:
                                lines.append(json_line)
                                count += 1

                    print("\n\n\n\n\n" + model_name + "\n\n\n\n")
                    # Decide prompt_type based on the prompt file’s name:
                    name_lower = prompt_name.lower()
                    if "basic" in name_lower:
                        prompt_type_flag = "basic"
                    elif "complex" in name_lower:
                        prompt_type_flag = "complex_01"
                    else:
                        prompt_type_flag = "basic"    # fallback
                
                    @retry(stop_max_attempt_number=4, wait_exponential_multiplier=1000, wait_exponential_max=10000)
                    def get_prediction(prompt_str, model_name, text):
                        options = {
                            "engine": model_name,
                            "top_p": 1.0,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,
                            "prompt_type": prompt_type_flag   # ← now dynamic
                        }
                        if temperature is not None:
                            options["temperature"] = temperature
                        logger.info(f"[DEBUG] get_prediction opts: {options}")
                        return instance.prediction(prompt_str, options, text)


                    already_done = {}

                    with jsonlines.open(output_file, mode) as f:
                        with jsonlines.open(input_file, "r") as g:
                            for idx, json_line in enumerate(g):
                                progress_bar.update(1)
                                if count is not None and idx <= count:
                                    continue

                                data = {Const.PREDICTION: {}}

                                for TEXT_LEVEL in [Const.LINE, Const.SENTENCE]:
                                    ocr_text = json_line[Const.OCR][TEXT_LEVEL]
                                    if ocr_text not in already_done:
                                        if ocr_text is not None:
                                            # Détermination de la langue
                                            if "ajmc" in dataset_name:
                                                language = "el"
                                            elif "overproof" in dataset_name:
                                                language = "en"
                                            elif "impresso" in dataset_name:
                                                language = "de"
                                            elif "htrec" in dataset_name:
                                                language = "el"
                                            elif "ina" in dataset_name:
                                                language = "fr"
                                            elif "icdar-2017" in dataset_name:
                                                language = json_line["filename"].split("/")[-2].split("_")[0]
                                                if language == "eng":
                                                    language = "en"
                                            else:
                                                language = json_line["language"]

                                            if few_shot and lang_specific:
                                                prompt_path = os.path.join(
                                                    prompt_dir, "few_shot_lang", dataset_name.replace("_", "-"),
                                                    "{}_{}_{}.txt".format(prompt_name.replace(".txt", ""), TEXT_LEVEL, language)
                                                )
                                                if os.path.exists(prompt_path):
                                                    with open(prompt_path, "r", encoding="utf-8") as g:
                                                        prompt_specific = g.read()
                                                else:
                                                    logger.info("----Model prompt missing: {}.".format(prompt_path))
                                                data[Const.PREDICTION][Const.PROMPT] = prompt_specific.replace("{{TEXT}}", ocr_text)
                                            elif few_shot:
                                                prompt_path = os.path.join(
                                                    prompt_dir, "few_shot", dataset_name.replace("_", "-"),
                                                    "{}_{}_{}.txt".format(prompt_name.replace(".txt", ""), TEXT_LEVEL, language)
                                                )
                                                if os.path.exists(prompt_path):
                                                    with open(prompt_path, "r", encoding="utf-8") as g:
                                                        few_shot_prompt = g.read()
                                                else:
                                                    logger.info("----Model prompt missing: {}.".format(prompt_path))
                                                data[Const.PREDICTION][Const.PROMPT] = few_shot_prompt.replace("{{TEXT}}", ocr_text)
                                            elif lang_specific:
                                                if os.path.exists(prompt_path):
                                                    logger.info("---Loading prompt from {}.".format(prompt_path))
                                                    with open(prompt_path, "r", encoding="utf-8") as g:
                                                        lang_prompt = g.read()
                                                    data[Const.PREDICTION][Const.PROMPT] = lang_prompt.replace("{{TEXT}}", ocr_text)
                                                else:
                                                    prompt_path = os.path.join(
                                                        prompt_dir, "prompt_complex_02_{}.txt".format(language)
                                                    )
                                                    if os.path.exists(prompt_path):
                                                        with open(prompt_path, "r", encoding="utf-8") as g:
                                                            lang_prompt = g.read()
                                                    else:
                                                        logger.info("----Model prompt missing: {}.".format(prompt_path))
                                                    data[Const.PREDICTION][Const.PROMPT] = lang_prompt.replace("{{TEXT}}", ocr_text)
                                            else:
                                                data[Const.PREDICTION][Const.PROMPT] = prompt.replace("{{TEXT}}", ocr_text)

                                            result = get_prediction(data[Const.PREDICTION][Const.PROMPT], model_name, ocr_text)
                                            data[Const.PREDICTION][TEXT_LEVEL] = result
                                            already_done[ocr_text] = result
                                    else:
                                        data[Const.PREDICTION].update({TEXT_LEVEL: already_done[ocr_text]})

                                data = json_line | data
                                f.write(data)
                    progress_bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir", type=str,
        help="Base folder with input files."
    )
    parser.add_argument(
        "--output_dir", type=str,
        help="Base folder for prediction files."
    )
    parser.add_argument(
        "--prompt_dir", type=str,
        help="Base folder with prompts data."
    )
    parser.add_argument(
        "--prompt", default="prompt_basic_02.txt", type=str,
        help="The selected prompt."
    )
    parser.add_argument(
        "--config_file", default="../data/config.yml", type=str,
        help="The selected config file."
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
        help="Print lots of debugging statements"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
        help="Be verbose"
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Enable few-shot prompts"
    )
    parser.add_argument(
        "--lang-specific",
        action="store_true",
        help="Enable language-specific prompts"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the default temperature for all models."
    )
    parser.add_argument(
    "--model-class",
    choices=["GPTPrompt", "NovitaPrompt"],
    default=None,
    help="Only run models that use this Prompt wrapper class."
    )


    args = parser.parse_args()
    print(args)
    
    generate(
        args.input_dir,
        args.output_dir,
        args.prompt_dir,
        args.config_file,
        args.prompt,
        args.few_shot,
        args.lang_specific,
        temperature=args.temperature,
        model_class_filter=args.model_class,    # ← new
    )


