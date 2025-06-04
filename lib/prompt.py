# -*- coding: utf-8 -*-
import os

from openai import OpenAI

class Prompt:
    '''
    Base Prompt interface. Subclasses must implement `prediction`.
    '''
    def __call__(self, prompt, options=None, text=""):
        return self.prediction(prompt, options, text)

    def prediction(self, prompt, options=None, text=""):
        raise NotImplementedError("Subclasses must implement prediction()");

class GPTPrompt(Prompt):
    def __init__(self, api_key=None, model='text-davinci-003', device='cpu', temperature: float = None):
        self.device = device
        # valeurs par défaut
        self.options = {
            'engine': model,
            'temperature': 0.7,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'max_tokens': 3000
        }
        # si on a passé un override, on l’applique
        if temperature is not None:
            self.options['temperature'] = temperature

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def prediction(self, prompt, options=None, text=""):
        opts = self.options.copy()
        if options:
            opts.update(options)
        is_chat = opts.get("prompt_type") == "complex_01" or any(x in opts.get('engine', '') for x in ['3', '4'])
        if opts.get("prompt_type") == "complex_01":
            # on récupère le modèle depuis engine
            opts['model'] = opts.pop('engine')
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user",   "content": text}
            ]
            extra = {"messages": messages}
        elif is_chat:
            # même traitement pour le cas purement chat
            opts['model'] = opts.pop('engine')
            messages = [{"role": "user", "content": prompt}]
            extra = {"messages": messages}
        else:
            extra = {"prompt": prompt}
        opts.pop("prompt_type", None)
        caller = self.client.chat.completions.create if is_chat else self.client.completions.create
        response = caller(**opts, **extra)

        if is_chat:
            return response.choices[0].message.content
        return response.choices[0].text

class NovitaPrompt(Prompt):
    '''
    Wrapper for NovitaAI chat completions…
    '''
    def __init__(
        self,
        api_key=None,
        model='deepseek-v3-turbo',
        temperature: float = None      # ← nouveau paramètre
    ):
        # token et client Novita…
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://api.novita.ai/v3/openai",
            api_key= os.getenv("OPENAI_NOVITA_KEY")
        )
        self.model = model

        # defaults incluant temperature
        self.defaults = {
            "max_tokens": 3000,
            "temperature": 0.7,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        # si override CLI, on l’applique
        if temperature is not None:
            self.defaults["temperature"] = temperature

    def prediction(self, prompt, options=None, text=""):
        # Merge defaults and overrides
        settings = {**self.defaults, **(options or {})}

        # Build messages…
        if settings.get("prompt_type") == "complex_01":
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user",   "content": text}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        # Utilisation de settings["temperature"] automatiquement
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            frequency_penalty=settings["frequency_penalty"],
            presence_penalty=settings["presence_penalty"]
        )
        return resp.choices[0].message.content


