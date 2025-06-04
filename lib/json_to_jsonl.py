#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:05:10 2025

@author: rayanleveque
"""

import json

with open("../data/ina_last.json", "r", encoding="utf-8") as infile, open("../ina_last.jsonl", "w", encoding="utf-8") as outfile:
    data = json.load(infile)
    for item in data:
        outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
