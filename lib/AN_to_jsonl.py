"""
Created on Fri Apr 18 21:08:05 2025

@author: rayanleveque
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

def make_json_line(dirname, dataset_name,
                   ocr_line, ocr_sentence,
                   gt_line, gt_sentence):
    obj = {
        "filename": direname + filename,
        "dataset_name": dataset_name,
        "ocr": {
            "line": ocr_line,
            "sentence": None
        },
        "groundtruth": {
            "line": gt_line,
            "sentence": None
        }
    }
    print(json.dumps(obj, ensure_ascii=False))

if __name__ == "__main__":
    dirname     = "../data/dataset/ocr/original/AN/"
    dataset_name = "AN"
    ocr_line     = input("OCR — line               : ")
    ocr_sentence = input("OCR — sentence           : ")
    gt_line      = input("Groundtruth — line      : ")
    gt_sentence  = input("Groundtruth — sentence  : ")
    make_json_line(dirname, dataset_name,
                   ocr_line, ocr_sentence,
                   gt_line, gt_sentence)
