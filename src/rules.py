# src/rules.py
"""
Cấu hình và mappings cho hệ thống gợi ý bài tập gym
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_PATH = os.path.join(DATA_DIR, 'data_with_bert.pkl')

BERT_MODEL_DIR = os.path.join(BASE_DIR, "model_bert")
BERT_MAX_LENGTH = 256

GOAL_MAP = {
    "gain abs": {
        "bodyparts": ["abdominals"],
        "types": ["Strength"]
    },
    "6 packes": {
        "bodyparts": ["abdominals"],
        "types": ["Cardio", "Strength"]
    },
    "thicked chest": {
        "bodyparts": ["chest"],
        "types": ["Strength"]
    },
    "broad shoulders": {
        "bodyparts": ["shoulders"],
        "types": ["Strength"]
    },
    "big arm": {
        "bodyparts": ["biceps", "triceps", "forearms"],
        "types": ["Strength"]
    },
    "big leg": {
        "bodyparts": ["quadriceps", "hamstrings", "calves", "glutes"],
        "types": ["Strength"]
    },
    "demon back": {
        "bodyparts": ["lats", "middle back", "lower back", "traps"],
        "types": ["Strength"]
    },
    "fat lose": {
        "bodyparts": [],
        "types": ["Cardio"]
    },
    "mana": {
        "bodyparts": [],
        "types": ["Cardio", "Plyometrics"]
    },
    "recover": {
        "bodyparts": [],
        "types": ["Stretching"]
    }
}

INJURY_MAP = {
    "spine": {
        "bodypart": ["lower back", "middle back", "lats"],
        "equipment": ["barbell", "machine"],
        "type": ["powerlifting", "olympic weightlifting", "strongman"]
    },
    "shoulder": {
        "bodypart": ["shoulders", "traps"],
        "equipment": ["barbell", "dumbbell", "cable"],
        "type": ["strength", "powerlifting", "olympic weightlifting"]
    },
    "knee": {
        "bodypart": ["quadriceps", "hamstrings", "calves", "glutes"],
        "equipment": ["barbell", "machine"],
        "type": ["plyometrics", "strength", "powerlifting"]
    },
    "wrist": {
        "bodypart": ["forearms", "biceps", "triceps"],
        "equipment": ["barbell", "dumbbell", "e-z curl bar", "kettlebells"],
        "type": ["strength"]
    },
    "neck": {
        "bodypart": ["traps", "shoulders"],
        "equipment": ["barbell", "cable"],
        "type": ["powerlifting", "olympic weightlifting"]
    }
}

DEFAULT_TOP_N = 10
DEFAULT_SAMPLE_SIZE = 2