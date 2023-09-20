"""
IndicXNLI: Evaluating Multilingual Inference for Indian Languages
https://aclanthology.org/2022.emnlp-main.755/

IndicXNLI is an automatically translated version of XNLI in 11 Indic languages.

Dataset page: https://huggingface.co/datasets/Divyanshu/indicxnli
"""
from .xnli import XNLIBase


_CITATION = """
@inproceedings{aggarwal-etal-2022-indicxnli,
    title = "IndicXNLI: Evaluating Multilingual Inference for Indian Languages",
    author = "Aggarwal, Divyanshu  and
      Gupta, Vivek  and
      Kunchukuttan, Anoop",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.755",
    doi = "10.18653/v1/2022.emnlp-main.755",
    pages = "10994--11006",
    abstract = "While Indic NLP has made rapid advances recently in terms of the availability of corpora and pre-trained models, benchmark datasets on standard NLU tasks are limited. To this end, we introduce INDICXNLI, an NLI dataset for 11 Indic languages. It has been created by high-quality machine translation of the original English XNLI dataset and our analysis attests to the quality of INDICXNLI. By finetuning different pre-trained LMs on this INDICXNLI, we analyze various cross-lingual transfer techniques with respect to the impact of the choice of language models, languages, multi-linguality, mix-language input, etc. These experiments provide us with useful insights into the behaviour of pre-trained models for a diverse set of languages.",
}
"""

class IndicXNLI(XNLIBase):  # English
    DATASET_PATH = "Divyanshu/indicxnli"
    DATASET_NAME = None

    QUESTION_WORD = "right"
    ENTAILMENT_LABEL = "Yes"
    NEUTRAL_LABEL = "Also"
    CONTRADICTION_LABEL = "No"


class IndicXNLI_as(IndicXNLI): # Assamese
    DATASET_NAME = "as"

    QUESTION_WORD = "নহয়নে"
    ENTAILMENT_LABEL = "হয়"
    NEUTRAL_LABEL = "লগতে"
    CONTRADICTION_LABEL = "নাই"


class IndicXNLI_bn(IndicXNLI): # Bengali
    DATASET_NAME = "bn"

    QUESTION_WORD = "তাই না"
    ENTAILMENT_LABEL = "হ্যাঁ"
    NEUTRAL_LABEL = "এছাড়াও"
    CONTRADICTION_LABEL = "না"


class IndicXNLI_gu(IndicXNLI): # Gujarati
    DATASET_NAME = "gu"

    QUESTION_WORD = "ખરું ને"
    ENTAILMENT_LABEL = "હા"
    NEUTRAL_LABEL = "ઉપરાંત"
    CONTRADICTION_LABEL = "ના"


class IndicXNLI_hi(IndicXNLI): # Hindi
    DATASET_NAME = "hi"

    QUESTION_WORD = "है ना"
    ENTAILMENT_LABEL = "हाँ"
    NEUTRAL_LABEL = "इसके अलावा"
    CONTRADICTION_LABEL = "नहीं"


class IndicXNLI_kn(IndicXNLI): # Kannada
    DATASET_NAME = "kn"

    QUESTION_WORD = "ಸರಿ"
    ENTAILMENT_LABEL = "ಹೌದು"
    NEUTRAL_LABEL = "ಅಲ್ಲದೆ"
    CONTRADICTION_LABEL = "ಇಲ್ಲ"


class IndicXNLI_ml(IndicXNLI): # Malayalam
    DATASET_NAME = "ml"

    QUESTION_WORD = "അല്ലേ"
    ENTAILMENT_LABEL = "അതെ"
    NEUTRAL_LABEL = "കൂടാതെ"
    CONTRADICTION_LABEL = "ഇല്ല"


class IndicXNLI_mr(IndicXNLI): # Marathi
    DATASET_NAME = "mr"

    QUESTION_WORD = "बरोबर"
    ENTAILMENT_LABEL = "होय"
    NEUTRAL_LABEL = "तसेच"
    CONTRADICTION_LABEL = "नाही"


class IndicXNLI_or(IndicXNLI): # Oriya
    DATASET_NAME = "or"

    QUESTION_WORD = "ଠିକ୍"
    ENTAILMENT_LABEL = "ହଁ"
    NEUTRAL_LABEL = "ଆହୁରି ମଧ୍ୟ"
    CONTRADICTION_LABEL = "ନା"


class IndicXNLI_pa(IndicXNLI): # Punjabi
    DATASET_NAME = "pa"

    QUESTION_WORD = "ਠੀਕ ਹੈ"
    ENTAILMENT_LABEL = "ਹਾਂ"
    NEUTRAL_LABEL = "ਨਾਲ ਹੀ"
    CONTRADICTION_LABEL = "ਨਹੀਂ"


class IndicXNLI_ta(IndicXNLI): # Tamil
    DATASET_NAME = "ta"

    QUESTION_WORD = "இல்லையா"
    ENTAILMENT_LABEL = "ஆம்"
    NEUTRAL_LABEL = "மேலும்"
    CONTRADICTION_LABEL = "இல்லை"


class IndicXNLI_te(IndicXNLI): # Telugu
    DATASET_NAME = "te"

    QUESTION_WORD = "సరియైనదా"
    ENTAILMENT_LABEL = "అవును"
    NEUTRAL_LABEL = "అలాగే"
    CONTRADICTION_LABEL = "లేదు"


LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

LANG_CLASSES = [
    IndicXNLI_as,
    IndicXNLI_bn,
    IndicXNLI_gu,
    IndicXNLI_hi,
    IndicXNLI_kn,
    IndicXNLI_ml,
    IndicXNLI_mr,
    IndicXNLI_or,
    IndicXNLI_pa,
    IndicXNLI_ta,
    IndicXNLI_te
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"indicxnli_{lang}"] = lang_class
    return tasks
