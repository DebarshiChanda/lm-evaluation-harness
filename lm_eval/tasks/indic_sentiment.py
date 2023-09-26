"""
Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages
https://arxiv.org/abs/2212.05409

IndicSentiment is a new, multilingual, and n-way parallel dataset for sentiment analysis in 13 Indic languages

Dataset page: https://huggingface.co/datasets/ai4bharat/IndicSentiment
"""
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, matthews_corrcoef, f1_score, yesno


_CITATION = """
@article{Doddapaneni2022towards,
  title={Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
  author={Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
  journal={ArXiv},
  year={2022},
  volume={abs/2212.05409}
}
"""

class IndicSentiment(Task):
    VERSION = 0
    DATASET_PATH = "ai4bharat/IndicSentiment"
    DATASET_NAME = None
    POSITIVE_LABEL = None
    NEGATIVE_LABEL = None
    SENTENCE_TEMPLATE = None # \nQuestion: Is this sentence POSITIVE_LABEL or NEGATIVE_LABEL?\nAnswer:

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        self.dataset["test"] = self.dataset["test"].filter(lambda example: example['LABEL'] is not None)
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "{}{}".format(doc["INDIC REVIEW"], self.SENTENCE_TEMPLATE)

    def doc_to_target(self, doc):
        return " {}".format(doc["LABEL"])

    def construct_requests(self, doc, ctx):
        ll_positive, _ = rf.loglikelihood(ctx, f" {self.POSITIVE_LABEL}")
        ll_negative, _ = rf.loglikelihood(ctx, f" {self.NEGATIVE_LABEL}")
        return ll_positive, ll_negative

    def process_results(self, doc, results):
        ll_positive, ll_negative = results
        pred = ll_positive > ll_negative
        gold = {"Negative": 0, "Positive": 1}[doc["LABEL"]]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class IndicSentiment_as(IndicSentiment): # Assamese
    DATASET_NAME = "translation-as"
    POSITIVE_LABEL = "ধনাত্মক"
    NEGATIVE_LABEL = "ঋণাত্মক"
    SENTENCE_TEMPLATE = f"\nপ্ৰশ্ন : এই বাক্যটো {POSITIVE_LABEL} নে {NEGATIVE_LABEL}?\nউত্তৰ:"


# class IndicSentiment_bd(IndicSentiment):
#     DATASET_NAME = "translation-bd"
#     POSITIVE_LABEL = "ধনাত্মক"
#     NEGATIVE_LABEL = "ঋণাত্মক"
#     SENTENCE_TEMPLATE = f"\nপ্ৰশ্ন : এই বাক্যটো {POSITIVE_LABEL} নে {NEGATIVE_LABEL}?\nউত্তৰ:"


class IndicSentiment_bn(IndicSentiment): # Bengali
    DATASET_NAME = "translation-bn"
    POSITIVE_LABEL = "ইতিবাচক"
    NEGATIVE_LABEL = "নেতিবাচক"
    SENTENCE_TEMPLATE = f"\nপ্রশ্ন: এই বাক্যটি কি {POSITIVE_LABEL} নাকি {NEGATIVE_LABEL}?\nউত্তর:"


class IndicSentiment_en(IndicSentiment): # English
    DATASET_NAME = "translation-hi"
    POSITIVE_LABEL = "Positive"
    NEGATIVE_LABEL = "Negative"
    SENTENCE_TEMPLATE = f"\nQuestion: Is this sentence {POSITIVE_LABEL} or {NEGATIVE_LABEL}?\nAnswer:"

    def doc_to_text(self, doc):
        return "{}{}".format(doc["ENGLISH REVIEW"], self.SENTENCE_TEMPLATE)


class IndicSentiment_gu(IndicSentiment): # Gujarati
    DATASET_NAME = "translation-gu"
    POSITIVE_LABEL = "સકારાત્મક"
    NEGATIVE_LABEL = "નકારાત્મક"
    SENTENCE_TEMPLATE = f"\nપ્રશ્ન: આ વાક્ય {POSITIVE_LABEL} છે કે {NEGATIVE_LABEL}?\nજવાબ:"


class IndicSentiment_hi(IndicSentiment): # Hindi
    DATASET_NAME = "translation-hi"
    POSITIVE_LABEL = "सकारात्मक"
    NEGATIVE_LABEL = "नकारात्मक"
    SENTENCE_TEMPLATE = f"\nप्रश्न: यह वाक्य {POSITIVE_LABEL} है या {NEGATIVE_LABEL}?\nउत्तर:"


class IndicSentiment_kn(IndicSentiment): # Kannada
    DATASET_NAME = "translation-kn"
    POSITIVE_LABEL = "ಧನಾತ್ಮಕ"
    NEGATIVE_LABEL = "ಋಣಾತ್ಮಕ"
    SENTENCE_TEMPLATE = f"\nಪ್ರಶ್ನೆ: ಈ ವಾಕ್ಯವು {POSITIVE_LABEL}ವೇ ಅಥವಾ {NEGATIVE_LABEL}ವೇ?\nಉತ್ತರ:"


class IndicSentiment_ml(IndicSentiment): # Malayalam
    DATASET_NAME = "translation-ml"
    POSITIVE_LABEL = "പോസിറ്റീവ്"
    NEGATIVE_LABEL = "നെഗറ്റീവ്"
    SENTENCE_TEMPLATE = f"\nചോദ്യം: ഈ വാചകം {POSITIVE_LABEL} ആണോ {NEGATIVE_LABEL} ആണോ?\nഉത്തരം:"


class IndicSentiment_mr(IndicSentiment): # Marathi
    DATASET_NAME = "translation-mr"
    POSITIVE_LABEL = "सकारात्मक"
    NEGATIVE_LABEL = "नकारात्मक"
    SENTENCE_TEMPLATE = f"\nप्रश्न: हे वाक्य {POSITIVE_LABEL} आहे की {NEGATIVE_LABEL}?\nउत्तर:"


class IndicSentiment_or(IndicSentiment): # Oriya
    DATASET_NAME = "translation-or"
    POSITIVE_LABEL = "ସକାରାତ୍ମକ"
    NEGATIVE_LABEL = "ନକାରାତ୍ମକ"
    SENTENCE_TEMPLATE = f"\nପ୍ରଶ୍ନ: ଏହି ବାକ୍ୟଟି {POSITIVE_LABEL} କି {NEGATIVE_LABEL}?\nଉତ୍ତର:"


class IndicSentiment_pa(IndicSentiment): # Punjabi
    DATASET_NAME = "translation-pa"
    POSITIVE_LABEL = "ਸਕਾਰਾਤਮਕ"
    NEGATIVE_LABEL = "ਨਕਾਰਾਤਮਕ"
    SENTENCE_TEMPLATE = f"\nਸਵਾਲ: ਕੀ ਇਹ ਵਾਕ {POSITIVE_LABEL} ਹੈ ਜਾਂ {NEGATIVE_LABEL}?\nਜਵਾਬ:"


class IndicSentiment_ta(IndicSentiment): # Tamil
    DATASET_NAME = "translation-ta"
    POSITIVE_LABEL = "நேர்மறை"
    NEGATIVE_LABEL = "எதிர்மறை"
    SENTENCE_TEMPLATE = f"\nகேள்வி: இந்த வாக்கியம் {POSITIVE_LABEL}யா {NEGATIVE_LABEL}யா?\nபதில்:"


class IndicSentiment_te(IndicSentiment): # Telugu
    DATASET_NAME = "translation-te"
    POSITIVE_LABEL = "సానుకూల"
    NEGATIVE_LABEL = "ప్రతికూల"
    SENTENCE_TEMPLATE = f"\nప్రశ్న: ఈ వాక్యం {POSITIVE_LABEL}మా లేదా {NEGATIVE_LABEL}మా?\nసమాధానం:"


class IndicSentiment_ur(IndicSentiment): # Urdu
    DATASET_NAME = "translation-ur"
    POSITIVE_LABEL = "مثبت"
    NEGATIVE_LABEL = "منفی"
    SENTENCE_TEMPLATE = f"\nسوال: کیا یہ جملہ مثبت ہے یا منفی؟\nجواب:"


LANGS = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]

LANG_CLASSES = [
    IndicSentiment_as,
    IndicSentiment_bn,
    IndicSentiment_en,
    IndicSentiment_gu,
    IndicSentiment_hi,
    IndicSentiment_kn,
    IndicSentiment_ml,
    IndicSentiment_mr,
    IndicSentiment_or,
    IndicSentiment_pa,
    IndicSentiment_ta,
    IndicSentiment_te,
    IndicSentiment_ur
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"indicsentiment_{lang}"] = lang_class
    return tasks
