"""
Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages
https://arxiv.org/abs/2212.05409

IndicCOPA is a manually translation of the COPA test set into 18 Indic languages

Dataset page: https://huggingface.co/datasets/ai4bharat/IndicCOPA/
"""
from .superglue import Copa


_CITATION = """
@article{Doddapaneni2022towards,
  title={Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
  author={Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
  journal={ArXiv},
  year={2022},
  volume={abs/2212.05409}
}
"""

class IndicCopa(Copa):
    VERSION = 0
    DATASET_PATH = "ai4bharat/IndicCOPA"
    DATASET_NAME = None
    CAUSE = "because"
    EFFECT = "therefore"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        # Drop the period
        connector = {
            "cause": self.CAUSE,
            "effect": self.EFFECT,
        }[doc["question"]]
        return doc["premise"].strip()[:-1] + f" {connector}"


class IndicCopa_as(IndicCopa):
    DATASET_NAME = "translation-as"
    CAUSE = "কাৰণ"
    EFFECT = "সেয়েহে"

class IndicCopa_bn(IndicCopa):
    DATASET_NAME = "translation-bn"
    CAUSE = "কারণ"
    EFFECT = "তাই"

class IndicCopa_en(IndicCopa):
    DATASET_NAME = "translation-en"
    CAUSE = "because"
    EFFECT = "therefore"

class IndicCopa_gom(IndicCopa):
    DATASET_NAME = "translation-gom"
    CAUSE = "कारण"
    EFFECT = "ताकालागून"
    
class IndicCopa_gu(IndicCopa):
    DATASET_NAME = "translation-gu"
    CAUSE = "કારણ કે"
    EFFECT = "તેથી"

class IndicCopa_hi(IndicCopa):
    DATASET_NAME = "translation-hi"
    CAUSE = "क्योंकि"
    EFFECT = "इसलिए"

class IndicCopa_kn(IndicCopa):
    DATASET_NAME = "translation-kn"
    CAUSE = "ಏಕೆಂದರೆ"
    EFFECT = "ಆದ್ದರಿಂದ"

class IndicCopa_mai(IndicCopa):
    DATASET_NAME = "translation-mai"
    CAUSE = "किएक तँ"
    EFFECT = "एहि लेल"

class IndicCopa_ml(IndicCopa):
    DATASET_NAME = "translation-ml"
    CAUSE = "കാരണം"
    EFFECT = "അതുകൊണ്ടു"

class IndicCopa_mr(IndicCopa):
    DATASET_NAME = "translation-mr"
    CAUSE = "कारण"
    EFFECT = "म्हणून"

class IndicCopa_ne(IndicCopa):
    DATASET_NAME = "translation-ne"
    CAUSE = "किनभने"
    EFFECT = "त्यसैले"

class IndicCopa_or(IndicCopa):
    DATASET_NAME = "translation-or"
    CAUSE = "କାରଣ"
    EFFECT = "ତେଣୁ"

class IndicCopa_pa(IndicCopa):
    DATASET_NAME = "translation-pa"
    CAUSE = "ਕਿਉਂਕਿ"
    EFFECT = "ਇਸ ਲਈ"

class IndicCopa_sa(IndicCopa):
    DATASET_NAME = "translation-sa"
    CAUSE = "यतः"
    EFFECT = "अतएव"

class IndicCopa_sat(IndicCopa):
    DATASET_NAME = "translation-sat"
    CAUSE = "ᱪᱮᱫᱟᱜ ᱥᱮ"
    EFFECT = "ᱚᱱᱟᱛᱮ"

class IndicCopa_sd(IndicCopa):
    DATASET_NAME = "translation-sd"
    CAUSE = "ڇاڪاڻ ته"
    EFFECT = "تنهن ڪري"

class IndicCopa_ta(IndicCopa):
    DATASET_NAME = "translation-ta"
    CAUSE = "ஏனெனில்"
    EFFECT = "எனவே"

class IndicCopa_te(IndicCopa):
    DATASET_NAME = "translation-te"
    CAUSE = "ఎందుకంటే"
    EFFECT = "అందువలన"

class IndicCopa_ur(IndicCopa):
    DATASET_NAME = "translation-ur"
    CAUSE = "کیونکہ"
    EFFECT = "لہذا"


LANGS = ["as", "bn", "en", "gom", "gu", "hi", "kn", "mai", "ml", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"]

LANG_CLASSES = [
    IndicCopa_as,
    IndicCopa_bn,
    IndicCopa_en,
    IndicCopa_gom,
    IndicCopa_gu,
    IndicCopa_hi,
    IndicCopa_kn,
    IndicCopa_mai,
    IndicCopa_ml,
    IndicCopa_mr,
    IndicCopa_ne,
    IndicCopa_or,
    IndicCopa_pa,
    IndicCopa_sa,
    IndicCopa_sat,
    IndicCopa_sd,
    IndicCopa_ta,
    IndicCopa_te,
    IndicCopa_ur
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"indiccopa_{lang}"] = lang_class
    return tasks
