"""
Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages
https://arxiv.org/abs/2212.05409

IndicCOPA is a manually translation of the COPA test set into 18 Indic languages

Dataset page: https://huggingface.co/datasets/ai4bharat/IndicCOPA/
"""
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, f1_score


_CITATION = """
@article{Doddapaneni2022towards,
  title={Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
  author={Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
  journal={ArXiv},
  year={2022},
  volume={abs/2212.05409}
}
"""


class IndicXParaphrase(Task):
    VERSION = 0
    DATASET_PATH = "ai4bharat/IndicXParaphrase"
    DATASET_NAME = None
    YES_LABEL = None
    NO_LABEL = None
    SENTENCE_TEMPLATE = None # Sentence 1: {}\nSentence 2: {}\nQuestion: Do both sentences mean the same thing?\nAnswer:

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return self.SENTENCE_TEMPLATE.format(
            doc["sentence1"],
            doc["sentence2"]
        )

    def doc_to_target(self, doc):
        return " {}".format(doc["label"])

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, f" {self.YES_LABEL}")
        ll_no, _ = rf.loglikelihood(ctx, f" {self.NO_LABEL}")
        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]
        pred = ll_yes > ll_no
        return {
            "acc": pred == gold,
            "f1": (gold, pred),
        }

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score}
    

class IndicXParaphrase_as(IndicXParaphrase): # Assamese
    DATASET_NAME = "as"
    YES_LABEL = "হয়"
    NO_LABEL = "নহয়"
    SENTENCE_TEMPLATE = "বাক্য 1: {}\nবাক্য 2: {}\nপ্ৰশ্ন: দুয়োটা বাক্যৰ অৰ্থ একে নেকি?\nউত্তৰ:"


class IndicXParaphrase_bn(IndicXParaphrase): # Bengali
    DATASET_NAME = "bn"
    YES_LABEL = "হ্যাঁ"
    NO_LABEL = "না"
    SENTENCE_TEMPLATE = "বাক্য 1: {}\nবাক্য 2: {}\nপ্রশ্ন: উভয় বাক্যই কি একই জিনিস বোঝায়?\nউত্তর:"


class IndicXParaphrase_gu(IndicXParaphrase): # Gujarati
    DATASET_NAME = "gu"
    YES_LABEL = "હા"
    NO_LABEL = "ના"
    SENTENCE_TEMPLATE = "વાક્ય 1: {}\nવાક્ય 2: {}\nપ્રશ્ન: શું બંને વાક્યોનો અર્થ એક જ છે?\nજવાબ:"


class IndicXParaphrase_hi(IndicXParaphrase): # Hindi
    DATASET_NAME = "hi"
    YES_LABEL = "हाँ"
    NO_LABEL = "नहीं"
    SENTENCE_TEMPLATE = "वाक्य 1: {}\nवाक्य 2: {}\nप्रश्न: क्या दोनों वाक्यों का मतलब एक ही है?\nउत्तर:"


class IndicXParaphrase_kn(IndicXParaphrase): # Kannada
    DATASET_NAME = "kn"
    YES_LABEL = "ಹೌದು"
    NO_LABEL = "ಇಲ್ಲ"
    SENTENCE_TEMPLATE = "ವಾಕ್ಯ 1: {}\nವಾಕ್ಯ 2: {}\nಪ್ರಶ್ನೆ: ಎರಡೂ ವಾಕ್ಯಗಳು ಒಂದೇ ಅರ್ಥವನ್ನು ಹೊಂದಿದೆಯೇ?\nಉತ್ತರ:"


class IndicXParaphrase_ml(IndicXParaphrase): # Malayalam
    DATASET_NAME = "ml"
    YES_LABEL = "അതെ"
    NO_LABEL = "ഇല്ല"
    SENTENCE_TEMPLATE = "വാക്യം 1: {}\nവാക്യം 2: {}\nചോദ്യം: രണ്ട് വാക്യങ്ങളും അർത്ഥമാക്കുന്നത് ഒരേ കാര്യമാണോ?\nഉത്തരം:"


class IndicXParaphrase_mr(IndicXParaphrase): # Marathi
    DATASET_NAME = "mr"
    YES_LABEL = "होय"
    NO_LABEL = "नाही"
    SENTENCE_TEMPLATE = "वाक्य 1: {}\nवाक्य 2: {}\nप्रश्न: दोन्ही वाक्यांचा अर्थ एकच आहे का?\nउत्तर:"


class IndicXParaphrase_or(IndicXParaphrase): # Oriya
    DATASET_NAME = "or"
    YES_LABEL = "ହଁ"
    NO_LABEL = "ନା"
    SENTENCE_TEMPLATE = "ବାକ୍ୟ 1: {}\nସେଣ୍ଟେନ୍ସ 2: {}\nପ୍ରଶ୍ନ: ଉଭୟ ବାକ୍ୟର ସମାନ ଅର୍ଥ ଅଛି କି?\nଉତ୍ତର:"


class IndicXParaphrase_pa(IndicXParaphrase): # Punjabi
    DATASET_NAME = "pa"
    YES_LABEL = "ਹਾਂ"
    NO_LABEL = "ਨਹੀਂ"
    SENTENCE_TEMPLATE = "ਵਾਕ 1: {}\nਵਾਕ 2: {}\nਸਵਾਲ: ਕੀ ਦੋਵੇਂ ਵਾਕਾਂ ਦਾ ਅਰਥ ਇੱਕੋ ਜਿਹਾ ਹੈ?\nਜਵਾਬ:"


class IndicXParaphrase_te(IndicXParaphrase): # Telugu
    DATASET_NAME = "te"
    YES_LABEL = "అవును"
    NO_LABEL = "సంఖ్య"
    SENTENCE_TEMPLATE = "వాక్యం 1: {}\nవాక్యం 2: {}\nప్రశ్న: రెండు వాక్యాల అర్థం ఒకటేనా?\nసమాధానం:"


LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "te"]

LANG_CLASSES = [
    IndicXParaphrase_as,
    IndicXParaphrase_bn,
    IndicXParaphrase_gu,
    IndicXParaphrase_hi,
    IndicXParaphrase_kn,
    IndicXParaphrase_ml,
    IndicXParaphrase_mr,
    IndicXParaphrase_or,
    IndicXParaphrase_pa,
    IndicXParaphrase_te
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"indicxparaphrase_{lang}"] = lang_class
    return tasks
