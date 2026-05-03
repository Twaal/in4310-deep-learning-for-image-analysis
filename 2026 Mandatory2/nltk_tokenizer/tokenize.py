import functools

from nltk_tokenizer.destructive import NLTKWordTokenizer
from nltk_tokenizer.punkt import PunktTokenizer

_treebank_word_tokenizer = NLTKWordTokenizer()


def word_tokenize(text, language="english", preserve_line=False):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently an improved :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into words
    :type text: str
    :param language: the model name in the Punkt corpus
    :type language: str
    :param preserve_line: A flag to decide whether to sentence tokenize the text or not.
    :type preserve_line: bool
    """
    sentences = [text] if preserve_line else sent_tokenize(text, language)
    return [
        token for sent in sentences for token in _treebank_word_tokenizer.tokenize(sent)
    ]


def sent_tokenize(text, language="english"):
    """
    Return a sentence-tokenized copy of *text*,
    using NLTK's recommended sentence tokenizer
    (currently :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
    tokenizer = _get_punkt_tokenizer(language)
    return tokenizer.tokenize(text)


@functools.lru_cache
def _get_punkt_tokenizer(language="english"):
    """
    A constructor for the PunktTokenizer that utilizes
    a lru cache for performance.

    :param language: the model name in the Punkt corpus
    :type language: str
    """
    return PunktTokenizer(language)
