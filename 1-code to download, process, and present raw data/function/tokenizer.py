import six
import unicodedata
import collections
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:18:04 2020

@author: zhong
"""
###############################################################################
'function'

def word_segment(input_str,vocab):  
    orig_tokens = list(input_str)
    def find_end(tokens,remain):
        for end in range(1,len(remain)+1):
            token = ''.join(remain[ : end]) 
            if token in vocab:
                yield np.append(tokens,token) ,remain[end:]            
    toks = []
    def loop(generator):
        for tok, rim in generator:
            loop(find_end(tok,rim))
            if len(rim) == 0:
                toks.append(tok)
    loop(find_end([],orig_tokens))    
    min_len = min([len(item) for item in toks])    
    candidate = [list(item) for item in toks if len(item) == min_len]
    return sorted(candidate,key = lambda words: np.std([len(word) for word in words]))[0]


###############################################################################
def word_clean(text,do_lower_case = True):
    """Tokenizes a piece of text."""
    
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    text = convert_to_unicode(text)
    
    """Performs invalid character removal and whitespace cleanup on text."""
    text = _clean_text(text)
    
    """Adds whitespace around any CJK character."""
    text = _tokenize_chinese_chars(text)
        
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    
    for token in orig_tokens:
        """transfer to lower case"""
        if do_lower_case:
            token = token.lower()
        """Strips accents from a piece of text."""
        """ orčpžsíáýd to orcpzsiayd"""
        token = _run_strip_accents(token)
        split_tokens.extend(_run_removel_on_punc(token))
        
    output_tokens = whitespace_tokenize(" ".join(split_tokens))

    return output_tokens

###############################################################################
def word_clean_stop_words(text,do_lower_case = True):
    """Tokenizes a piece of text."""
    
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    text = convert_to_unicode(text)
    
    """Performs invalid character removal and whitespace cleanup on text."""
    text = _clean_text(text)
    
    """Adds whitespace around any CJK character."""
    text = _tokenize_chinese_chars(text)
        
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    
    for token in orig_tokens:
        """transfer to lower case"""
        if do_lower_case:
            token = token.lower()
        """Strips accents from a piece of text."""
        """ orčpžsíáýd to orcpzsiayd"""
        token = _run_strip_accents(token)
        split_tokens.extend(_run_removel_on_punc(token))
        
    output_tokens = whitespace_tokenize(" ".join(split_tokens))

    'stop words'
    """delete all stop words from dictionary"""
    for token in output_tokens:
        if token in stopwords.words('english'):
            output_tokens.remove(token)

    return output_tokens

###############################################################################
'class'
class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, stop_words  = False):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.stop_words = stop_words
        
    def tokenize(self, text):
        """Tokenizes a piece of text."""
        
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        text = convert_to_unicode(text)
        
        """Performs invalid character removal and whitespace cleanup on text."""
        text = _clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        """Adds whitespace around any CJK character."""
        text = _tokenize_chinese_chars(text)
        
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            """transfer to lower case"""
            if self.do_lower_case:
                token = token.lower()
            """Strips accents from a piece of text."""
            """ orčpžsíáýd to orcpzsiayd"""
            token = _run_strip_accents(token)
            split_tokens.extend(_run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        
        'stop words'
        """delete all stop words from dictionary"""
        if self.stop_words:
            for token in output_tokens:
                if token in stopwords.words('english'):
                    output_tokens.remove(token)
        
        return output_tokens
    
###############################################################################
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

###############################################################################

def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _clean_text(text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

###############################################################################

def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

def _tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
    
###############################################################################

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

###############################################################################

def _run_strip_accents(text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def _run_removel_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]

def _run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]

###############################################################################
class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        
        
    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

###############################################################################

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab