import logging

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
# nltk.data.path.insert(0, '/data/share/kingxu/nltk_data')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def process_msg(obj):
    """
    obj是一个tree_diff的实例,是已经json.loads的一条数据
    """
    msg = obj['msg'].lower()
    file_names = get_diff_file_names(obj['diff'])
    # logger.info(file_names)
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(pattern='\w+')
    words = []
    for word in tokenizer.tokenize(msg):
        if word in file_names:
            words.append('FILE')
        elif '_' in word:
            words.extend(word.split('_'))
        elif word.isdigit():
            words.append('NUMBER')
        else:
            words.append(word)
    lemmatized_words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words if w.isalnum()]
    return lemmatized_words


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ,
                'N': wordnet.NOUN,
                'V': wordnet.VERB,
                'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def get_diff_file_names(diff_str):
    file_names = []
    for line in diff_str.splitlines():
        if line.startswith('diff --git '):
            for p in line.split(' ')[2:]:
                file_names.append(p.rsplit('/', 1)[-1].split('.', 1)[0].lower())
    return set(file_names)
