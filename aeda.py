import random
import konlpy
from konlpy.tag import Okt
from koeda import AEDA


SPACE_TOKEN = "\u241F"


def replace_space(text: str) -> str:
    return text.replace(" ", SPACE_TOKEN)


def revert_space(text: list) -> str:
    clean = " ".join("".join(text).replace(SPACE_TOKEN, " ").split()).strip()
    return clean



def aeda(data: str, p: float) -> str:
    punctuations = ('.', ',', '!', '?', ';', ':')
    punc_ratio = 0.3
    #morpheme_analyzer = 'Okt'
    if p is None:
        p = punc_ratio

    split_words = Okt(replace_space(data))
    words = Okt(data)

    new_words = []
    q = random.randint(1, int(p * len(words) + 1))
    qs_list = [
        index
        for index in range(len(split_words))
        if split_words[index] != SPACE_TOKEN
    ]
    qs = random.sample(qs_list, q)

    for j, word in enumerate(split_words):
        if j in qs:
            new_words.append(SPACE_TOKEN)
            new_words.append(
                punctuations[random.randint(0, len(punctuations) - 1)]
            )
            new_words.append(SPACE_TOKEN)
            new_words.append(word)
        else:
            new_words.append(word)

    augmented_sentences = revert_space(new_words)

    return augmented_sentences
