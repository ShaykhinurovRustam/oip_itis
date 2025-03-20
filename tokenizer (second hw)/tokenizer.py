import re
import pymorphy2
import nltk
import shutil

from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nltk.download('stopwords')


class Tokenizer:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))

    def extract_tokens(self, file_path: str) -> set[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            html = f.read()

        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator=' ')

        tokens_raw = re.findall(r'\b[а-яё]+\b', text, flags=re.IGNORECASE)
        tokens = set()

        for token in tokens_raw:
            token_lower = token.lower()
            if token_lower.isalpha() and token_lower not in self.stop_words:
                tokens.add(token_lower)
        
        return tokens

    def group_by_lemma(self, tokens: set[str]) -> dict[str, set[str]]:
        lemmas = defaultdict(set)

        for token in tokens:
            parsed = self.morph.parse(token)
            if parsed:
                lemma = parsed[0].normal_form
                lemmas[lemma].add(token)

        return lemmas

    def process_file(self, file_path: str, num: int) -> None:
        tokens = self.extract_tokens(file_path)
        lemmas = self.group_by_lemma(tokens)

        tokens_file = f'results/tokens_{num}.txt'
        lemmas_file = f'results/lemmas_{num}.txt'

        with open(tokens_file, 'w', encoding='utf-8') as f:
            for token in sorted(tokens):
                f.write(token + '\n')

        with open(lemmas_file, 'w', encoding='utf-8') as f:
            for lemma in sorted(lemmas):
                f.write(lemma + ' ' + ' '.join(sorted(lemmas[lemma])) + '\n')


def main():
    tokenizer = Tokenizer()

    for num in range(1, 101):
        tokenizer.process_file(f'data/page-{num}.html', num)
        
    shutil.make_archive('results', 'zip', 'results')


if __name__ == '__main__':
    main()
