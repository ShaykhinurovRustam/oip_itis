import os
import re
import pymorphy2


class BooleanSearch:
    def __init__(self, lemmas_folder: str, data_folder: str) -> None:
        self.lemmas_folder = lemmas_folder
        self.data_folder = data_folder
        self.morph = pymorphy2.MorphAnalyzer()
        self.index = self._build_inverted_index()
        self.docs = self._get_documents()
        self.tokens = []
        self.current = 0

    def boolean_search(self, query: str) -> set[str]:
        self.tokens = self._tokenize_query(query)
        self.current = 0
        result = self._parse_or()
        
        if self.current != len(self.tokens):
            raise Exception
        
        return result

    def write_index(self, output_filename: str) -> None:
        with open(output_filename, 'w', encoding='utf-8') as f:
            for term, docs in sorted(self.index.items()):
                doc_list = ', '.join(sorted(docs))
                f.write(f'{term}: {doc_list}\n')

    def _build_inverted_index(self) -> dict[str, set[str]]:
        index = {}

        for filename in os.listdir(self.lemmas_folder):
            doc_id = filename.split('_')[1].split('.')[0]
            doc_name = f'page-{doc_id}.html'
            file_path = os.path.join(self.lemmas_folder, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    parts = line.split()

                    lemma = parts[0].lower()
                    if lemma not in index:
                        index[lemma] = set()

                    index[lemma].add(doc_name)

        return index

    def _get_documents(self) -> set[str]:
        docs = set()

        for filename in os.listdir(self.lemmas_folder):
            doc_id = filename.split('_')[1].split('.')[0]
            doc_name = f'page-{doc_id}.html'
            docs.add(doc_name)

        return docs

    def _tokenize_query(self, query: str) -> list[str]:
        tokens = re.findall(r'\(|\)|AND|OR|NOT|[\wа-яА-ЯёЁ]+', query, flags=re.IGNORECASE)
        return tokens

    def _lemmatize_token(self, token: str) -> str:
        if token.upper() in {'AND', 'OR', 'NOT'} or token in {'(', ')'}:
            return token
        
        parsed = self.morph.parse(token)
        
        return parsed[0].normal_form if parsed else token.lower()

    def _parse_or(self) -> set[str]:
        result = self._parse_and()
        
        while self.current < len(self.tokens) and self.tokens[self.current].upper() == 'OR':
            self.current += 1
            right = self._parse_and()
            result = result | right
        
        return result

    def _parse_and(self) -> set[str]:
        result = self._parse_not()
        
        while self.current < len(self.tokens) and self.tokens[self.current].upper() == 'AND':
            self.current += 1
            right = self._parse_not()
            result = result & right
        
        return result

    def _parse_not(self) -> set[str]:
        if self.current < len(self.tokens) and self.tokens[self.current].upper() == 'NOT':
            self.current += 1
            operand = self._parse_not()
            
            return self.docs - operand

        return self._parse_atom()

    def _parse_atom(self) -> set[str]:
        if self.current >= len(self.tokens):
            raise Exception
    
        token = self.tokens[self.current]
        if token == '(':
            self.current += 1
            result = self._parse_or()
            
            if self.current >= len(self.tokens) or self.tokens[self.current] != ')':
                raise Exception
            
            self.current += 1
            return result

        self.current += 1
        lemma = self._lemmatize_token(token)
        return self.index.get(lemma, set())


if __name__ == '__main__':
    bs = BooleanSearch(lemmas_folder='lemmas', data_folder='data')
    bs.write_index('inverted_index.txt')
    
    while True:
        query = input('Запрос: ')
        try:
            result_docs = bs.boolean_search(query)
            print('Найденные документы: ' + ', '.join(sorted(result_docs)))
        except Exception:
            print('Ошибка в запросе')
