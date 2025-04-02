import os
import math
import collections


class TfidfProcessor:
    def __init__(
        self, 
        data_dir: str, 
        results_dir: str,
        output_dir_tokens: str, 
        output_dir_lemmas: str,
    ) -> None:
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.output_dir_tokens = output_dir_tokens
        self.output_dir_lemmas = output_dir_lemmas

        os.makedirs(self.output_dir_tokens, exist_ok=True)
        os.makedirs(self.output_dir_lemmas, exist_ok=True)

        self.token_df = collections.Counter()
        self.lemma_df = collections.Counter()

    def _compute_global_df(self) -> None:
        for i in range(1, 101):
            token_file = os.path.join(self.results_dir, f'tokens_{i}.txt')
            lemma_file = os.path.join(self.results_dir, f'lemmas_{i}.txt')
            
            with open(token_file, 'r', encoding='utf-8') as f:
                tokens = f.read().splitlines()
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.token_df[token] += 1

            with open(lemma_file, 'r', encoding='utf-8') as f:
                lemma_lines = f.read().splitlines()
            lemmas_in_doc = set()
            for line in lemma_lines:
                parts = line.split()
                if parts:
                    lemma = parts[0]
                    lemmas_in_doc.add(lemma)
            for lemma in lemmas_in_doc:
                self.lemma_df[lemma] += 1

    def _process_document(self, doc_index: int) -> None:
        token_file = os.path.join(self.results_dir, f'tokens_{doc_index}.txt')
        with open(token_file, 'r', encoding='utf-8') as f:
            tokens = f.read().splitlines()
        token_tf = collections.Counter(tokens)

        lemma_file = os.path.join(self.results_dir, f'lemmas_{doc_index}.txt')
        with open(lemma_file, 'r', encoding='utf-8') as f:
            lemma_lines = f.read().splitlines()
        
        lemma_tf_counts = collections.Counter()
        
        total_lemmas = 0
        for line in lemma_lines:
            parts = line.split()
            lemma = parts[0]
            
            count = len(parts) - 1
            lemma_tf_counts[lemma] = count
            total_lemmas += count

        # idf = ln(общ. кол-во документов / df(токена)), tf-idf = tf * idf
        tokens_output_lines = []
        for token, count in token_tf.items():
            df_value = self.token_df.get(token, 1)
            idf_val = math.log(100 / df_value)
            tfidf = count * idf_val
            tokens_output_lines.append(f'{token} {idf_val} {tfidf}\n')

        output_tokens_file = os.path.join(self.output_dir_tokens, f'tfidf_tokens_{doc_index}.txt')
        with open(output_tokens_file, 'w', encoding='utf-8') as f:
            f.writelines(tokens_output_lines)

        # tf = (количество вхождений леммы / общее число лемматизированных токенов),
        # idf = ln(общ. кол-во документов / df(леммы)), tf-idf = tf * idf
        lemmas_output_lines = []
        for lemma, count in lemma_tf_counts.items():
            df_value = self.lemma_df.get(lemma, 1)
            idf_val = math.log(100 / df_value)
            tf = count / total_lemmas if total_lemmas > 0 else 0
            tfidf = tf * idf_val
            lemmas_output_lines.append(f'{lemma} {idf_val} {tfidf}\n')

        output_lemmas_file = os.path.join(self.output_dir_lemmas, f'tfidf_lemmas_{doc_index}.txt')
        with open(output_lemmas_file, 'w', encoding='utf-8') as f:
            f.writelines(lemmas_output_lines)

    def process_all_documents(self) -> None:
        self._compute_global_df()
        for i in range(1, 101):
            self._process_document(i)


if __name__ == '__main__':
    processor = TfidfProcessor(
        data_dir='data', 
        results_dir='results',
        output_dir_tokens='tfidf_tokens', 
        output_dir_lemmas='tfidf_lemmas',
    )
    processor.process_all_documents()