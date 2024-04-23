import json
import os
import pickle
from itertools import chain
from typing import Iterable

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api._v2.keras import Model
from shap import Explainer
from shap.maskers import Text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from stanfordcorenlp import StanfordCoreNLP
from transformers import RobertaTokenizerFast

import EASTER_en
from EASTER_en import DataGenerator
from EASTER_en import load_model
from config import *

std_p_value = 0.8

nlp = StanfordCoreNLP(STANFORD_CORE_NLP_PATH)
class OpinionExtractor:
    def __init__(self, sentiment_file: str,shap_file: str,):
        self.shap_file = shap_file
        self.sentiment_file = sentiment_file
        self.pos_map = {
            "NNS": "n", "NNPS": "n", "NN": "n", "NNP": "n",
            "VBN": "v", "VB": "v", "VBD": "v", "VBZ": "v","VBP": "v", "VBG": "v",
            "JJR": "a", "JJS": "a", "JJ": "a",
            "RBS": "r", "RB": "r", "RP": "r", "WRB": "r", "RBR": "r",
        }
        self.words_lead_to_clause = [""]
        self.pos_tags_lead_to_clause = ["CC", ","]
        self.clause_label_list = ['S']
        self.phrase_label_list= ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ',
                                'LST', 'NAC', 'NP', 'NX', 'PP',
                                'PRN', 'PRT', 'QP', 'RRC', 'UCP',
                                'VP', 'WHADJP', 'WHAVP', 'WHNP','WHPP','X']
        # load sentiwordnet
        sentiwordnet = dict()
        with open(f'{BASE_DIR}/data/dict/SentiWordNet_3.0.0.txt', 'r', encoding='utf-8') as f:
            for line in map(lambda x: x.split('\t'), f.read().splitlines()[26:-1]):
                for word_number in map(lambda x: x.split('#'), line[4].split(' ')):
                    sentiwordnet.setdefault(word_number[0], {}).setdefault(line[0], []).append([int(word_number[1]), float(line[2]), float(line[3])])
        self.sentiwordnet = {word: {pos: score_list for pos, score_list in info.items()} for word, info in sentiwordnet.items()}
        # laod dataset
        self.data_list = self._load_data()

    def get_texts(self):
        """
        Get the input text.
        :return:
        """
        return [data['sentence'] for data in self.data_list]

    def _get_priori_value(self, word: str):
        """
        Get the priori sentiment value.

        :param word: the word
        :return: return -1 if not exists
        """
        score_list = [score for pos_scores in self.sentiwordnet.get(word, {}).values() for score in pos_scores]
        score = sum(abs(score[1] + score[2]) / score[0] for score in score_list) if score_list else -1
        return score

    def _get_priori_value_with_pos(self, word: str, pos: str):
        """
        Considering the POS, get the priori sentiment value.
        score = Σi=1 to n (PosScore+NegScore)/sense_rank
                n is the number of senses of a word given its POS

        :param word: the word
        :param pos: the pos(part of speech) of the word
        :return: return -1 if not exists
        """
        score_list = self.sentiwordnet.get(word, {}).get(self.pos_map.get(pos, ''), [])
        score = sum(abs(score[1] + score[2]) / score[0] for score in score_list)  if score_list else -1
        return score

    def _get_priori_value_with_senti(self, word: str, pos: str, pola: int):
        """
        Considering the POS and the text sentiment, get the priori sentiment value.

        :param word: the word
        :param pos: the pos(part of speech) of the word
        :param pola: the sentiment of the text where the word is located.
        :return: return -1 if not exists
        """
        score_list = self.sentiwordnet.get(word, {}).get(self.pos_map.get(pos, ''), [])
        if len(score_list)==0:
            score = -1
        else:
            if pola==1:
                score = sum(abs(score[1]) / score[0] for score in score_list)
            elif pola==-1:
                score = sum(abs(score[2]) / score[0] for score in score_list)
            elif pola==0:
                score = sum(abs(score[1]+score[2]) / score[0] for score in score_list)
        return score

    def _is_potential_aspect(self,priori_value: float, pos: str):
        """
        To determine if it is a potential aspect,
        the conditions for being a potential aspect are:
            1) There is no prior sentiment value or the prior sentiment value is 0,
            2) It is a noun.
        :param priori_value: the prior sentiment value
        :param pos: the pos of the word
        :return:
        """
        return priori_value<=0 and self.pos_map.__contains__(pos) and self.pos_map[pos]=="n"

    def _get_clause_with_comma(self,data):
        '''
        Split text based on commas
        '''
        clauses = []
        current_clause = []
        for item in data['words_info']:
            if item[1] == ",":
                if current_clause:
                    clauses.append(current_clause)
                    current_clause = []
            current_clause.append(item)
        if current_clause:
            clauses.append(current_clause)
        return clauses

    def _get_clause_with_constituency(self, data):
        '''
        Split text based on constituency analysis
        '''
        if len(data['words_info']) == 0:
            return data['words_info']
        constituency_tree_str = data['constituency']
        word_list = [ x[0] for x in data['words_info'] ]
        start_index = 0
        for i in range(len(word_list)):
            if start_index >= len(constituency_tree_str):
                break
            word = "-LRB-" if word_list[i] == "(" else "-RRB-" if word_list[i] == ")" else word_list[i]
            head_str = constituency_tree_str[:start_index]
            next_rpare_index = constituency_tree_str.find(")",constituency_tree_str.find("(",start_index))
            middle_str = constituency_tree_str[start_index:next_rpare_index+1]
            tail_str = constituency_tree_str[next_rpare_index+1:]
            middle_str = middle_str.replace(f"{word})",f"{i})",1)
            constituency_tree_str = head_str + middle_str + tail_str
            start_index = len(head_str + middle_str)
        tree = nltk.Tree.fromstring(constituency_tree_str)
        clause_start_index_set = set()
        phrase_start_index_set = set()
        for subtree in tree.subtrees():
            try:
                start_index = int(subtree.leaves()[0])
            except ValueError:
                continue
            if subtree.label() in self.clause_label_list:
                clause_start_index_set.add(start_index)
            elif subtree.label() in self.phrase_label_list:
                phrase_start_index_set.add(start_index)
        clauses = []
        current_clause = []
        start_index_set = clause_start_index_set
        if len(clause_start_index_set) == 0:
            start_index_set = phrase_start_index_set
        for i in range(len(data['words_info'])):
            item = data['words_info'][i]
            # if i in start_index_set:
            if (i in start_index_set) \
                    or (item[1] in self.pos_tags_lead_to_clause) \
                    or (item[0] in self.words_lead_to_clause):
                if current_clause:
                    clauses.append(current_clause)
                    current_clause = []
            current_clause.append(item)
        if current_clause:
            clauses.append(current_clause)

        return clauses

    def extract_clause_opinion(self, std_p: float):
        '''
        Select one opinion for each clause
        :param std_p: Coefficient of standard deviation, default value is 0.8
        The returned opinion has three attributes:
          'actual_index': The true index range of opinion,
          'is_potential_aspect': is it a potential aspect
        '''
        import numpy as np
        opinion_list = []
        for data in self.data_list:
            senti_pola = data['sentiment']
            shap_values = [x[3] for x in data['words_info']]
            standard = np.mean(shap_values) + std_p * np.std(shap_values)
            clause_words_info_list = self._get_clause_with_constituency(data)
            one_text_opinion_list = []
            max_clause_opinion = None
            for clause_words_info in clause_words_info_list:
                clause_words_info = sorted(filter(lambda x: self.pos_map.get(x[1], None) is not None, clause_words_info),
                                    key=lambda x: x[3], reverse=True)
                if len(clause_words_info)==0:
                    continue
                clause_opinion = clause_words_info[0]
                if max_clause_opinion is None or clause_opinion[3]>max_clause_opinion[3]:
                    max_clause_opinion = clause_opinion
                if clause_opinion[3]<standard:
                    continue
                opinion = {'actual_index': [clause_opinion[4][0], clause_opinion[4][-1] + 1],
                           'is_potential_aspect': self._is_potential_aspect(
                               priori_value=self._get_priori_value(word=clause_opinion[0]),
                               pos=clause_opinion[1])}
                one_text_opinion_list.append(opinion)

            if len(one_text_opinion_list)==0:
                if max_clause_opinion is not None:
                    opinion = {'actual_index': [max_clause_opinion[4][0], max_clause_opinion[4][-1] + 1],
                               'is_potential_aspect': self._is_potential_aspect(
                                   priori_value=self._get_priori_value(word=max_clause_opinion[0]),
                                   pos=max_clause_opinion[1])}

                else:
                    opinion = {'actual_index': [-1, -1],
                               'is_potential_aspect': False}
                one_text_opinion_list.append(opinion)
            opinion_list.append(one_text_opinion_list)
        return opinion_list

    def extract_top_k_opinion(self, k: int):
        """
        Extract the top k highest shap value as opinion
        """
        opinion_list = []
        for data in self.data_list:
            senti_pola = data['sentiment']
            one_text_opinion_list = []
            words_info = sorted(filter(lambda x: self.pos_map.get(x[1], None) is not None, data['words_info']), key=lambda x: x[3], reverse=True)
            if len(words_info) == 0:
                opinion = {'actual_index': [-1, -1],
                           'is_potential_aspect': False}
                one_text_opinion_list.append( opinion )
            else:
                temp = list(filter(lambda x: x[3] > 0, words_info))
                if len(temp) == 0:
                    temp.append(words_info[0])
                for x in temp[0:k]:
                    opinion = {'actual_index': [x[4][0], x[4][-1] + 1],
                               'is_potential_aspect': self._is_potential_aspect(
                                                        priori_value=self._get_priori_value(word=x[0]),
                                                        pos=x[1])}

                    one_text_opinion_list.append( opinion )
            opinion_list.append( one_text_opinion_list )
        return opinion_list

    def _is_true(self, range1: list, range2: list, which: str):
        range1 = [range1[0], range1[1] - 1] if range1[0] != -1 else range1
        range2 = [range2[0], range2[1] - 1] if range2[0] != -1 else range2
        if which == 'strict':
            return range1[0] == range2[0] and range1[1] == range2[1]
        elif which == 'loose':
            return range1[0] <= range2[0] <= range1[1] or range2[0] <= range1[0] <= range1[1] <= range2[1] or range1[0] <= range2[1] <= range1[1]
        raise RuntimeError('illegal argument')

    def _process_sentence(self, sentence: str) -> str:
        index, words, result = 0, sentence.split(' '), []
        while index < len(words):
            if index < len(words) - 3 and words[index+1] == "'" and words[index+2] in ('t', 'd', 'm', 's', 're', 've', 'll'):
                result.append(''.join(words[index:index+3]))
                index += 3
                continue
            result.append(words[index])
            index += 1
        return ' '.join(result)

    def _index_token(self, sentence: str, token_list: Iterable):
        index, result = 0, []
        for token in token_list:
            token_index = sentence.find(token[0], index)
            if token_index < 0:
                return []
            index = token_index + len(token[0])
            result.append((token[0], token[1], (token_index, index)))
        return result

    def _merge(self, iter_pos: list, iter_lemma: list, iter_shap: list, iter_index: list):
        if 0 in (len(iter_pos), len(iter_lemma), len(iter_shap), len(iter_index)):
            return []
        return list(map(lambda x: (x[0][0], x[0][1], x[1], sum(map(lambda y: y[1], filter(lambda y: self._is_true(x[0][2], y[2], 'loose'), iter_shap))), list(map(lambda y: y[1], filter(lambda y: self._is_true(x[0][2], y[2], 'loose'), iter_index)))), zip(iter_pos, iter_lemma)))

    def _load_data(self):
        data_list = []
        text_sentiment_data = pd.read_csv(self.sentiment_file)
        text_list = list(text_sentiment_data['text'])
        sentiment_list = list(text_sentiment_data['sentiment'])
        with open(self.shap_file, 'rb') as f:
            shap_list = pickle.load(f)
        for text, shaps, sentiment in zip(text_list, shap_list, sentiment_list):
            sentence = self._process_sentence(text)
            iter_lemma = list(map(lambda x: x['lemma'],
                                  chain(*map(lambda x: x['tokens'], json.loads(nlp.annotate(sentence))['sentences']))))
            iter_pos, iter_shap, iter_index = nlp.pos_tag(sentence), map(lambda x: (x[0].strip(), x[1]), shaps), map(
                lambda x: (x[1], x[0]), enumerate(text.split(' ')))
            iter_pos, iter_shap, iter_index = self._index_token(sentence, iter_pos), self._index_token(sentence,
                                                                                                       iter_shap), self._index_token(
                sentence, iter_index)
            data_list.append({'sentence': text,
                              'processed_sentence': sentence,
                              'sentiment': sentiment,
                              'dependency': nlp.dependency_parse(sentence),
                              'constituency': nlp.parse(sentence),
                              'words_info': self._merge(iter_pos, iter_lemma, iter_shap, iter_index)})
        return data_list

# Write after analyzing all the texts:
def analysis_shap_1(model: Model, text_list: list, sentiment_list: list):
    """
    calculate shapely value of texts using model
    """
    def func(x):
        data = DataGenerator(x, [0 for _ in range(len(x))])
        outputs = model.predict(data)
        return outputs

    piece_list, shap_list = [], []
    explainer = Explainer(func, Text(DataGenerator.tokenizer))
    for text, sentiment in zip(text_list, sentiment_list):
        shap_value = explainer([text])
        piece, shap = shap_value.data[0][1:-1], shap_value.values[0, :, sentiment+1][1:-1]
        print(piece, shap)
        shap_list.append(shap)
        piece_list.append(piece)
    return list(map(lambda x: list(zip(x[0], x[1])), zip(piece_list, shap_list)))

# While analyzing SHAP, intermediate results are recorded to prevent the program from crashing due to resource issues, ensuring that early results are not lost:
def analysis_shap_2(target_name: str,model: Model, text_list: list, sentiment_list: list,shap_tmp_file: str,shap_file: str):
    # Load parsed files：
    tmp_fname = shap_tmp_file
    output_fname = shap_file
    parsed_text_list = []
    if os.path.exists(tmp_fname):
        with open(tmp_fname, 'r') as file:
            for line in file:
                line = line.strip()
                parsed_text_shaps = json.loads(line)
                parsed_text_list.append( parsed_text_shaps['text'] )

    # Start parsing：
    def func(x):
        data = DataGenerator(x, [0 for _ in range(len(x))])
        outputs = model.predict(data)
        return outputs

    explainer = Explainer(func, Text(DataGenerator.tokenizer))
    size = len(text_list)
    for i in range(size):
        text = text_list[i]
        sentiment = sentiment_list[i]
        parsed_text = parsed_text_list[i] if i<len(parsed_text_list) else None
        # Already parsed, skip：
        if text==parsed_text:
            continue
        #Conduct parsing：
        else:
            shap_value = explainer([text])
            piece, shap = shap_value.data[0][1:-1], shap_value.values[0, :, sentiment+1][1:-1]
            piece_shap_list = list( zip(piece,shap) )
            print(piece_shap_list)
            parsed_text = {'text': text, 'shaps': piece_shap_list}
            parsed_text_json = json.dumps(parsed_text)
            with open(tmp_fname, 'a') as file:
                file.write(parsed_text_json+"\n")

    # In the end, collectively write to the final file
    shaps_list = []
    with open(tmp_fname, 'r') as file:
        for line in file:
            line = line.strip()
            parsed_text_shaps = json.loads(line)
            shaps = parsed_text_shaps['shaps']
            shaps_list.append( shaps )
    # Save SHAP values as a pickle file.
    if len(shaps_list)==len(text_list):
        print("Save SHAP values as pickle files!")
        with open(output_fname, 'wb') as f:
            pickle.dump(shaps_list, f)


def extract_opinion(target_name,best_model,sentiment_fname,shap_tmp_file,shap_file,opinion_file):
    print(f"Start extract_opinion()!")
    print(f"Model: {best_model}")
    print(f"Sentiment File: {sentiment_fname}")
    print(f"Shap Tmp File: {shap_tmp_file}")
    print(f"Shap Final File: {shap_file}")

    # Step 2.1 ：SHAP Analysis
    text_sentiment_data = pd.read_csv(sentiment_fname)
    text_list = list(text_sentiment_data['text'])
    sentiment_list = list(text_sentiment_data['sentiment'])
    analysis_shap_2(target_name,best_model,text_list,sentiment_list,shap_tmp_file,shap_file)

    # Step 2.2 ：Extract representative word based on SHAP values.
    extractor = OpinionExtractor(sentiment_fname,shap_file)
    text_list = extractor.get_texts()
    opinion_list = extractor.extract_clause_opinion(std_p=std_p_value)
    with open(opinion_file, 'w') as file:
        for i in range(len(opinion_list)):
            text_unit = {"text": text_list[i], "opinions": opinion_list[i]}
            # print(opinion_list[i])
            file.write(f"{json.dumps(text_unit)}\n")

    print(f"Opinion File: {opinion_file}")


if __name__ == "__main__":
    """
        second step work, calculate SHAP value, extract representative words
    """

    target_name = 'laptop'

    model_name = f'{target_name}_model'
    include_text_cnn = True
    best_model = load_model(model_name, *EASTER_en.model_params[target_name], include_text_cnn)

    best_model = None
    sentiment_fname = f'{BASE_DIR}/data/pred_senti/{target_name}_test_sentiment.csv'
    shap_tmp_file = f'{BASE_DIR}/data/pred_opinion/{target_name}_test_shaps_tmp.txt'
    shap_file = f'{BASE_DIR}/data/pred_opinion/{target_name}_test_shaps.pkl'
    opinion_file = f'{BASE_DIR}/data/pred_opinion/{target_name}_opinion.txt'

    extract_opinion(target_name,best_model,sentiment_fname,shap_tmp_file,shap_file,opinion_file)
