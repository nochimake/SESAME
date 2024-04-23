import os
import pandas as pd
import json
import pickle
from itertools import chain
from typing import Iterable
from keras.api._v2.keras import Model
from pycorenlp import StanfordCoreNLP as pyCoreNLP
from config import *
import EASTER_ch
from shap.maskers import Text
from shap import Explainer
import nltk

std_p_value = 0.8

nlp = pyCoreNLP('http://localhost:9876')
# 1. Copy StanfordCoreNLP-chinese.properties into STANFORD_CORE_NLP_PATH.
# 2. cd STANFORD_CORE_NLP_PATH
# 3. execute "java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port 9876 -timeout 15000"
class OpinionExtractor:
    def __init__(self, sentiment_file: str, shap_file: str, ):
        self.shap_file = shap_file
        self.sentiment_file = sentiment_file
        self.words_lead_to_clause = ["，"]
        self.pos_tags_lead_to_clause = ["CC"]
        self.clause_label_list = ['IP']
        self.pos_map = {
            "NR": "n", "NT": "n", "NN": "n",
            "VC": "v", "VE": "v", "VV": "v",
            "JJ": "a", "VA": "a",
            "AD": "r",
        }
        # load sentiwordnet
        df = pd.read_excel(f'{BASE_DIR}/data/dict/情感词汇本体.xlsx')
        self.sentiwordnet = df["词语"].tolist()
        # laod dataset
        self.data_list = self._load_data()

    def get_texts(self):
        return [data['sentence'] for data in self.data_list]

    def _is_potential_aspect(self, word: str, pos: str):
        """
        To determine if it is a potential aspect,
        the conditions for being a potential aspect are:
            1) The word is not in the affective lexicon ontology (情感词汇本体) ,
            2) It is a noun.
        """
        return (not self.sentiwordnet.__contains__(word)) and self.pos_map[pos]=="n"

    def _get_clause_with_constituency(self, data):
        '''
        Split text based on constituency analysis
        '''
        if len(data['words_info']) == 0:
            return data['words_info']
        constituency_tree_str = data['constituency']
        word_list = [ x[2] for x in data['words_info'] ]
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
        for subtree in tree.subtrees():
            try:
                start_index = int(subtree.leaves()[0])
            except ValueError:
                continue
            if subtree.label() in self.clause_label_list:
                clause_start_index_set.add(start_index)
        clauses = []
        current_clause = []
        start_index_set = clause_start_index_set
        for i in range(len(data['words_info'])):
            item = data['words_info'][i]
            # if (i in start_index_set) :
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

    def extract_clause_opinion(self,std_p: float):
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
            shap_values = [x[4] for x in data['words_info']]
            standard = np.mean(shap_values) + std_p * np.std(shap_values)
            clause_words_info_list = self._get_clause_with_constituency(data)
            one_text_opinion_list = []
            max_clause_opinion = None
            for clause_words_info in clause_words_info_list:
                clause_words_info = sorted(filter(lambda x: self.pos_map.get(x[3], None) is not None, clause_words_info),
                                    key=lambda x: x[4], reverse=True)
                if len(clause_words_info)==0:
                    continue
                clause_opinion = clause_words_info[0]
                if max_clause_opinion is None or clause_opinion[4]>max_clause_opinion[4]:
                    max_clause_opinion = clause_opinion
                if clause_opinion[4]<standard:
                    continue
                opinion = {'actual_index': clause_opinion[1],
                           'is_potential_aspect': self._is_potential_aspect(word=clause_opinion[2],pos=clause_opinion[3])}
                one_text_opinion_list.append(opinion)
            if len(one_text_opinion_list)==0:
                if max_clause_opinion is not None:
                    opinion = {'actual_index': max_clause_opinion[1],
                               'is_potential_aspect': self._is_potential_aspect(word=max_clause_opinion[2],pos=max_clause_opinion[3])}

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
            one_text_opinion_list = []
            words_info = sorted(filter(lambda x: self.pos_map.get(x[3], None) is not None, data['words_info']), key=lambda x: x[4], reverse=True)
            if len(words_info) == 0:
                opinion = {'actual_index':[-1, -1],
                           'is_potential_aspect':False}
                one_text_opinion_list.append( opinion )
            else:
                for x in words_info[0:k]:
                    opinion = {'actual_index':x[1],
                               'is_potential_aspect':self._is_potential_aspect(word=x[2],pos=x[3])}
                    one_text_opinion_list.append( opinion )
            opinion_list.append( one_text_opinion_list )
        return opinion_list

    def _load_data(self):
        data_list = []
        text_sentiment_data = pd.read_csv(self.sentiment_file)
        text_list = list(text_sentiment_data['text'])
        sentiment_list = list(text_sentiment_data['sentiment'])
        with open(self.shap_file, 'rb') as f:
            shap_list = pickle.load(f)
        for text, shaps, sentiment in zip(text_list, shap_list, sentiment_list):
            parse = nlp.annotate(text, properties={
                'annotators': 'tokenize, ssplit, pos, lemma, parse, coref',
                'outputFormat': 'json',
                'timeout': 30000,
            })
            sentences = json.loads(parse)['sentences']
            iter_word = list(map(lambda x: x['word'], chain(*map(lambda x: x['tokens'], sentences))))
            iter_pos = list(map(lambda x: x['pos'], chain(*map(lambda x: x['tokens'], sentences))))
            iter_index = list(map(lambda x: x['index'], chain(*map(lambda x: x['tokens'], sentences))))
            iter_character_offset_begin = list(map(lambda x: x['characterOffsetBegin'], chain(*map(lambda x: x['tokens'], sentences))))
            iter_character_offset_end = list(map(lambda x: x['characterOffsetEnd'], chain(*map(lambda x: x['tokens'], sentences))))
            parse_list = [sentence["parse"] for sentence in sentences]
            parse_string = " ".join(parse_list)
            words_info = []
            start_index = 0
            for word,pos,index,characterOffsetBegin,characterOffsetEnd in zip(iter_word,iter_pos,iter_index,iter_character_offset_begin,iter_character_offset_end):
                word_shap = sum(shap[1] for shap in shaps[start_index:start_index+len(word)])
                start_index = start_index+len(word)
                word_info = [index-1,[characterOffsetBegin,characterOffsetEnd],word,pos,word_shap]
                words_info.append(word_info)
            data_list.append({'sentence': text,
                              'sentiment': sentiment,
                              'words_info': words_info,
                              'constituency': parse_string})
        return data_list


# Write after analyzing all the texts:
def analysis_shap_1(model: Model, text_list: list, sentiment_list: list):
    def func(x):
        data = EASTER_ch.DataGenerator(x, [0 for _ in range(len(x))])
        outputs = model.predict(data)
        return outputs

    piece_list, shap_list = [], []
    explainer = Explainer(func, Text(EASTER_ch.DataGenerator.tokenizer))
    for text, sentiment in zip(text_list, sentiment_list):
        shap_value = explainer([text])
        piece, shap = shap_value.data[0][1:-1], shap_value.values[0, :, sentiment+1][1:-1]
        print(piece, shap)
        shap_list.append(shap)
        piece_list.append(piece)
    return list(map(lambda x: list(zip(x[0], x[1])), zip(piece_list, shap_list)))

# While analyzing SHAP, intermediate results are recorded to prevent the program from crashing due to resource issues, ensuring that early results are not lost:
def analysis_shap_2(target_name: str,model: Model, text_list: list, sentiment_list: list, shap_tmp_file: str, shap_file: str):
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
        data = EASTER_ch.DataGenerator(x, [0 for _ in range(len(x))])
        outputs = model.predict(data)
        return outputs

    explainer = Explainer(func, Text(EASTER_ch.DataGenerator.tokenizer))
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
            parsed_text =  {}
            parsed_text['text'] = text
            parsed_text['shaps'] = piece_shap_list
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

def extract_opinion(target_name, best_model, sentiment_fname, shap_tmp_file, shap_file, opinion_file):
    print(f"Start extract_opinion()!")
    print(f"Model: {best_model}")
    print(f"Sentiment File: {sentiment_fname}")
    print(f"Shap Tmp File: {shap_tmp_file}")
    print(f"Shap Final File: {shap_file}")

    # Step 2.1 ：SHAP Analysis
    text_sentiment_data = pd.read_csv(sentiment_fname)
    text_list = list(text_sentiment_data['text'])
    sentiment_list = list(text_sentiment_data['sentiment'])
    analysis_shap_2(target_name, best_model, text_list, sentiment_list, shap_tmp_file, shap_file)

    # Step 2.2 ：Extract representative word based on SHAP values.
    extractor = OpinionExtractor(sentiment_fname, shap_file)
    text_list = extractor.get_texts()
    opinion_list = extractor.extract_clause_opinion(std_p=std_p_value)
    with open(opinion_file, 'w') as file:
        for i in range(len(opinion_list)):
            text_unit = {"text": text_list[i], "opinions": opinion_list[i]}
            file.write(f"{json.dumps(text_unit,ensure_ascii=False)}\n")
    print(f"Opinion File: {opinion_file}")


if __name__ == "__main__":
    """
        second step work, calculate SHAP value, extract representative words
    """

    target_name = 'phone'
    model_name = f'{target_name}_model'
    include_text_cnn = True
    best_model = EASTER_ch.load_model(model_name, *EASTER_ch.model_params[target_name], include_text_cnn)

    best_model = None
    sentiment_fname = f'{BASE_DIR}/data/pred_senti/{target_name}_test_sentiment.csv'
    shap_tmp_file = f'{BASE_DIR}/data/pred_opinion/{target_name}_test_shaps_tmp.txt'
    shap_file = f'{BASE_DIR}/data/pred_opinion/{target_name}_test_shaps.pkl'
    opinion_file = f'{BASE_DIR}/data/pred_opinion/{target_name}_opinion.txt'

    extract_opinion(target_name, best_model, sentiment_fname, shap_tmp_file, shap_file, opinion_file)
