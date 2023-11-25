import os
import pandas as pd
import json
import pickle
from config import *


# configure java environment
os.environ['PATH'] = f'{os.environ["PATH"]}:{JAVA_PATH}/bin:{JAVA_PATH}/jre/bin'

def extract_aspect(lang:str, text_list:list,  opinion_list: list, potential_aspect_boolean_list: list = []):
    input_fname = f'{BASE_DIR}/data/pred_aspect/temp/opinion.txt'
    output_fname = f'{BASE_DIR}/data/pred_aspect/temp/aspect.txt'
    dictionary_fname = f'{BASE_DIR}/data/pred_aspect/dictionary/'
    jar_fname = f'{BASE_DIR}/data/pred_aspect/SentiAspectExtractor.jar'
    if len(potential_aspect_boolean_list)==len(opinion_list):
        with open(input_fname, 'w', encoding='utf-8') as f:
            for text, opinions, potential_aspect_booleans in zip(text_list, opinion_list,potential_aspect_boolean_list):
                f.write(text)
                f.write('\t')
                opinion_string_list = [
                    f'{",".join(map(str, opinion))}(SPA)' if boolean
                    else f'{",".join(map(str, opinion))}'
                    for opinion, boolean in zip(opinions, potential_aspect_booleans)
                ]
                opinion_string = '; '.join(opinion_string_list)
                f.write(opinion_string)
                f.write('\n')
    else:
        with open(input_fname, 'w', encoding='utf-8') as f:
            for text, opinions in zip(text_list, opinion_list):
                f.write(text)
                f.write('\t')
                f.write('; '.join(map(lambda x: ','.join(map(str, x)), opinions)))
                f.write('\n')
    command = f'java -jar {jar_fname} -inputfile {input_fname} -outputfile {output_fname} -dict {dictionary_fname}'
    if lang=="ch":
        command += ' -lang ch'
    command += ' -analysisparseresult'
    os.system (command)
    with open(output_fname, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_aos():
    target_name = 'rest16'
    lang = 'en'
    res_fpath = f'{BASE_DIR}/data/pred_aspect/{target_name}_aos.txt'
    sentiment_fpath = f'{BASE_DIR}/data/pred_senti/{target_name}_test_sentiment.csv'
    opinion_fpath = f'{BASE_DIR}/data/pred_opinion/{target_name}_opinion.txt'

    # target_name = 'camera'
    # lang = 'ch'
    # res_fpath = f'{BASE_DIR}/data/pred_aspect/{target_name}_aos.txt'
    # sentiment_fpath = f'{BASE_DIR}/data/pred_senti/{target_name}_test_sentiment.csv'
    # opinion_fpath = f'{BASE_DIR}/data/pred_opinion/{target_name}_opinion.txt'

    # Read sentiment polarity.
    text_sentiment_data = pd.read_csv(sentiment_fpath)
    text_list = list(text_sentiment_data['text'])
    sentiment_list = list(text_sentiment_data['sentiment'])
    # Read representative word.
    opinions_list = []
    potential_aspect_boolean_list = []
    implicit_map_list = []
    with open(opinion_fpath, 'r') as file:
        for line in file:
            text_unit = json.loads( line.strip() )
            text_opinions = text_unit["opinions"]
            opinions_list.append( [opinion["actual_index"] for opinion in text_opinions] )
            potential_aspect_boolean_list.append( [opinion["is_potential_aspect"] for opinion in text_opinions] )
            implicit_map_list.append( {tuple(opinion["actual_index"]): opinion["after_implicit"] for opinion in text_opinions} )

    # Extract aspect.
    aspects_list = extract_aspect(lang,text_list, opinions_list, potential_aspect_boolean_list)
    # Organize (s,o,a) triplets s:sentiment; o:opinion; a:aspect
    res_list = []
    for text,senti,implicit_map,aspect_units in zip(text_list,sentiment_list,implicit_map_list,aspects_list):
        aos_list = []
        for aspect_unit in aspect_units:
            aos = {'index': aspect_unit['opinion_index'],
                   'sentiment': senti,
                   'opinion': implicit_map[tuple(aspect_unit["opinion"])],
                   'aspect': aspect_unit["aspect"]}
            aos_list.append( aos )
        res = {'sentence': text,'aos_list': aos_list}
        res_list.append(res)
    # Write the results to a file.
    with open(res_fpath, 'w') as f:
        for res in res_list:
            f.write(json.dumps(res,ensure_ascii=False)+"\n")
    print(f"AOS 三元组结果已输出在：{res_fpath}!")


if __name__ == "__main__":
    """
        third step work，extract aspect
    """
    extract_aos()



