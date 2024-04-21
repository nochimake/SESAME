import json
import os

import pandas as pd

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
    os.system (command)
    with open(output_fname, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_aos(target_name,lang,res_fpath,sentiment_fpath,opinion_fpath):
    print(f"Start extract_aos()!")
    print(f"Target Name: {target_name}")
    print(f"Target Language: {lang}")
    print(f"Sentiment File: {sentiment_fpath}")
    print(f"Opinion File: {opinion_fpath}")

    # Read sentiment polarity.
    text_sentiment_data = pd.read_csv(sentiment_fpath)
    text_list = list(text_sentiment_data['text'])
    sentiment_list = list(text_sentiment_data['sentiment'])
    # Read representative word.
    opinions_list = []
    potential_aspect_boolean_list = []
    with open(opinion_fpath, 'r') as file:
        for line in file:
            text_unit = json.loads( line.strip() )
            text_opinions = text_unit["opinions"]
            opinions_list.append( [opinion["actual_index"] for opinion in text_opinions] )
            potential_aspect_boolean_list.append( [opinion["is_potential_aspect"] for opinion in text_opinions] )

    # Extract aspect.
    aspects_list = extract_aspect(lang,text_list, opinions_list, potential_aspect_boolean_list)
    # Organize (a,s) pairs s:sentiment; a:aspect
    res_list = []
    for text,senti,aspect_units in zip(text_list,sentiment_list,aspects_list):
        aos_list = []
        text_aspect_list = []
        text_aspect_string_list = []
        for aspect_unit in aspect_units:
            aspect = aspect_unit["aspect"]
            aspect_string = str(aspect)
            if not text_aspect_string_list.__contains__(aspect_string):
                text_aspect_string_list.append(aspect_string)
                text_aspect_list.append(aspect)
            continue
        for aspect in text_aspect_list:
            as_pair = {'sentiment': senti,'aspect': aspect}
            aos_list.append( as_pair )
        res = {'sentence': text,'as_list': aos_list}
        res_list.append(res)
    # Write the results to a file.
    with open(res_fpath, 'w') as f:
        for res in res_list:
            f.write(json.dumps(res,ensure_ascii=False)+"\n")
    print(f"The aspect-sentiment pairs have been output in: {res_fpath}!")


lang_map = {'laptop': 'en',
            'rest16': 'en',
            'phone': 'ch',
            'camera': 'ch' }

if __name__ == "__main__":
    """
        third step work，extract aspect
    """

    target_name = 'laptop'
    lang = lang_map[target_name]
    sentiment_fpath = f'{BASE_DIR}/data/pred_senti/{target_name}_test_sentiment.csv'
    opinion_fpath = f'{BASE_DIR}/data/pred_opinion/{target_name}_opinion.txt'
    res_fpath = f'{BASE_DIR}/data/pred_aspect/{target_name}_aos.txt'

    extract_aos(target_name,lang,res_fpath,sentiment_fpath,opinion_fpath)



