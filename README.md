# SESAME: Leveraging Labeled Overall Polarities for Aspect-based Sentiment Analysis based on SHAP-Enhanced Syntactic Analysis

![image](https://github.com/nochimake/SESAME/blob/main/schematicDiagram.png)

  In this paper, we propose an Aspect-Sentiment-Pair-Extraction (ASPE) approach that extracts aspect-sentiment pairs (a,s) through syntactic analysis enhanced by AI explainable framework on labeled overall polarities, rather than introduce additional manual annotations. Specifically, our approach first solely leverages the overall-polarity-labeled data to train a classifier, and then deduces aspects based on our syntactic analysis guided by the calculated contribution values for each word in the texts from our trained classifier. We named it **SESAME** (SHAP-Enhanced Sytactic Analysis for Aspect-based sentiMEnt analysis). In particular, our approach consists of three stages. **First**, we use a RoBERTa-incorporated TextCNN framework to train a sentiment-polarity (i.e., positive, neutral, and negative) classifier. **Second**, we use the AI-Explanation framework SHAP (SHapley-Additive-exPlanation) to analyze our classification results, and select the representative words based on the SHAP values. **Third**, we extract aspects from the representative words based on our proposed syntactic rules considering the word dependencies in each human-written sentence from input texts. The major advantage of the proposed SHAP-Enhanced syntactic analysis lies in that the exploited word dependencies can greatly reveal the implicit relations between the sentiment polarities and aspects, thus largely alleviating the manual efforts of additional human-annotated, intermediate sentiment elements, such as opinion and category. As a result, the proposed approach is easy-to-use in practice for different domains compared to traditional approaches.

   Our evaluation is based on two English datasets  [[Cai et al.@ACL2021]](https://github.com/NUSTM/ACOS)  with the complete (𝑎,𝑐,𝑜,𝑠) quadruples annotated (for SESAME the quadruples are only used for evaluation), and two Chinese datasets [[Peng et al. @KBS2018]](http://sentic.net/chinese-review-datasets.zip) with the (𝑎,𝑠) pair annotated to show that SESAME can also support different languages when slightly adapted. Our four baseline approaches are the recently popular ChatGPT and three state-of-the-art (SOTA) learning-based ABSA approaches that consider (𝑎,𝑐,𝑜,𝑠) quadruples [Extract-Classify-ACOS](https://github.com/NUSTM/ACOS), (𝑎,𝑜,𝑠) triplets [Span-ASTE](https://github.com/chiayewken/Span-ASTE), and (𝑎,𝑠) pairs [LCF-ATEPC](https://github.com/yangheng95/LCF-ATEPC) (the newest implementation is in [PyABSA](https://github.com/yangheng95/PyABSA/tree/release/demos/aspect_term_extraction)) respectively. For the ASPE task, our approach, which solely learns from overall-polarity- labeled data, can achieve an average 80.1\% of the quadruple-learning approach in F1-score of precision and recall, an average 91.7\% of the triple-learning approach, and an average 102.8\% of the pair-learning approach. The evaluation shows that our approach is an easy-to-use and also explainable ABSA approaches that achieve a comparable level of performance (especially for solely extracting aspects) to SOTA learning-based approaches, while it only requires the la- beled sentiments (i.e., one out of four manual labels required by the best-performed baseline), indicating that our approach is more applicable in specific domains (e.g., SE) where most datasets are only labeled by the overall sentiments.

  Since the benchmark dataset is not self-generated and published, we do not provide them here. If you want to use any of them, you should complete the license for their publication and consider referencing the original paper. You can download them from the link we provided above. (We publish the processed Chinese dataset in ```/Chinese AS Dataset``` folder of the project.)

## Overview

1. ```config.py``` The configurations of project.

2. ```EASTER_en.py```  Used for sentiment analysis of English text.

3. ```EASTER_ch.py``` Used for sentiment analysis of Chinese text.

   EASTER_en.py and EASTER_ch.py differ in :

   (1) They use different pretrained models and corresponding tokenizers.

   (2) In EASTER_ch.py, a CustomClassificationHead constructed to mimic the TFRobertaClassificationHead is used as the residual connections.

4. ```extract_opinion_en.py```  Calculate SHAP value, extract representative words for English text.

5. ```extract_opinion_ch.py```  Calculate SHAP value, extract representative words for Chinese text.

   extract_opinion_en.py and extract_opinion_ch.py differ in :

   (1) When computing SHAP values, they rely on the `load_model()` and `DataGenerator` specific to the corresponding language.

   (2) The parsing tags for different languages vary. For details, please refer to the [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) official website.

6. ```extrac_aspect.py```  Extracte aspects using syntactic rules.

7. ```SentiAspectExtractor Introduction```  Detailed introduction of aspect extraction rules

8. ```data/sentiment_acos```  Files required for training and testing the sentiment classifier.

9. ```data/pretrained``` Store trained models

10. ```data/pred_senti```  Store the results of predicted sentiment

11. ```data/pred_opinion```  Store the output results of representative words 

12. ```data/pred_aspect```  Store the final extracted (a, s)

13. ```data/dict```  Store dictionary resources

14. ```data/acos```  Store manually annotated test set results


## Dependencies

1. python=3.10.12

2. tensorflow=2.13.0

3. transformers=4.31.0


## Run

**Preparation**:

- *For the classifier in stage one*:

1. If you wish to reproduce our paper's data, you can download our pre-trained models from [nochimake/SESAME on Hugging Face](https://huggingface.co/nochimake/SESAME/tree/main) (the four models in the provided link serve as the foundational models for our ablation experiments).
2. If you intend to retrain the classifier, for English, please download the pre-trained RoBERTa model from  [HuggingFace](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment/tree/main), and for Chinese, please download it from [ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm).

- *For the extraction of representative words in the second stage*:

1. Download stanfordcorenlp tool from [CoreNLP](https://nlp.stanford.edu/software/stanford-corenlp-4.5.1.zip)  

2. For Chinese, you will need to download an additional Chinese model, copy both the Chinese model and StanfordCoreNLP-chinese.properties into STANFORD_CORE_NLP_PATH, and then run the following command under STANFORD_CORE_NLP_PATH to start the StanfordCoreNLP service:

   ```
   java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port 9000 -timeout 15000
   ```

  - *For aspect extraction in the third stage*:

    You can directly download SentiAspectExtractor.jar from [nochimake/SESAME on Hugging Face](https://huggingface.co/nochimake/SESAME/tree/main) . **Additionally**, we have open-sourced this code on GitHub at [nochimake/SentiAspectExtractor.](https://github.com/nochimake/SentiAspectExtractor)

    Finally, configure your config.py



**Running**:

1. run EASTER_en.py / EASTER_ch.py to to train or predict for your data.

2. run extract_opinion_en.py / extract_opinion_ch.py to select representative words

3. run extrac_aspect.py to extract aspects

4. run evaluate.py to assess result

