#coding:utf-8
import sys
import json
import sys
import json
import os
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from utils import cut_sentences
import jieba
data_folder = '/kaggle/input/cail2023/cail2023_/datasets/'
def get_corpus():
    corpus = {}
    for filename in os.listdir(data_folder+'corpus'):
        doc_id = filename.split('.')[0]
        doc_text = ''
        with open(data_folder+'corpus/'+filename,'r',encoding='utf-8') as lines:
            for line in lines:
                doc_text+=line.strip()
        corpus[doc_id] = doc_text
    return corpus
def get_search_content(filename):
    text = ''
    with open(filename,'r',encoding='utf-8') as lines:
        for line in lines:
            text+=line.strip()
    return text
def get_train_data(filename):
    return json.load(open(filename,'r',encoding='utf-8'))

corpus = get_corpus()
def rerank_by_bm25(query,candidates,with_ids=False):
    query = ' '.join(jieba.lcut(query))
    docs = [corpus[id] for id in candidates] if with_ids else candidates
    cut_corpus =  [' '.join(jieba.lcut(doc)) for doc in docs ]
    bm25 = BM25Okapi(cut_corpus)
    scores = bm25.get_scores(query)
    candidate_scores = {id:score  for id,score  in zip(candidates,scores) }
    sorted_candidates = sorted(candidate_scores.items(),key=lambda x:x[-1],reverse=True)
    return ([w[0] for w in sorted_candidates],docs)

writer = open('test_cail_data_stage2.jsonl','a+',encoding='utf-8')
test_data_file = data_folder+'/test_candidates_stage2.json'
test_data  = get_train_data(test_data_file)
for filename in tqdm(os.listdir(data_folder+'test_stage2')):
    if '.txt' not in filename:continue
    search_text=get_search_content(data_folder+'test_stage2/'+filename)
    q_id = filename.split('.txt')[0]
    candidate_docis = test_data[filename.split('.txt')[0]]
    sorted_candidates,doc_contents =  rerank_by_bm25(search_text,candidate_docis,with_ids=True)
    new_docs = []
    for doc, doc_id in zip(doc_contents, candidate_docis):
        sentences = cut_sentences(doc)
        rerank_sentences, _ = rerank_by_bm25(search_text, sentences)
        new_doc = ''
        for sentence in rerank_sentences:
            if len(new_doc) >= 512:
                break
            new_doc += sentence
        new_docs.append(new_doc)
    json_data = {'idx':q_id,'docs':new_docs,'query':search_text,'candidates':candidate_docis}
    writer.write(json.dumps(json_data,ensure_ascii=False)+'\n')
writer.close()
