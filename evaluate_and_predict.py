#coding:utf-8
import sys
import json
import sys

import argparse

parser = argparse.ArgumentParser(description='命令行中传入一个数字')
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('model_path', type=str, help='训练后的模型文件')
parser.add_argument('result_file',type=str,help='结果文件')
args = parser.parse_args()
print(args)
from sentence_transformers import SentenceTransformer
model_name = './finetuned_bge'
model = SentenceTransformer(model_name,device='cuda')
score = 0.0
cnt = 0
with open('dev_bge_data.jsonl','r',encoding='utf-8') as lines:
  for line in lines:
    cnt+=1
    data = json.loads(line.strip())
    docs = data['pos']+data['neg']
    query_emb = model.encode(data['query'], normalize_embeddings=True)
    doc_emb = model.encode(docs,normalize_embeddings=True)
    similarity = query_emb@doc_emb.T
    doc_score = {i:score for i,score in enumerate(similarity)}
    sorted_ids = [w[0] for w in sorted(doc_score.items(),key=lambda x:x[1],reverse=True)]
    score+=1.0/(sorted_ids.index(0)+1)
print('evaluate ...'+str(float(score)/cnt))
