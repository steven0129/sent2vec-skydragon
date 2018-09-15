from tqdm import tqdm
from gensim.models import FastText
import fire
import ltp
import os

DICT_DIR = 'dict'
DATA_DIR = 'data'

def text(**kwrags):
    S = ltp.Sentence()
    W = ltp.Word(dictDir=DICT_DIR)

    print('處理天龍八部文本中...')
    with open(f'{DATA_DIR}/sky_dragon.txt') as f:
        contents = filter(lambda x: not x.startswith('['), f.readlines())
        contents = map(lambda x: x.rstrip('\n'), list(contents))
        content = ''.join(list(contents))

    print('斷句中...')
    sents = S.split(content)

    print('斷詞中...')
    wordLists = list(map(W.split, tqdm(sents)))

    print('產生句向量中...')
    with open('tmp.txt', 'w') as f:
        sents = list(map(lambda x: ' '.join(x), wordLists))
        f.write('\n'.join(sents))
    
    os.system('./fasttext sent2vec -input tmp.txt -output model/textModelS -dim 700 -epoch 1000 -minCount 0')
    os.system('./fasttext print-sentence-vectors model/textModelS.bin < tmp.txt > model/textModelS.vec')

    print('產生詞向量中...')
    with open('tmp.txt', 'w') as f:
        sents = list(map(lambda x: ' '.join(x), wordLists))
        f.write('\n'.join(sents))
    
    os.system('./fasttext skipgram -input tmp.txt -output model/textModelW -dim 300 -epoch 5000 -minCount 0')
    os.system('rm tmp.txt')

def abstract(**kwrags):
    S = ltp.Sentence()
    W = ltp.Word(dictDir=DICT_DIR)

    print('處理天龍八部摘要中...')
    with open(f'{DATA_DIR}/sky_dragon_abstract.txt') as f:
        contents = map(lambda x: x.rstrip('\n'), list(f.readlines()))
        content = ''.join(list(contents))

    print('斷句中...')
    sents = S.split(content)

    print('斷詞中...')
    wordLists = list(map(W.split, tqdm(sents)))

    print('產生句向量中...')
    with open('tmp.txt', 'w') as f:
        sents = list(map(lambda x: ' '.join(x), wordLists))
        f.write('\n'.join(sents))
    os.system('./fasttext print-sentence-vectors model/textModelS.bin < tmp.txt > model/abstractModelS.vec')
    os.system('rm tmp.txt')

if __name__ == '__main__':
    fire.Fire()