# -*- coding: utf-8 -*-
# ltp模块
import sys, os
from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
import pandas as pd
import numpy as np
from tqdm import tqdm
#segmentor.release()  # 释放模型

class ltp_api(object):
    def __init__(self,MODELDIR,exword_path = None):
        self.MODELDIR = MODELDIR
        self.output = {}
        self.words = None
        self.postags = None
        self.netags = None
        self.arcs = None
        self.exword_path = exword_path  #  e.x: '/data1/research/matt/ltp/exwords.txt'
        # 分词
        self.segmentor = Segmentor()
        if not self.exword_path:
            # 是否加载额外词典
            self.segmentor.load(os.path.join(self.MODELDIR, "cws.model"))
        else:
            self.segmentor.load_with_lexicon(os.path.join(self.MODELDIR, "cws.model"), self.exword_path)
        
        # 词性标注
        self.postagger = Postagger()
        self.postagger.load(os.path.join(self.MODELDIR, "pos.model"))
        # 依存句法
        self.parser = Parser()
        self.parser.load(os.path.join(self.MODELDIR, "parser.model"))
        # 命名实体识别
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(self.MODELDIR, "ner.model"))
        # 语义角色
        self.labeller = SementicRoleLabeller()
        self.labeller.load(os.path.join(MODELDIR, "pisrl.model"))
        
    # 分词
    def ltp_segmentor(self,sentence):
        words = self.segmentor.segment(sentence)
        return words

    # 词性标注
    def ltp_postagger(self,words):
        postags = self.postagger.postag(words)
        return postags
    
    # 依存语法
    def ltp_parser(self,words, postags):
        arcs = self.parser.parse(words, postags)
        return arcs
    
    # 命名实体识别
    def ltp_recognizer(self,words, postags):
        netags = self.recognizer.recognize(words, postags)
        return netags
    
    # 语义角色识别
    def ltp_labeller(self,words,postags, arcs):
        output = []
        roles = self.labeller.label(words, postags, arcs)
        for role in roles:
            output.append([(role.index,arg.name, arg.range.start, arg.range.end) for arg in role.arguments])
        return output
    
    def release(self):
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()
        self.recognizer.release()
        self.labeller.release()
        
    def get_result(self,sentence):
        self.words = self.ltp_segmentor(sentence)
        self.postags = self.ltp_postagger(self.words)
        self.arcs = self.ltp_parser(self.words, self.postags)
        self.netags = self.ltp_recognizer(self.words, self.postags)
        self.output['role'] = self.ltp_labeller(self.words,self.postags, self.arcs)
    
        # 载入output
        self.output['words'] = list(self.words)
        self.output['postags'] = list(self.postags)
        self.output['arcs'] = [(arc.head, arc.relation) for arc in self.arcs]
        self.output['netags'] = list(self.netags)



'''
语义角色的解读
主要定位到动词，然后动词实施者与动作的影响人

A0 - A1 ，A0代表主语，A1代表动作的影响

'''
def FindA0(labelle,word,postags,neg_word = ['就是','是'],n_pos = ['n','ns','nt']):
    ''' 
    找到是否有A0  
    
    输入:labelle,word,postags相关词类型
    
    输出：
        A0 是否有A0 bool,True/False
        result:[名词,动词,修饰词（相当于定语）]
    '''
    result = []
    A0 = False
    # 是否有A0，动作实施者，相当于主语
    sign_n = [n for n,la in enumerate(labelle) if la[1] == 'A0']
    if len(sign_n) > 0:
        A0 = True
        la = labelle[sign_n[0]]
        verb_word = word[la[0]]
        if verb_word in neg_word:
            return A0,result
        low = la[2] 
        high = la[3] if (la[3] + 1) > len(words) else la[3] + 1
        long_words = [words[n]  for n in range(low,high) if postags[n] in n_pos]
        n_word =  word[la[2]] if la[2] == la[3] else long_words
        
        
        # A1 动作影响，想当于宾语
        sign_n_A1 = [n for n,la in enumerate(labelle) if la[1] == 'A1']
        adore_word = []
        if len(sign_n_A1) > 0:
            la2 = labelle[sign_n_A1[0]]
            low = la2[2]
            high = la2[3] if (la2[3] + 1) > len(words) else la2[3] + 1
            adore_word = word[la2[2]] if la2[2] == la2[3] else words[ low : high ]
        result = [n_word,verb_word,adore_word]
    return A0,result
            
def SRLparsing(labeller,words,postags,ToAfter = ['TMP','A1','DIS'],neg_word = ['就是','是'],n_pos = ['n','ns','nt']):
    '''
    输入:
    ToAfter，指的是这些语义角色的类型，TMP(时间),A1(动作的影响),DIS(标记语),这三个影响的对象在后面
    
    输出：
    ([['ADV', ('最后', '打')], ['ADV', (['平均', '下来'], '便宜')], ['A0', ('40', '便宜')]], (True, ['40', '便宜', []]))
    
    '''
    labeller_refine = []
    labeller_A0 = []
    for labelle in labeller:
        #print(labelle)
        for la in labelle:
            if la[2] == la[3]:
                tmp = [la[1],(words[la[0]],words[la[3]])] if la[1] in ToAfter else [la[1],(words[la[3]],words[la[0]])]
                labeller_refine.append(tmp)
                #print('keypoint word :',words[la[0]])
                #print(tmp)
            else:
                low = la[2]
                high = la[3] if (la[3] + 1) > len(words) else la[3] + 1
                tmp = [la[1],(words[la[0]],words[low:high])] if la[1] in ToAfter else [la[1],(words[low:high],words[la[0]])]
                labeller_refine.append(tmp)
                #print('keypoint word :',words[la[0]])
                #print(tmp)
        #print('\n A0A1 ==== > ',FindA0(labelle,words,postags))
        labeller_A0 = FindA0(labelle,words,postags,neg_word = neg_word,n_pos = n_pos)
        #print('-----------\n')
    return labeller_refine,labeller_A0

if __name__=="__main__":
    MODELDIR='ltp-models/ltp_data_v3.4.0'   #  模型文件
    ltp = ltp_api(MODELDIR)
    # ltp.release()  
    sentence = '环境很好，位置独立性很强，比较安静很切合店名，半闲居，偷得半日闲。点了比较经典的菜品，味道果然不错！'
    words = ltp.ltp_segmentor(sentence)  # 分词
    postags = ltp.ltp_postagger(words)  # 词性
    arcs = ltp.ltp_parser(words,postags)  #依存
    netags = ltp.ltp_recognizer(words,postags)# 命名实体识别
    labeller = ltp.ltp_labeller(words,postags, arcs) #语义角色
    print(SRLparsing(labeller,words,postags,ToAfter = ['TMP','A1','DIS']))
