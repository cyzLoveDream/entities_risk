# -*- coding:utf-8 -*-
import pandas as pd

from tqdm import tqdm
import jieba
import jieba.posseg
import re
import time
import warnings
from untils.tools import textpreprocessing
from untils.tools import sentence_preprocessing
from untils.tools import get_word_pos_list
from untils.tools import get_entity_label_list
warnings.filterwarnings("ignore")
cutlist = list("。！？;； ")

def load_data(raw_file_path, raw_result_path):
	raw_data = []
	raw_result_data = []
	with open(raw_file_path, 'r', encoding="utf-8") as rf:
		lines = rf.readlines()
		for line in tqdm(lines):
			raw_data.append(eval(line))
	with open(raw_result_path, 'r', encoding='utf-8') as rf1:
		result_lines = rf1.readlines()
		for l in tqdm(result_lines):
			raw_result_data.append(eval(l))
	raw_data_pd = pd.DataFrame(raw_data)
	raw_result_pd = pd.DataFrame(raw_result_data)
	# print(raw_data_pd.head(), raw_result_pd.head())
	the_all_data = raw_data_pd.merge(raw_result_pd, how="left", on="newsId")
	return the_all_data

def deal_sents_list(textspd):
	"""
	处理句子
	:param textspd:
	:return:
	"""
	trainstring = []
	textslist = textspd['combine_title_body'].values
	entitieslist = textspd['entities'].values
	
	for text,entities in tqdm(zip(textslist,entitieslist)):
		sents = sentence_preprocessing(text)
		for sent in sents:
			str = ''
			wordlist = []
			poslist = []
			seg = jieba.posseg.lcut(sent)
			for wordtag in seg:
				# pattern = re.compile(u"[\u4e00-\u9fa5]+")
				# result = re.findall(pattern,wordtag.word)
				# if result != []:
				wordlist.append(wordtag.word)
				poslist.append(wordtag.flag)
			wordlist,poslist = get_word_pos_list(wordlist,poslist)
			entitylabellist = get_entity_label_list(wordlist,entities)
			for word,pos,entitylabel in zip(wordlist,poslist,entitylabellist):
				str += word + '\t' + pos + '\t' + entitylabel + '\n'
			trainstring.append(str)
	return trainstring
def main():
	start = time.time()
	the_all_data = load_data("../data/DATA/DATA_T.txt", "../data/DATA/DATA_T_RESULT.txt")
	the_all_data["combine_title_body"] = the_all_data["title"].apply(lambda x: x.strip()) +"。"+ the_all_data["body"].apply(lambda x:x.strip())
	the_all_data["combine_title_body"] = the_all_data["combine_title_body"].apply(textpreprocessing)
	the_all_data["entities"] = the_all_data["entities"].apply(lambda x: [d["name"] for d in x])
	train_str = deal_sents_list(the_all_data)
	with open('../feature_data/traintokens_pos_ch.txt', 'w', encoding='utf8') as file:
		for data in train_str:
			file.write(data + '\n')
	print('total cost time: %0.4f' % (time.time() - start),'s')
	pass

if __name__ == '__main__':
	main()