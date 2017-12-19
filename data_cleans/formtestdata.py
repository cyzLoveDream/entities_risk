# !/usr/bin/python
# -*- coding:utf-8 -*-

import time, re, os

os.chdir('E:\heting\ccf-competition\BDCI2017-fahainew')

from crffeature.formtraindata import *
import jieba
import jieba.posseg
from tqdm import tqdm

def dealtextslist(textspd):
	'''
	对文章内容进行处理，获得crf++ 所能进行测试的内容格式并保存
	:param textspd:dataframe的
	:return:无
	'''
	textslist = textspd['content'].values
	idslist = textspd['newsId'].values
	for text, id in tqdm(zip(textslist, idslist)):
		sents = sentence_preprocessing(text)
		str = ''
		for sent in sents:
			wordlist = []
			poslist = []
			seg = jieba.posseg.lcut(sent)
			for wordtag in seg:
				pattern = re.compile(u"[\u4e00-\u9fa5]+")
				result = re.findall(pattern, wordtag.word)
				if result != []:
					wordlist.append(wordtag.word)
					poslist.append(wordtag.flag)
			wordlist, poslist = getwordposlist(wordlist, poslist)
			for word, pos in zip(wordlist, poslist):
				str += word + '\t' + pos + '\n'
			str += '\n'
			# print(''.join(wordlist))
		savetestcrf(id, str)


def savetestcrf(id, str):
	'''
	保存字符串到id.txt中
	:param id: 保存的文件名称
	:param str:写进文件的字符串
	:return:无
	'''
	with open('resultdata\crftrain\\testdata_pos_ch\\' + id + '.txt', 'w', encoding='utf8') as file:
		file.write(str + '\n')




def formtestdatamain():
	'''
	加载测试数据，将数据转为dataframe格式，形成只有content和newsId列的测试数据
	对content列数据做一些预处理
	## 形成crf需要的格式（其中包含保存）
	:return:无
	'''
	start = time.time()
	test_data = load_data('data_all\data-11.23\DATA_TEST.txt')  # DATA_1018\small_data_test.txt
	testdata_pd = dict_to_df(test_data)
	
	texts_pd = pd.concat([testdata_pd['newsId'], combine(testdata_pd)], axis=1)
	texts_pd.rename(columns={0: 'content'}, inplace=True)
	texts_pd['content'] = texts_pd['content'].apply(textpreprocessing)
	# print(texts_pd['entities'])
	
	dealtextslist(texts_pd)
	
	print('total cost time:', time.time() - start, 's')

if __name__ == '__main__':
	formtestdatamain()
	
	# word = '2016-12-17'
	# pattern = re.compile(u"[\u4e00-\u9fa5]+")
	# result = re.findall(pattern, word)
	# if result != "":
	# 	print(result)
