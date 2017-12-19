# !/usr/bin/python
# -*- coding:utf-8 -*-

import time, re, os, jieba
# os.chdir('E:\heting\ccf-competition\BDCI2017-fahainew')

import pandas as pd
import jieba
import jieba.posseg
from tqdm import tqdm

# 设置分句的标志符号；可以根据实际需要进行修改
cutlist = list("。！？;； ")
print(cutlist)

'''

将给出的训练集转换成 crf++ 所需要的训练格式
'''

def load_data(file):
	# train_data_file = 'DATA_1018\small_data_test.txt'  # DATA_T.txt'
	# train_data_result_file = 'DATA_1018\DATA_T\DATA_T_RESULT.txt'
	for line in open(file, 'r', encoding='utf8'):
		yield eval(line)

'''
通过给定的标点符号进行分句
'''
def sentence_preprocessing(sentence):
	global cutlist
	l = []  # 句子列表，用于存储单个分句成功后的整句内容，为函数的返回值
	line = []  # 临时列表，用于存储捕获到分句标志符之前的每个字符，一旦发现分句符号后，就会将其内容全部赋给l，然后就会被清空
	for char in sentence:  # 对函数参数2中的每一字符逐个进行检查 （本函数中，如果将if和else对换一下位置，会更好懂）
		if char in cutlist:
			line.append(char)
			l.append(''.join(line))
			line = []
		else:
			line.append(char)
	if line!=[]:
		l.append(''.join(line))
	return l

def dict_to_df(data_list):
	data_pd = pd.DataFrame(data_list)
	print(data_pd.shape)
	return data_pd

## 处理两列数据，使之形成一列
def combine(pddata):
	list1 = pddata['title'].values
	list2 = pddata['body'].values
	list = []
	for data1, data2 in zip(list1, list2):
		list.append(data1 + '。' + data2)
	# print(pd.Series(list))
	return pd.Series(list)

def getentity(entities):
	# entity =
	return [dict['name'] for dict in entities]

def textpreprocessing(text):
	pattern = re.compile('<.*?>')
	text = pattern.sub('', text)
	return text.replace('　', '').replace('\n', '').strip()

def dealsentslist(textspd):
	trainstring = []
	textslist = textspd['content'].values
	entitieslist = textspd['entities'].values
	
	for text, entities in tqdm(zip(textslist, entitieslist)):
		sents = sentence_preprocessing(text)
		for sent in sents:
			str = ''
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
			entitylabellist = getentitylabellist(wordlist, entities)
			for word, pos, entitylabel in zip(wordlist, poslist, entitylabellist):
				str += word + '\t' + pos + '\t' + entitylabel + '\n'
			trainstring.append(str)
	return trainstring

def getwordposlist(wordlist, poslist):
	list1 = []
	list2 = []
	for word, flag in zip(wordlist, poslist):
		wordseg = list(word)
		list1.extend(wordseg)
		if len(wordseg) > 1:
			list2.append('B' + flag)
			for i in range(1, len(wordseg) - 1):
				list2.append('M' + flag)
			list2.append('E' + flag)
		else:
			if(flag=='x'):flag='w'
			list2.append('S' + flag)
	return list1, list2

def getentitylabellist(wordlist, entities):
	entitylabellist = []
	mark = 0
	m = 0
	for i in range(len(wordlist)):
		if mark==0:
			for entity in entities:
				if entity[0]==wordlist[i] and ''.join(wordlist[i : i+len(entity)]) == entity:
					entitylabellist.append('B')
					markentity = entity
					mark = 1
					m = i
					break
			else:
				entitylabellist.append('O')
				# i += 1
			continue
		if mark == 1:
			# print(markentity)
			if i < m + len(markentity)-1:
				entitylabellist.append('I')
			elif i == m + len(markentity)-1:
				entitylabellist.append('E')
				mark = 0
	# print(entitylabellist)
	return entitylabellist

def formtrain(wordlist, poslist, entitylabellist):
	str = ''
	for word, pos, entitylabel in zip(wordlist, poslist, entitylabellist):
		str += word +'\t' + pos + '\t' +  entitylabel + '\n'
	return str

def savedata(datalist):
	with open('resultdata\crftrain\\traintokens_pos_ch.txt', 'w', encoding='utf8') as file:
		for data in datalist:
			# print(data)
			file.write(data + '\n')


def fromtraindatamain():
	start = time.time()
	train_data = load_data('data_all\data-11.23\DATA_T.txt')  # DATA_1018\small_data_test.txt # 获取训练数据到list
	traindata_result = load_data('data_all\data-11.23\DATA_T_RESULT.txt')  # 获取训练数据的结果到list
	traindata_pd = dict_to_df(train_data)  # 将训练数据转换为dataframe
	traindata_result_pd = dict_to_df(traindata_result)  # 将训练数据结果转换为dataframe
	data_pd = pd.merge(traindata_pd, traindata_result_pd, on=['newsId'])  # 根据id信息合并训练数据和结果
	texts_pd = pd.concat([combine(data_pd), data_pd['entities']], axis=1)  # 将文章标题和正文合并形成DF并和实体列做连接形成新的DF
	texts_pd.rename(columns={0: 'content'}, inplace=True)  # 对列重命名
	texts_pd['content'] = texts_pd['content'].apply(textpreprocessing)  #预处理content数据
	texts_pd['entities'] = texts_pd['entities'].apply(getentity)  #处理entity，形成只包含entity的list
	
	# print(texts_pd['content'].values[:10])
	
	trainstring = dealsentslist(texts_pd)
	savedata(trainstring)
	
	print('total cost time:', time.time() - start, 's')


# print(text_pd[0].values)


if __name__ == '__main__':
	# str = getcrfdata('&省卫生计生监督局&公共场所卫生监督科副科长章燕介绍，为保障广大百姓游泳卫生安全，预防传染病传播，全省各级卫生计生监督机构在今年5月至7月间，对全省318家游泳场所进行了专项监督检查，并在相关技术服务机构的配合下，对正常营业的306家游泳场所的水质进行了抽检，检测结果显示：247家水质各项指标均符合国家标准，合格率为80.7%，59家水质的个别指标未达到国家卫生标准要求。')
	# print(str)
	# str = '国家食品药品监督管理总局'
	# print(str[0])
	
	fromtraindatamain()
	