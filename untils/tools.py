# -*- coding:utf-8 -*-
import os
import re
from tqdm import tqdm
import pandas as pd
import warnings
import time
import json
warnings.filterwarnings("ignore")
cutlist = list("。！？;； ")

def get_entity_label_list(wordlist, entities):
	"""
	获取实体的list
	:param wordlist:
	:param entities:
	:return:
	"""
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
			continue
		if mark == 1:
			if i < m + len(markentity)-1:
				entitylabellist.append('I')
			elif i == m + len(markentity)-1:
				entitylabellist.append('E')
				mark = 0
	return entitylabellist

def get_word_pos_list(wordlist, poslist):
	"""
	获取词性list
	:param wordlist:
	:param poslist:
	:return:
	"""
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

def sentence_preprocessing(sentence):
	"""
	处理句子
	:param sentence:
	:return:
	"""
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

def textpreprocessing(text):
	"""
	替换不合理的标点
	:param text:
	:return:
	"""
	pattern = re.compile('<.*?>')
	text = pattern.sub('', text)
	return text.replace('　', '').replace('\n', '').strip()

def get_all_file_list(path):
    """
    读取给定path下的所有文件名，而不直接给出来
    :return:给定path下的所有文件名list
    """
    allfile = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            allfile.append(filepath)
    return allfile

def get_entity_result(input_file):
	"""
	:param inputfile:
	:return:
	"""
	lines = []
	with open(input_file,'r',encoding='utf8') as f:
		for line in f.readlines():
			lines.append(line.strip('\n'))
	ner_list = []
	phrase_list = []
	for li in lines:
		word_list = li.split("\t")
		# print(word_list)
		# 判断是否单个词是否是命名实体
		if len(word_list) > 1:
			if re.search(r'^S',word_list[-1]):
				ner_list.append(word_list[0])
			# 判断多个词构成的命名实体，并对其进行连词处理
			elif re.search(r'^B',word_list[-1]):
				phrase_list.append(word_list[0])
			elif re.search(r'^I',word_list[-1]):
				phrase_list.append(word_list[0])
			elif re.search(r'^E',word_list[-1]):
				phrase_list.append(word_list[0])
				# 把list转换为字符串.
				ner_phrase = ''.join(phrase_list)
				ner_list.append(ner_phrase)
				phrase_list = []
	ner_list = set(ner_list)
	# print(ner_list)
	return ner_list

def get_test_data():
	test_data = []
	with open("../data/DATA_TEST.txt", 'r', encoding='utf-8') as rf1:
		result_lines = rf1.readlines()
		for l in tqdm(result_lines):
			test_data.append(eval(l))
	test_data_pd = pd.DataFrame(test_data)
	test_data_pd["combine_title_body"] = test_data_pd["title"].apply(lambda x: x.strip()) + "。" + test_data_pd[
		"body"].apply(lambda x: x.strip())
	test_data_pd['combine_title_body'] = test_data_pd['combine_title_body'].apply(textpreprocessing)
	return test_data_pd

def cut_sentences(lines):  # 参数：被分句的文本，为一行中文字符
	global cutlist
	l = []  # 句子列表，用于存储单个分句成功后的整句内容，为函数的返回值
	line = []  # 临时列表，用于存储捕获到分句标志符之前的每个字符，一旦发现分句符号后，就会将其内容全部赋给l，然后就会被清空
	for char in lines:  # 对函数参数2中的每一字符逐个进行检查 （本函数中，如果将if和else对换一下位置，会更好懂）
		if char in cutlist:
			line.append(char)
			l.append(''.join(line))
			line = []
		else:
			line.append(char)
	if line != []:
		# print(''.join(line))
		l.append(''.join(line))
	return l


def get_results(texts_pd,match_entities,result_file_path):
	idlist = texts_pd['newsId'].values
	entitylist = texts_pd['entity'].values
	contentslist = texts_pd['combine_title_body'].values
	titlelist = texts_pd['title'].values
	result_list = []
	for id,entity,sents,title in tqdm(zip(idlist,entitylist,contentslist,titlelist)):
		json_data = {}
		json_data["newsId"] = id
		json_data["entities"] = []
		# print('entity', entity)
		entity += match_entities
		# entity = matchentities
		entity = set(entity)
		if entity != None and (len(entity) > 0):
			entitydic = {}
			for sent in sents:
				for en in entity:
					# 对实体进行长度限制
					if len(en) >= 5 and len(en) <= 17:
						temp = {}
						if en in sent and en not in entitydic.keys():
							temp["name"] = en
							if len(sent) > 300:
								# digestlist = re.split(r'[。；：;\s]',sent.strip())
								temp["digest"] = title
							else:
								temp["digest"] = sent
							json_data["entities"].append(temp)
							entitydic[en] = 1
		result_list.append(json_data)
	with open(result_file_path,'w',encoding='utf8') as file:
		for rs in result_list:
			json.dump(rs,file,ensure_ascii=False)
			file.write('\n')
