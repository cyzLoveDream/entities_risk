# -*- coding:utf-8 -*-
import pandas as pd
from tqdm import tqdm
import jieba
import jieba.posseg
import time
import warnings
from data_cleans.generate_train_data import sentence_preprocessing
from data_cleans.generate_train_data import getwordposlist
from data_cleans.generate_train_data import textpreprocessing
warnings.filterwarnings("ignore")
cutlist = list("。！？;； ")

def dealtextslist(textspd):
	'''
	对文章内容进行处理，获得crf++ 所能进行测试的内容格式并保存
	:param textspd:dataframe
	:return:无
	'''
	textslist = textspd['combine_title_body'].values
	idslist = textspd['newsId'].values
	for text, id in tqdm(zip(textslist, idslist)):
		sents = sentence_preprocessing(text)
		str = ''
		for sent in sents:
			wordlist = []
			poslist = []
			seg = jieba.posseg.lcut(sent)
			for wordtag in seg:
				# pattern = re.compile(u"[\u4e00-\u9fa5]+")
				# result = re.findall(pattern, wordtag.word)
				# if result != []:
				wordlist.append(wordtag.word)
				poslist.append(wordtag.flag)
			wordlist, poslist = getwordposlist(wordlist, poslist)
			for word, pos in zip(wordlist, poslist):
				str += word + '\t' + pos + '\n'
			str += '\n'
			# print(''.join(wordlist))
		with open('../feature_data/testdata_pos_ch/' + id + '.txt', 'w', encoding='utf8') as file:
			file.write(str + '\n')
def generate_test(file_path):
	start = time.time()
	raw_data = []
	with open(file_path, 'r', encoding="utf-8") as rf:
		lines = rf.readlines()
		for line in  lines:
			raw_data.append(eval(line))
	raw_data_pd = pd.DataFrame(raw_data)
	raw_data_pd["combine_title_body"] = raw_data_pd["title"].apply(lambda x: x.strip()) + "。" + raw_data_pd[
		"body"].apply(lambda x: x.strip())
	raw_data_pd["combine_title_body"] = raw_data_pd["combine_title_body"].apply(textpreprocessing)
	dealtextslist(raw_data_pd)
	print('total cost time: %0.4f' % (time.time() - start),'s')
def main():
	generate_test("../data/DATA/DATA_TEST.txt")
	pass

if __name__ == '__main__':
	main()