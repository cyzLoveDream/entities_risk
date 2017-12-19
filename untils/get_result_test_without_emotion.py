# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
import jieba
import time
import re
from tqdm import tqdm
from untils.tools import get_all_file_list
from untils.tools import get_entity_result
from untils.tools import get_test_data
from untils.tools import cut_sentences
from untils.tools import get_results
import warnings

warnings.filterwarnings("ignore")

def main():
	file_list = get_all_file_list("../feature_data/testresult/")
	id_list = []
	entity_list = []
	print("\nbegin handle result files..")
	for file in tqdm(file_list):
		id = re.findall(r"testresult/(.+?)\.txt",file)[0]
		entity = get_entity_result(file)
		id_list.append(id)
		entity_list.append(list(entity))
	alldata_pd = pd.DataFrame({'newsId': id_list,'entity': entity_list})
	texts_pd = get_test_data()
	texts_pd = alldata_pd.merge(texts_pd, how="left",on="newsId")
	texts_pd['combine_title_body'] = texts_pd['combine_title_body'].apply(cut_sentences)
	match_entities = []
	print("begin load enties dict...")
	with open("../extra_data/enties.txt",'r',encoding='utf8') as rf:
		lines = rf.readlines()
		for line in lines:
			match_entities.append(line[:-1])
	print("finish load enties...")
	date = time.strftime("%m_%d",time.localtime(time.time()))
	result_file_path = "../submission/sub_" + date +".txt"
	get_results(texts_pd, match_entities, result_file_path)
	pass

if __name__ == '__main__':
	main()