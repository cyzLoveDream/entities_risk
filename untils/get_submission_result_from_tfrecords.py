import pandas as pd
import os
import json
import tensorflow as tf
import numpy as np
import time
from tqdm import  tqdm
from pprint import pprint

id_list = []
data_list = []
id_entity = {}
with open("../submission/sub_12_18.txt",encoding="utf-8") as rf:
	lines = rf.readlines()
	for line in lines:
		data_list.append(eval(line))
test_data_pd = pd.DataFrame()
for dl in data_list:
	# if len(dl.get("entities")) != 0:
	pd_dl = pd.DataFrame(dl.get("entities"),columns=["digest","name"])
	id_list.append(dl.get("newsId"))
	id_entity[dl.get("newsId")] = list(pd_dl.name.values)
	test_data_pd = pd.concat([test_data_pd,pd_dl])
test_data_pd = test_data_pd.fillna(value="missing")
print(test_data_pd.shape)

id2indexs = {}
test_indexs = test_data_pd.index.values
temp_index = []
id_list = []
i = 0
for k,v in id_entity.items():
	id2indexs[k] = list(range(len(v)))
	if v == []:
		temp_index.append(i)
	i += 1
	id_list.append(k)

def parase_tfrecords_to_dataFrame(filename,data_shape):
	"""
	解析预测文件
	:param filename:预测文件的路径
	:param data_shape: 预测文件内容的大小
	:return: 一个DataFrame格式的数据
	"""
	data_list = []
	with tf.Session() as sess:
		filename_queue = tf.train.string_input_producer([filename],shuffle=False)
		read = tf.TFRecordReader()
		_,serialized_example = read.read(filename_queue)
		
		features = tf.parse_single_example(serialized_example,
		                                   features={
			                                   "name": tf.FixedLenFeature([],tf.string),
			                                   "digest": tf.FixedLenFeature([], tf.string),
			                                   "label": tf.FixedLenFeature([],tf.int64),
			                                   })
		name = features['name']
		digest = features["digest"]
		label = tf.cast(features['label'],tf.int64)
		init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		# 23352
		for i in range(data_shape):
			content_list = sess.run([name,digest,label])
			c_l = []
			c_l.append(str(content_list[0],"utf-8"))
			c_l.append(str(content_list[1],"utf-8"))
			c_l.append(content_list[2])
			data_list.append(c_l)
		coord.request_stop()
		coord.join(threads)
		sess.close()
	data_pd = pd.DataFrame(data_list,columns=["name","digest","label"])
	data_pd["label"] = data_pd["label"].apply(lambda x: "正向" if x == 1 else ("负向" if x == 2 else "中性"))
	data_pd["indexs"] = test_indexs
	# data_pd.to_csv("final\\test_pred.csv",index=False)
	# print(data_pd.head(20))
	# print(data_pd.shape)
	return data_pd

def get_eventLevel_dict():
	"""
	创建实体与预测结果的字典
	:return: 返回一个字典
	"""
	data_pd = parase_tfrecords_to_dataFrame("../feature_data/submission_final_11.tfrecords",16913)
	# data_pd.to_csv("final\\sub.csv")
	# print(data_pd[data_pd.name.values == "北京知识产权法院"])
	labels = list(data_pd.label.values)
	digest = list(data_pd.digest.values)
	names = list(data_pd.name.values)
	indexs = list(data_pd.indexs.values)
	result_eventLevel_dict = {}
	eventLevel_dict = {}
	all_list = []
	package_list = []
	for i in range(len(names) - 1):
		tag = 0
			# 获取当前实体的名字与正负性
		if "局" in names[i] or "会" in names[i]:
			eventLevel_dict[names[i]] = "中性"
		else:
			eventLevel_dict[names[i]] = labels[i]
			if "不合格" in digest[i]:
				eventLevel_dict[names[i]] = "负向"
			else:
				if "合格" in digest[i]:
					eventLevel_dict[names[i]] = "正向"
				else:
					pass
			if "不符合" in digest[i]:
				eventLevel_dict[names[i]] = "负向"
			else:
				if "符合" in digest[i]:
					eventLevel_dict[names[i]] = "正向"
				else:
					pass
		if indexs[i+1] == tag:
			package_list.append(eventLevel_dict)
			all_list.append(package_list[0])
			package_list = []
			eventLevel_dict = {}
		else:
			package_list.append(eventLevel_dict)
	all_list.append(package_list[0])
	all_list[-1].update({names[-1]:labels[-1]})
	count = 0
	for i in range(2000):
		if i not in temp_index:
			result_eventLevel_dict[id_list[i]] = all_list[count]
			count += 1
		else:
			result_eventLevel_dict[id_list[i]] = {}
			# count -= 1
	# print(len(all_list[-1]))
	return result_eventLevel_dict
	
def main():
	eventLevel_dict = get_eventLevel_dict()
	# print(len(eventLevel_dict))
	# pprint(eventLevel_dict)
	result_data = []
	result_file = "../submission/sub_12_18.txt"
	with open(result_file,encoding="utf-8") as rf:
		lines = rf.readlines()
		for line in lines:
			result_data.append(eval(line))
	count = 0
	for rd in tqdm(result_data):
		count += 1
		entities = rd.get("entities")
		newsId = rd.get("newsId")
		# print(count,newsId)
		if len(entities) != 0:
			for e in entities:
				name = e.get("name")
				digist = e.get("digest")
				# print(eventLevel_dict.get(newsId).get(e.get("name")))
				e["eventLevel"] = eventLevel_dict.get(newsId).get(e.get("name"))
				e["keywords"] = ""
				del e["name"]
				del e["digest"]
				e["name"] = name
				e["digest"] = digist
			rd["entities"] = entities
	date = time.strftime("%m_%d",time.localtime(time.time()))
	# 通过扩展名指定文件存储的数据为json格式
	file_name = "../submission/final_baseline(perfer)_" + date + ".txt"
	with open(file_name,'w', encoding="utf-8") as file_object:
		for result in result_data:
			json.dump(result,file_object,ensure_ascii=False)
			file_object.write("\n")
	print("finish")

if __name__ == '__main__':
	main()