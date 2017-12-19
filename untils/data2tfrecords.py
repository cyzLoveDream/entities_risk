import pandas as pd
import warnings
import numpy as np
import timeit   #查看运行开始到结束所用的时间
import tensorflow as tf
import os
warnings.filterwarnings("ignore")


def generate_train_tfrecords(input_filename,output_filename,label_col_index):
	"""
	转化训练数据成tfrecords
	:param input_filename: 输入文件的路径，csv文件
	:param output_filename: 输出tfrecords文件的路径
	:param label_col_index: 标签列
	:return:
	"""
	print("\nStart to convert {} to {}\n".format(input_filename,output_filename))
	start_time = timeit.default_timer()
	writer = tf.python_io.TFRecordWriter(output_filename)
	num = 0
	for line in open(input_filename,"r"):
		num += 1
		data = line.split(",")
		label = int(data[label_col_index])
		digest = [bytes(str(data[0]),"utf-8")]
		eventLevel = [bytes(str(data[1]),"utf-8")]
		keywords = [bytes(str(data[2]),"utf-8")]
		name = [bytes(str(data[3]),"utf-8")]
		# print(name)
		# 将数据转化为原生 bytes
		example = tf.train.Example(features=tf.train.Features(feature={
			"label":
				tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
			"digest":
				tf.train.Feature(bytes_list=tf.train.BytesList(value=digest)),
			"eventLevel":
				tf.train.Feature(bytes_list=tf.train.BytesList(value=eventLevel)),
			"keywords":
				tf.train.Feature(bytes_list=tf.train.BytesList(value=keywords)),
			"name":
				tf.train.Feature(bytes_list=tf.train.BytesList(value=name))
			}))
		writer.write(example.SerializeToString())  # 序列化为字符串
	writer.close()
	print("Successfully convert {} to {},the num is {}".format(input_filename,output_filename,num))
	end_time = timeit.default_timer()
	print("\nThe pretraining process ran for {0} minutes\n".format((end_time - start_time) / 60))

def generate_tst_tfrecords(input_filename,output_filename):
	"""
	转化测试数据到tfrecords
	:param input_filename: 输入文件的路径
	:param output_filename: 输出文件的路径
	:return:
	"""
	print("\nStart to convert {} to {}\n".format(input_filename,output_filename))
	start_time = timeit.default_timer()
	writer = tf.python_io.TFRecordWriter(output_filename)
	num = 0
	with open(input_filename, mode='r', encoding="utf-8") as f:
		lines = f.readlines()
		for line in lines:
			if line != "\n":
				num += 1
				data = line.split("\t")
				# print(data)
				digest = [bytes(str(data[0]),"utf-8")]
				name = [bytes(str(data[1].strip()),"utf-8")]
				# 将数据转化为原生 bytes
				example = tf.train.Example(features=tf.train.Features(feature={
					"digest":
						tf.train.Feature(bytes_list=tf.train.BytesList(value=digest)),
					"name":
						tf.train.Feature(bytes_list=tf.train.BytesList(value=name))
					}))
				writer.write(example.SerializeToString())  # 序列化为字符串
	writer.close()
	print("Successfully convert {} to {}, the number is {}".format(input_filename,output_filename,num))
	end_time = timeit.default_timer()
	print("\nThe pretraining process ran for {0} minutes\n".format((end_time - start_time) / 60))

def generate_user_dict_tfrecords(input_filename,output_filename):
	"""
	转化自定义字典到tfrecords
	:param input_filename: 字典的输入路径
	:param output_filename: 转化后的输出路径
	:return:
	"""
	print("\nStart to convert {} to {}\n".format(input_filename,output_filename))
	start_time = timeit.default_timer()
	writer = tf.python_io.TFRecordWriter(output_filename)
	num = 0
	for line in open(input_filename,"r",encoding="utf-8"):
		num += 1
		line = [bytes(str(line.strip()),"utf-8")]
		example = tf.train.Example(features=tf.train.Features(feature={
			"line":
				tf.train.Feature(bytes_list=tf.train.BytesList(value=line))
			}))
		writer.write(example.SerializeToString())  # 序列化为字符串
	writer.close()
	print("Successfully convert {} to {}".format(input_filename,output_filename))
	end_time = timeit.default_timer()
	print("\nThe pretraining process ran for {0} minutes\n".format((end_time - start_time) / 60))


def convert_tst():
	"""
	直接调用此函数转化测试集
	:return:
	"""
	data_list = []
	with open("../submission/sub_12_18.txt",encoding="utf-8") as rf:
		lines = rf.readlines()
		for line in lines:
			data_list.append(eval(line))
	data_pd = pd.DataFrame()
	for dl in data_list:
		pd_dl = pd.DataFrame(dl.get("entities"))
		data_pd = pd.concat([data_pd,pd_dl])
	data_pd = data_pd.fillna(value="missing")
	# data_pd.to_csv("raw_data\\test.csv",index=False,header=False)
	data_pd_list = data_pd.values
	with open("../feature_data/test.tsv",mode="w+",encoding="utf-8") as wf:
		for dpl in data_pd_list:
			line = dpl[0] + "\t" + dpl[1] + "\n"
			wf.write(line)
	print(data_pd.head())
	print(data_pd.shape)
	print(data_pd.isnull().sum())
	generate_tst_tfrecords("../feature_data/test.tsv","../feature_data/test.tfrecords")

def convert_train():
	"""
	调用此函数转化训练集
	:return:
	"""
	data_list = []
	with open("../data/DATA_T_RESULT.txt",encoding="utf-8") as rf:
		lines = rf.readlines()
		for line in lines:
			data_list.append(eval(line))
	data_pd = pd.DataFrame()
	for dl in data_list:
		pd_dl = pd.DataFrame(dl.get("entities"))
		data_pd = pd.concat([data_pd,pd_dl])
	# user_dict = pd.read_excel("raw_data\\DATA_1018\\DATA_1018\\NEGATIVE_WORDS.xls",encoding='utf-8')
	data_pd["label"] = data_pd["eventLevel"].apply(lambda x: 1 if x == "正向" else (0 if x == "中性" else 2))
	# print(data_pd.isnull().sum)
	data_pd = data_pd.fillna(value="missing")
	data_pd.to_csv("../feature_data/train.csv",index=False,header=False)
	generate_train_tfrecords("../feature_data/train.csv","../feature_data/train.tfrecords",4)

def main():
	generate_user_dict_tfrecords("../feature_data/user_dict.txt","../extra_data/user_dict.tfrecords")
	# convert_train()
	convert_tst()
	
	pass
if __name__ == '__main__':
	main()
