import pandas as pd
import warnings
import numpy as np
import timeit  # 查看运行开始到结束所用的时间
import tensorflow as tf
import os

warnings.filterwarnings("ignore")
os.chdir("E:\\ccf\\Entity_Extraction")


def eval_test_tfrecords_data(filename, data_shape):
	"""
	解析转化的测试集tfrecords文件
	:param filename: tfrecords文件的路径
	:param data_shape: 文件内容的数目
	:return: 一个DataFrame格式的数据
	"""
	data_list = []
	with tf.Session() as sess:
		# file_name = "raw_data\\train_new.tfrecords"
		filename_queue = tf.train.string_input_producer([filename],shuffle=False,seed=0)
		read = tf.TFRecordReader()
		_,serialized_example = read.read(filename_queue)

		features = tf.parse_single_example(serialized_example,
		                                   features={
			                                   "digest": tf.FixedLenFeature([],tf.string),
			                                   "name": tf.FixedLenFeature([],tf.string),
			                                   })
		digest = features['digest']
		name = features["name"]
		init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		for i in range(data_shape):
			content_list = sess.run([digest, name])
			c_l = []
			for d in content_list:
				c_l.append(str(d,"utf-8"))
			data_list.append(c_l)
		coord.request_stop()
		coord.join(threads)
		sess.close()
	data_pd = pd.DataFrame(data_list, columns=["digest","name"])
	# data_pd.to_csv("raw_data\\evaltf.csv",header=False,index=False)
	print(data_pd.head())
	print(data_pd.shape)
	return data_pd


def eval_train_tfrecords_data(filename,data_shape):
	"""
	解析训练集的tfrecords文件
	:param filename: tfrecords文件的路径
	:param data_shape: 文件的内容的数量
	:return: 一个DataFrame格式的数据
	"""
	data = []
	with tf.Session() as sess:
		# file_name = "raw_data\\train_tf.tfrecords"
		filename_queue = tf.train.string_input_producer([filename],shuffle=False,seed=0)
		read = tf.TFRecordReader()
		_,serialized_example = read.read(filename_queue)
		
		features = tf.parse_single_example(serialized_example,
		                                   features={
			                                   "label": tf.FixedLenFeature([],tf.int64),
			                                   "digest": tf.FixedLenFeature([],tf.string),
			                                   "eventLevel": tf.FixedLenFeature([],tf.string),
			                                   "keywords": tf.FixedLenFeature([],tf.string),
			                                   "name": tf.FixedLenFeature([],tf.string),
			                                   })
		label = tf.cast(features['label'],tf.int64)
		digest = features['digest']
		name = features["name"]
		keyword = features['keywords']
		eventLevel = features['eventLevel']
		init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		# 14170
		for i in range(data_shape):
			content_list = sess.run([name,digest,keyword,eventLevel,label])
			c_l = []
			for d in content_list[0:-1]:
				c_l.append(str(d,"utf-8"))
			c_l.append(content_list[-1])
			data.append(c_l)
		coord.request_stop()
		coord.join(threads)
		sess.close()
	data_pd = pd.DataFrame(data,columns=["name","digest","keyword","eventLevel","label"])
	return data_pd

def eval_user_dict_tfrecords(input_filename, data_shape):
	"""
	解析用户字典所形成的tfrecords文件
	:param input_filename: tfrecords文件的路径
	:param data_shape: 原始文件的大小
	:return: 文件内容的list
	"""
	data = []
	with tf.Session() as sess:
		# file_name = "raw_data\\train_tf.tfrecords"
		filename_queue = tf.train.string_input_producer([input_filename],shuffle=False)
		read = tf.TFRecordReader()
		_,serialized_example = read.read(filename_queue)
		features = tf.parse_single_example(serialized_example,
		                                   features={
			                                   "line": tf.FixedLenFeature([],tf.string),
			                                   })
		line = features['line']
		init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		for i in range(data_shape):
			lines = str(sess.run(line),"utf-8")
			data.append(lines)
		coord.request_stop()
		coord.join(threads)
		sess.close()
	print(len(data))
	print(data)
	return data


def main():
	pass

if __name__ == '__main__':
	main()