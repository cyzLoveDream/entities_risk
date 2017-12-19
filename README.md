# entities_risk


-------
	|
	|---auto_digest   ：做自动摘要相关的一些代码、模型等
	|		|
	|		|---------（未完待续。。。）
	|
	|------data
	|		|---DATA: 包含了初赛的训练集和测试集
	|		|
	|		|----NEGATIVE_WORDS.excel: 基于字典情感分析的情感词列表
	|		|
	|		|---others: 包含了复赛的训练集和测试集，（训练集初赛和一样）
	|
	|---data_cleans
	|		|
	|		|--------generate_test_data.py : 生成测试数据集格式，结果用来利用crf++进行预测
	|		|
	|		|--------generate_train_data.py: 生成训练集，其生成的数据需要使用crf++进行实体训练
	|		
	|---entity_extraction： 做实体抽取的相关代码模块、模型等
	|		|
	|		|-----------（未完待续）
	|		
	|---extra_data
	|		|
	|		|--------enties.txt: 实体字典 （已改进地方一：将实体字典与模型抽取出的字典进行融合，然后再预测出训练集的实体）
	|		|
	|		|--------user_dict.txt: 分词所使用的用户自定义字典，包含了所给的情感词
	|		
	|---feature_data
	|		|
	|		|-------testdata_pos_ch： 用来抽取实体的测试集原始数据，由data_cleans.generate_test_data.py 生成
	|		|
	|		|-------testresult: 存放已经经过crf++模型预测出来实体数据集
	|		|
	|		|-------submission_final_11.tfrecords: 经过情感深度学习模型预测出来的实体情感文件
	|		|
	|		|-------test.csv、train.csv: 测试集中出现的实体及它的摘要，用来生成情感学习模型需要的tsv格式, 为了处理里面出现的"\r"字符先转成tsv格式，再转tfrecords
	|		|
	|		|-------test.tfrecords、train.tfrecords、user_dict.tfrecords: 数据集的tfrecords格式
	|		|
	|		|-------test.tsv：测试集中出现的实体及它的摘要，用来生成情感学习模型需要的tfrecords格式
	|		|
	|		|-------traintokens_pos_ch.txt: 进行训练crf++模型的训练集,以单字为特征，并且加上了词性特征
	|
	|---model
	|		|-------train_tag_new_2.0: 训练出来的crf++实体抽取模型
	|
	|---sentiment_analysis：做情感分析的相关代码模块、模型等
	|		|
	|		|-------（未完待续）
	|
	|---submission
	|		|
	|		|-------final_baseline(perfer)_12_18.txt：最终的结果文件
	|		|
	|		|-------sub_12_18.txt: 不带实体情感的结果文件
	|
	|---untils
	|		|
	|		|-------cnn_clound.py: 利用摘要训练实体的情感（需改进的地方：加上关键字、采用预训练词向量进行情感的分类）
	|		|
	|		|-------data2tfrecords.py: 将使用到的数据集转换为tfrecords格式的数据
	|		|
	|		|-------get_entities.py: 从测试集中获取crf++模型抽取的实体结果文件并且保存到feature_data.testresult下面
	|		|
	|		|-------get_result_test_without_emotion.py: 得到submission.sub_12_18.txt文件
	|		|
	|		|-------get_submission_result_from_tfrecords.py 得到submission.final_baseline(perfer)_12_18.txt文件
	|		|
	|		|-------tfrecords2data.py: 验证tfrecords格式的数据
	|		|
	|		|-------tools.py: 该项目所使用的工具函数
	|
	|---others
	|		|
	|		|-------crf_test：利用模型进行测试数据生成结果文件，其他的都是该文件的依赖
	|		|
	|		|-------CRF++-0.58.rar： crf++的工具
	|
	|
备注：该项目的运行顺序是：
	（1）先运行data_cleans-generate_train_data.py 和data_cleans-generate_test_data.py,生成想要的模型训练的测试集和训练集
	（2）再运行crf_learn进行实体抽取的模型训练（在虚拟机1里进行训练）生成实体抽取模型
	（3）运行untils-get_entities.py 利用训练好的模型进行实体的抽取
	（4）运行untils-get_result_test_without_emotion.py 初步获取最终的结果文件，其中包括摘要、关键字等
	（5）运行untils-data2tfrecords.py 将所使用的数据集转换为tfrecords格式，其中包括训练集、测试集、字典等，为了进行情感的抽取
	（6）运行untils-cnn_clound.py 利用摘要训练实体情感分类模型，产生feature_data-submission_final_11.tfrecords文件
	（7）运行untils-get_submission_result_from_tfrecords.py 产生最终的结果文件
	
