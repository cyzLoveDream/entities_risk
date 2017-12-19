# !/usr/bin/python
# -*- coding:utf-8 -*-

import os, re, tqdm
os.chdir('E:\\entity_risk')

crf_test = 'crf_test '
model = '-m model/train_tag_new_2.0 '

def getallfilelist(path):
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

if __name__ == '__main__':
    commod = crf_test + model
    filenamelist = getallfilelist("feature_data\\testdata_pos_ch")
    for file in tqdm.tqdm(filenamelist):
        outfile = 'feature_data\\testresult\\' + re.findall(r"testdata_pos_ch\\(.+?)\.txt", file)[0] + '.txt'
        os.system(commod + file + ' > ' + outfile)
