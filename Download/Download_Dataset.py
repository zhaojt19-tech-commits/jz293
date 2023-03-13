import numpy as np
import requests
import csv
import torch
import matplotlib.pyplot as plt
from scipy import sparse
import urllib.request as urllibreq
from scipy import sparse
import  xml.dom.minidom
import zipfile
from tqdm import tqdm

if __name__ == '__main__':
    query_dir = './Data/query.xml'
    zip_dir = './Data/cache.zip'
    raw_dir = '/density.raw'

    path = './Pictures/'

    gene_list = []
    gene_id_list = []
    with open("query.csv", "rt") as csvfile:
        reader = csv.reader(csvfile)
        column = [row[1] for row in reader]
        gene_list = column[1:]

    with open("query.csv", "rt") as csvfile:        
        reader = csv.reader(csvfile)
        gene_id = [row[0] for row in reader]
        gene_id_list = gene_id[1:]
    # print(len(gene_list))
    # print(gene_list[0])
    # print(gene_id_list[0])
    # print(len(gene_id_list))

    # 下载.xml文件的链接片段
    url_0 = "http://api.brain-map.org/api/v2/data/query.xml?criteria=model::SectionDataSet,rma::criteria,[failed$eq%27false%27],genes[acronym$eq%27"
    url_1 = "%27],products[abbreviation$eq%27DevMouse%27],specimen(donor(age[name$eq%27"
    url_2 = "%27]))"

    # 下载.zip文件的链接片段
    url_3 = "http://api.brain-map.org/grid_data/download/"
    url_4 = "?include=density"

    # time_str = 'E11.5'
    cache_list = []
    time_str_list = ['E18.5', 'P4', 'P14', 'P56']
    for time_str in tqdm(time_str_list):
        gene_id_non_blank = []

        save_dir = './Data_matrix/' + time_str + '_ish_data_density_csc.npz'
        save_id_dir = './Data_id/' + time_str + '_id.npy'

        t = 0
        for gene in tqdm(gene_list):
            url_xml = url_0 + gene + url_1 + time_str + url_2
            urllibreq.urlretrieve(url_xml, query_dir)
            # 打开xml文档
            dom = xml.dom.minidom.parse(query_dir)
            # 得到文档元素id对象
            root = dom.documentElement
            id_Node = root.getElementsByTagName('id')
            if(id_Node.length == 0):
                # 重复进行5次请求
                for k in range(5):
                    urllibreq.urlretrieve(url_xml, query_dir)
                    # 打开xml文档
                    dom = xml.dom.minidom.parse(query_dir)
                    # 得到文档元素id对象
                    root = dom.documentElement
                    id_Node = root.getElementsByTagName('id')
                    if(id_Node.length != 0):
                        break
                if(id_Node.length == 0):
                    t = t + 1
                    continue
                else:
                    pass
            if(time_str == 'E15.5' and t == 1751):
                pass
            else:
                gene_id_non_blank.append(gene_id_list[t])
            # id_str = id_Node[0].firstChild.data
            # url_zip = url_3 + id_str + url_4
            # urllibreq.urlretrieve(url_zip, zip_dir)
            # zip = zipfile.ZipFile(zip_dir)
            # zip.extract('density.raw', './Data/')
            # img = np.fromfile(file='./Data/density.raw', dtype=np.float32).reshape(1, -1)
            # img[img == -1] = 0
                
            # if(t == 0):
            #     arr = img
            #     zip.extract('density.mhd', './Data/')
            # else:
            #     if(img.shape[1] != arr.shape[1]):
            #         cache_list.append(t)
            #         t = t + 1
            #         continue
            #     arr = np.concatenate((arr, img), axis=0)
            t = t + 1

        # print(arr.shape)
        # arr = sparse.csc_matrix(arr)
        # sparse.save_npz(save_dir, arr)
        # print(cache_list)
        gene_id_non_blank = np.array(gene_id_non_blank)
        print(gene_id_non_blank.shape)
        np.save(save_id_dir, gene_id_non_blank)


        # url_test = "http://api.brain-map.org/grid_data/download/100072539?include=density"
