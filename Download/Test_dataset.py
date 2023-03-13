import numpy as np
import csv
import torch
import matplotlib.pyplot as plt
from scipy import sparse
import urllib.request as urllibreq
import  xml.dom.minidom
import zipfile
from tqdm import tqdm

if __name__ == '__main__':
    ish_arr = sparse.load_npz('./Data_matrix/E18.5_ish_data_density_csc.npz')
    
    arr = ish_arr.toarray().reshape(-1, 40, 43, 67)  # 取出基因

    # arr = np.load('./Data_matrix/E15.5_1751.npy').reshape(1, 47, 134, 134)

    # print(arr.max())
    print(arr.shape)

    path = './Pictures/'
    for i in range(0, 2045, 10):
        dir = path + str(i) + '.png'
        plt.imsave(dir, arr[i][18])

    # for i in range(47):
    #     dir = path + str(i) + '.png'
    #     plt.imsave(dir, arr[0][i])

    # query_dir = './Data/query.xml'
    # zip_dir = './Data/cache.zip'
    # raw_dir = '/density.raw'

    # path = './Pictures/'

    # gene_list = []
    # with open("query.csv", "rt") as csvfile:
    #     reader = csv.reader(csvfile)
    #     column = [row[1] for row in reader]
    #     gene_list = column[1:]
    # # 下载.xml文件的链接片段
    # url_0 = "http://api.brain-map.org/api/v2/data/query.xml?criteria=model::SectionDataSet,rma::criteria,[failed$eq%27false%27],genes[acronym$eq%27"
    # url_1 = "%27],products[abbreviation$eq%27DevMouse%27],specimen(donor(age[name$eq%27"
    # url_2 = "%27]))"

    # # 下载.zip文件的链接片段
    # url_3 = "http://api.brain-map.org/grid_data/download/"
    # url_4 = "?include=density"

    # # time_str = 'E11.5'
    # cache_list = []
    # time_str_list = ['E15.5']
    # for time_str in tqdm(time_str_list):

    #     save_dir = './Data_matrix/' + time_str + '_ish_data_density_csc.npz'

    #     t = 0
    #     for gene in tqdm([gene_list[1751]]):
    #         url_xml = url_0 + gene + url_1 + time_str + url_2
    #         urllibreq.urlretrieve(url_xml, query_dir)
    #         # 打开xml文档
    #         dom = xml.dom.minidom.parse(query_dir)
    #         # 得到文档元素id对象
    #         root = dom.documentElement
    #         id_Node = root.getElementsByTagName('id')
    #         if(id_Node.length == 0):
    #             # 重复进行5次请求
    #             for k in range(5):
    #                 urllibreq.urlretrieve(url_xml, query_dir)
    #                 # 打开xml文档
    #                 dom = xml.dom.minidom.parse(query_dir)
    #                 # 得到文档元素id对象
    #                 root = dom.documentElement
    #                 id_Node = root.getElementsByTagName('id')
    #                 if(id_Node.length != 0):
    #                     break
    #             if(id_Node.length == 0):
    #                 t = t + 1
    #                 continue
    #             else:
    #                 pass
    #         id_str = id_Node[0].firstChild.data
    #         url_zip = url_3 + id_str + url_4
    #         urllibreq.urlretrieve(url_zip, zip_dir)
    #         zip = zipfile.ZipFile(zip_dir)
    #         zip.extract('density.raw', './Data/')
    #         img = np.fromfile(file='./Data/density.raw', dtype=np.float32).reshape(1, -1)
    #         img[img == -1] = 0
                
    #         if(t == 0):
    #             arr = img
    #             zip.extract('density.mhd', './Data/')
    #         else:
    #             if(img.shape[1] != arr.shape[1]):
    #                 cache_list.append(t)
    #                 t = t + 1
    #                 continue
    #             arr = np.concatenate((arr, img), axis=0)
    #         t = t + 1
    #     print(arr.shape)
    #     np.save('./Data_matrix/E15.5_1751.npy', arr)
        # arr = sparse.csc_matrix(arr)
        # sparse.save_npz(save_dir, arr)
        # print(cache_list)