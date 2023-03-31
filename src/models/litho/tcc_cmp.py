"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-03-31 11:19:58
LastEditTime: 2023-03-31 14:26:12
Contact: cgjcuhk@gmail.com
Description: 
"""
from tcc import TCCDB
from utils import torch_arr_bound, arr_bound


tcc_db_paths = [
    './db/sci.sparse.svd.tcc',
    './db/torch.linalg.svd.tcc'
]


np_tdb = TCCDB(tcc_db_paths[0])
np_tdb.load_db()

tdb = TCCDB(tcc_db_paths[1])
tdb.load_db()


for i in range(len(tdb.kernelList)):
    arr_bound(np_tdb.coefList[i], f"np_tdb.coefList[{i}]")
    arr_bound(np_tdb.kernelList[i], f"np_tdb.kernelList[{i}]")

    # torch_arr_bound(tdb.coefList[i], f"tdb.coefList[{i}]")
    # torch_arr_bound(tdb.kernelList[i], f"tdb.kernelList[{i}]")



