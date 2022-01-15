import os.path as osp

dataroot = 'data/'
data_list = 'test_pairs.txt'
new_file_name = 'data/new_test_pairs.txt'
new_f = open(new_file_name, 'w')
with open(osp.join(dataroot, data_list), 'r') as f:
    for line in f.readlines():
        im_name, c_name = line.strip().split()
        im_index = im_name[: im_name.find('_') + 1]
        new_f.write('%s0.jpg %s1.jpg\n' % (im_index, im_index))