'''
数据预处理
将黑白PHP文件转化为操作符序列并储存到txt中

'''

from utils import *
import os

'''
递归遍历目标文件夹内的所有的PHP文件，
将其转化为操作符序列并写入到文件中
'''


def recursion_trans_php_file_opcode(dir, write_dir):

    print('开始生成 {} 路径中的PHP的opcode操作码文件'.format(dir))

    with open(write_dir, 'w') as fw:
        for root, dirs, files in os.walk(dir):
            for filename in files:
                if filename.endswith('.php'):
                    try:
                        full_path = os.path.join(root, filename)
                        file_content = load_php_opcode(full_path)
                        if(file_content == ''):  # 空文件或读取失败时跳过该文件
                            continue
                        fw.write(file_content + '\n')
                    except:
                        continue


def prepare_data():
    recursion_trans_php_file_opcode(good_dir, good_ops_dir)
    recursion_trans_php_file_opcode(bad_dir, bad_ops_dir)


if __name__ == '__main__':
    prepare_data()
