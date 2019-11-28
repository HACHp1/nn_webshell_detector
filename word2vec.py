'''
功能：
1. 使用word2vec将字符串文件向量化
2. 将向量化的结果储存在bin路径下
'''

from gensim.models.word2vec import Word2Vec
from utils import *
import pickle


FAST_LOAD = False  # 是否要重新训练w2v


def most_similar(w2v_model, word, topn=10):
    try:
        similar_words = w2v_model.wv.most_similar(word, topn=topn)
    except:
        print(word, "not found in Word2Vec model!")
    return similar_words


if __name__ == '__main__':
    if (not FAST_LOAD):
        y = []
        payloads = []
        payloads_seged = []
        lens = []

        # 加载非恶意数据
        with open(good_ops_dir) as fr:
            while(1):
                payload = fr.readline()
                if(payload == '\r\n' or payload == '\n' or payload == '\r'):
                    continue
                if(not payload):
                    good_len = len(payloads)
                    print('[num] 非恶意数据量为：', good_len)
                    break
                payload = payload.strip()
                payloads.append(payload)
                y.append(0)

        # 加载恶意数据
        with open(bad_ops_dir) as fr:
            while(1):
                payload = fr.readline()
                if(payload == '\r\n' or payload == '\n' or payload == '\r'):
                    continue
                if(not payload):
                    print('[num] 恶意数据量为：', len(payloads)-good_len)
                    break
                payload = payload.strip()
                payloads.append(payload)
                y.append(1)

        y = np.array(y)
        np.save(y_train_dir, y)

        for payload in payloads:
            tempseg = payload.split(' ')
            if(tempseg == []):
                print(payload)
            payloads_seged.append(tempseg)
            lens.append(len(tempseg))

        with open(lens_dir, 'wb') as f:
            pickle.dump(lens, f)

        # Word2vec模型构建

        model = Word2Vec(
            payloads_seged,
            size=embedding_size,
            iter=iter_num, sg=1,
            min_count=min_num,
            max_vocab_size=max_voc
        )

        model.save(vec_dir)

        x = []
        tempvx = []
        for payload in payloads_seged:
            for word in payload:
                try:
                    tempvx.append(model.wv.get_vector(word))
                except KeyError as e:
                    tempvx.append(np.zeros((embedding_size)))  # 若不在字典中，则输入0占位
            tempvx = np.array(tempvx)
            if(tempvx.shape[0] == 0):
                print(payload)
            x.append(tempvx)
            # print(tempvx.shape)
            tempvx = []

        # 字符串向量长度填充
        lenth = time_step
        for i in range(y.shape[0]):
            if (x[i].shape[0] < lenth):
                try:
                    x[i] = np.pad(x[i], ((0, lenth - x[i].shape[0]),
                                         (0, 0)), 'constant', constant_values=0)
                except ValueError as e:
                    print(i)
                    print(x[i].shape)
                    print(x[i])
                    exit()
            elif (x[i].shape[0] > lenth):
                x[i] = x[i][0:lenth]
        x = np.array(list(x))
        # print(x.shape)
        np.save(x_train_dir, x)

    else:
        model = Word2Vec.load(vec_dir)

    # 验证w2v模型质量

    # print(model.wv.vocab)
    # print(most_similar(model, 'EXT_STMT'))
    print(most_similar(model, 'JMPZ', 5))
    print(model.wv['EXT_STMT'].shape)
