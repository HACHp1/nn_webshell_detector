import tensorflow as tf
from tensorflow.keras import layers
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from numpy import argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

n_inputs = embedding_size  # 输入维度，等于embedding大小
n_steps = time_step

# --参数区
n_hidden_units = 60  # 隐藏层层数（不一定等于时序长度）
n_classes = 2
input_keep_prob = 0.6  # dropout层
CONTINUE_TRAIN = False
random_state = 0
print_step = 10
TEST_SIZE = 0.25


# 超参数
lr = 0.0001
training_iters = 1000  # 迭代次数（不是epoch数） epoch=training_iters/data_len
BATCH_SIZE = 128  # 训练时批的大小


tf.random.set_seed(0)


def lstm_model():  # 构建lstm模型
    inputs = tf.keras.Input(shape=(n_steps, n_inputs,), batch_size=BATCH_SIZE)

    h1 = layers.Dense(n_hidden_units, input_shape=(
        time_step, embedding_size))(inputs)

    lstm_out, hidden_state, cell_state = layers.LSTM(
        n_hidden_units, return_state=True)(h1)  # 只取最后一个状态作为输出

    h3 = layers.Dense(n_classes, input_shape=(
        n_inputs,), activation='sigmoid')(lstm_out)

    outputs = layers.Activation('softmax')(h3)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':

    if CONTINUE_TRAIN:
        model = tf.keras.models.load_model('model/lstm.keras')
    else:
        model = lstm_model()

    model.summary()
    values = np.array([0, 1])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)

    vx_train = np.load(x_train_dir)
    vy_train = np.load(y_train_dir)

    vy_train = onehot_encoder.transform(
        vy_train.reshape(-1, 1))  # 对y进行onehot编码

    x_train, x_test, y_train, y_test = train_test_split(
        vx_train, vy_train, test_size=TEST_SIZE, random_state=random_state)

    model.fit(x_train, y_train, batch_size=BATCH_SIZE,
              epochs=training_iters//x_train.shape[0], validation_split=0.1)  # 训练

    model.save('model/lstm.keras')

    y_pred = model.predict(x_test)

    y_pred_real = []
    for i in range(y_pred.shape[0]):
        y_pred_real.append(
            label_encoder.inverse_transform([argmax(y_pred[i])]))

    y_test_real = []
    for i in range(y_test.shape[0]):
        y_test_real.append(
            label_encoder.inverse_transform([argmax(y_test[i])]))  # 将y_test进行onehot还原

    print('test acc:', accuracy_score(y_test_real, y_pred_real))
    print('test recall:', recall_score(
        y_test_real, y_pred_real, average='macro'))
