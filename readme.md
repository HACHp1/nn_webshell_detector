# 基于神经网络的 PHP webshell检测器

## requirements

* php环境，安装vld拓展
* python库：tensorflow（1或2）、numpy、sklearn等

## 文件结构

* bin: 储存二进制文件，如内存中加载好的数据
* data: 数据集，由于原数据集过大，请自行下载并解压数据集（解压后代码会读取data子目录中所有后缀为PHP的文件），推荐的数据集在最下方
* etx: 杂项、临时文件等
* log: 训练记录
* model: 模型存放
* result: 准确率等结果存放
* utils.py: 参数、函数库
* data_proc.py: 数据预处理，将所有PHP文件转化为操作码字符串，并写入黑白两个txt中
* word2vec.py: 向量化预处理并把结果pickle打包
* show_lens.py：查看所有payload的长度分布统计图
* lstm_tf1.py:tf1 lstm模型代码
* lstm_tf2.py:tf2 lstm模型代码
* detect_lstm_tf1.py:tensorflow1检测API，使训练好的LSTM模型支持单个webshell检测
* detect_lstm_tf2.py:tensorflow2检测API，使训练好的LSTM模型支持单个webshell检测
* attention_tf2.py：tf2 attention模型代码
* detect_attention_tf2.py:tensorflow2检测API，使训练好的attention模型支持单个webshell检测
* test.py:草稿文件

## 执行顺序

0. 在utils.py中调整各项参数，在其他文件中适当调整各项参数
1. data_proc.py
2. word2vec.py
3. lstm_tf2.py

## 参考资料

* https://paper.seebug.org/526/
* https://xz.aliyun.com/t/2016

## 数据集来源

### bad

https://github.com/JohnTroony/php-webshells
https://github.com/ysrc/webshell-sample
https://github.com/tennc/webshell
https://github.com/xl7dev/WebShell

### good

https://github.com/WordPress/WordPress
https://github.com/typecho/typecho
https://github.com/phpmyadmin/phpmyadmin
https://github.com/laravel/laravel
http://www.phpcms.cn/
https://github.com/symfony/symfony
https://github.com/yiisoft/yii2-redis
https://github.com/bcit-ci/CodeIgniter/
https://github.com/smarty-php/smarty
http://www.thinkphp.cn
https://github.com/top-think/framework
