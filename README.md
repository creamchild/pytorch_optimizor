# pytorch_AGM

## 更新记录
### 2020年12月5日
#### 上传了plot_static.py这是一个画静态图的方法
>但是现在里面什么都没有> <

#### 将之前的ipynb文件替换为了py文件
|ipynb|py|
|-----|-----|
|GWDC_cifar10_resnet34_densnet121.ipynb|gwdc_cifar10.py|
|GWDC_mnist_multi_perceptron.ipynb|gwdc_mnist.py|
|GWDC_penn_treebank_gru.ipynb|gwdc_ptb.py|

现在你不再需要ipynb文件啦，试试在命令行中执行吧：
```bash

python3 gwdc_cifar10.py

```
#### 删除了ipynb文件
>谢谢你ipynb，再见！

#### 学习了argparse, 这是一个传入参数的方法
现在你可以给py文件传入参数啦！试试这个吧：
```bash
python3 gwdc_ptb.py --optim ADAM
python3 gwdc_ptb.py --optim GWDC

```
#### 关于C1C2C3的构想
参考现有的例子，可否使用shell脚本来循环参数，最终运行完全部的参数

关于可执行脚本的执行方法：
```bash
chmod +x run.sh
./run.sh 
```


### 2020年12月4日
上传了PTB——gru的运行代码
