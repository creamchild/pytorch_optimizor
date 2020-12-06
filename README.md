# pytorch_AGM

## 说明手册

### 运行流程

### 输入参数说明书

## 更新记录
### 2020年12月6日

#### 开始撰写说明手册
>快结束了，加油！

#### 上传了plot的方法
将之前的plot_static分为plot1234（前4个实验的画图）
和plot5（第五个实验的画图）
plot需传入参数选择实验，具体的参数请参考说明手册

#### 加入了C1C2C3D1D2D3支持
>非常重要！！！

现在只要传入参数就可以更改，具体的参数请参考说明手册

#### 删除了mnist.py的部分代码
决定通过shell开启多进程，py文件中只用实现一个进程

#### 调整了文件结构，创建了一些文件夹
|文件夹|说明|
|-----|-----|
|result|用来放训练文件的运行结果|
|utils|用来放一些依赖文件|
|optimizer|用来放优化器文件|
|experiment|用来放实际训练的文件|

#### shell运行结果写入文件的方法
```bash

python3 gwdc_cifar10.py > result/gwdc_cifar10.txt

``` 

### 2020年12月5日
#### 上传了plot_static.py这是一个画静态图的方法
>但是现在里面什么都没有> <

#### 将之前的ipynb文件替换为了py文件
|ipynb|py|
|-----|-----|
|GWDC_cifar10_resnet34_densnet121.ipynb|cifar10.py|
|GWDC_mnist_multi_perceptron.ipynb|mnist.py|
|GWDC_penn_treebank_gru.ipynb|ptb.py|

现在你不再需要ipynb文件啦，试试在命令行中执行吧：
```bash

python3 cifar10.py

```
#### 删除了ipynb文件
>谢谢你ipynb，再见！

#### 学习了argparse, 这是一个传入参数的方法
现在你可以给py文件传入参数啦！试试这个吧：
```bash
python3 ptb.py --optim ADAM
python3 ptb.py --optim GWDC

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
