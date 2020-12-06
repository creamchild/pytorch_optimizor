# pytorch_AGM

## 说明手册（更新中）

### 运行流程

#### 安装requirements.txt依赖
```bash
pip install -r requirements.txt
```
#### 编辑run.sh文件
```bash
cd pytorch_AGM
cd experiment
vim run.sh
```
run.sh文件中记录了本次运行的文件列表
修改例:
```bash
python3 mnist.py --optim ADAM --optimswitch C2 --amsgrad True > ../result/exp1/AMSG-C2.txt
python3 mnist.py --optim ADAM --optimswitch C3 --amsgrad True > ../result/exp1/AMSG-C3.txt
```
第一行代表了在python3环境下运行mnist.py文件，输入参数为--optim ADAM --optimswitch C2 --amsgrad True表示运行的是AMSG-C2
'>'后面的内容表示结果存放的地址 在这个例子中，结果存放在result的exp1文件夹下，命名为AMSG-C2.txt
以此类推

在运行py文件之间，你可以插入一些echo命令用来打印信息，这样就可以实时看到运行的进度啦
```bash
echo "ADAM-C1 finished"
```
>具体的参数请参考输入参数说明书

>如需开启多进程，则在两行命令之间加上'&'字符

>警告：不推荐进行多进程处理，这有可能会使你的gpu内存溢出！

#### 运行run.sh文件

run.sh文件编辑完成之后运行，开始实验
```bash
chmod +x run.sh
./run.sh 
```

在运行完所有实验结果之后就可以进行画图啦

#### 关于试运行
在运行整个项目之前可以进行一些试运行，这样可以确保运行的顺利，也可以提前加载数据集，使得打印结果时更稳定
```bash
cd pytorch_AGM
cd experiment
python3 mnist.py
python3 cifar10.py
python3 ptb.py
```

#### plot画图
首先确认result/exp1等的文件夹中是否已经完成所有的进程，并以txt文本格式保存

确认完成之后返回主目录运行plot1234.py文件
```bash
cd ../
python3 plot1234.py --exp 1
```
注意这里的plot1234.py存在输入参数exp，代表需要画图的实验文件夹名称，但是plot5.py不需要

运行完成之后你就可以在对应的文件夹中看到结果图片啦。
例如： result/exp1/

#### 完结撒花 Yeah！

### 输入参数说明书
|输入参数|说明|默认值|可选范围|使用场景|
|-----|-----|-----|-----|-----|
|--epoch|可以变更epoch|5|大于0的任意整数|通用|
|--optim|可以变更优化器|ADAM|ADAM和GWDC|通用|
|--amsgrad|可以选择是否使用amsgrad方法|False|True和False|通用|
|--optimswitch|可以变更学习率更新的C1C2C3|C1|C1C2C3D1D2D3|通用|
|--nntype|可以变更神经网络类型|1|1和2|仅在mnist.py中使用|
|--nntype|可以变更神经网络类型|resnet|resnet和densenet|仅在cifar10.py中使用|
|--exp|可以变更需要绘图的实验文件夹|1|1 2 3 4|仅在plot1234.py中使用|

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
