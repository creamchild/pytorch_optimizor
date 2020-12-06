#!/bin/bash
python3 mnist.py --optim ADAM --optimswitch C1 > ../result/exp1/ADAM-C1.txt
python3 mnist.py --optim ADAM --optimswitch C2 > ../result/exp1/ADAM-C2.txt
python3 mnist.py --optim ADAM --optimswitch C3 > ../result/exp1/ADAM-C3.txt
python3 mnist.py --optim ADAM --optimswitch C1 --amsgrad True > ../result/exp1/AMSG-C1.txt
python3 mnist.py --optim ADAM --optimswitch C2 --amsgrad True > ../result/exp1/AMSG-C2.txt
python3 mnist.py --optim ADAM --optimswitch C3 --amsgrad True > ../result/exp1/AMSG-C3.txt