# project
本科毕业论文

## 模型结构
ResNet34 encoder + U-Net decoder + scSE attention + ASPP bottleneck。

## 数据集
有训练集和验证集  
将训练集的一部分作为测试集

## 说明
所有训练输出都保存在runs目录下

## 需要修改的地方


## 运行步骤
%cd /kaggle/working  
!rm -rf /kaggle/working/project  
!git clone https://github.com/LIKE9426334946/project.git  
%cd /kaggle/working/project  
!python3 utils/split.py  
!python3 train.py  
