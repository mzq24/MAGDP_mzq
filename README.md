# MAGDP

# T-IV 补充实验
## 路径
```~/sim/MAGDP/ ```
## 数据：
原始数据集在`~/sim/Waymo_Dataset`。处理后的数据集在 `~/sim/SimAgent_Dataset/pyg_data_Jun23`
数据处理 
```
python -W ignore data_preprocess.py --data_split validation_0-140
python -W ignore data_preprocess.py --data_split validation_140-145
python -W ignore data_preprocess.py --data_split validation_145-150
```
## MAG

### 复现 IV-24 的结果
不同的是数据集划分不一样了。原来是将 validation 随机划分成 80，5，15.
这次的实验先将 validation 的150个 shards，划分成 0-140，140-145，145-150.
然后在 0-140 上训练，140-145 上验证，在 145-150 上测试。

```
python train_MGA.py --loss_type Joint
python train_MGA.py --loss_type Marginal
```
其他的参数可以在 `train.py` 文件的`args`里面改.

### 添加简单的 Graph structure learning (GSL)
需要重新搞一个model的文件，避免在原来的代码上直接改动。
GSL的部分参考 `gsl_encoder.py` 文件中`Edge_Decoder`和`GNN_Encoder`

## 可直接使用的 arGDPmax 模型
``` xy@xy-Legion:~/sim/Jul08_rrc/GDP/models$ ls arGDPmaxout6GDLI-Loss0.49.ckpt ```

## Controllability 画图
``` xy@xy-Legion:~/sim/Jul08_rrc/VIZ$ viz_rollouts_ego_change_TCL.py```

