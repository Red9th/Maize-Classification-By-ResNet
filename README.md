# 基于 ResNet18 的果蔬分类

>参考[ B 站教程](https://www.bilibili.com/video/BV1vZ4y1h7X9/?spm_id_from=333.788)实现

## Dataset
- 数据集：https://aistudio.baidu.com/aistudio/datasetdetail/119023/0
- 下载完毕之后解压到`dataset_fruit_veg`目录下，并将文件夹命名为`raw`。
- 运行`split_dataset.py`。

## Train
- 将`if __name__ == '__main__'`下的 mode 改成 train。
- 运行`train.py`
  ```shell
  python train.py
  ```
- 如果要修改训练时的参数，参考`train.py`文件中的`get_args_parser`函数修改默认参数，或者是在上面的命令行中带上相关参数。例如：
  ```shell
  python train.py --batch_size=36 --epochs=30
  ```
- 训练完毕之后会将模型文件保存到`output_dir_pretrained`下。所以在测试时，将`get_args_parser`中的 resume 的 default 值修改为跑出来的模型文件，就可以用训练得到的模型进行测试。

## Test
- 将`if __name__ == '__main__'`下的 mode 改成 infer。
- 运行`train.py`
  ```shell
  python train.py
  ```
- 程序会遍历`dataset_fruit_veg/test`下的图片，每张图片都会输出准确率和预测的结果。