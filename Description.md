解题思路：

从图像来看，低分图像存在噪声。而这部分噪声会极大影响超分的结果。所以去噪是必不可少的一个步骤。

对于图像超分，我们使用在视频超分领域表现比较好的EDVR作为baseline。



项目运行环境：
1. pytorch 1.0 以上
2. python 3.6 以上
3. cuda 10.0

运行方法：
python test_distribution.py VSR cfg/EDVR_CBAM_bmp.json {model_path} {output_folder} 1 1

其中model_path 是训练好的模型路径， output_folder是结果输出的文件夹。
最终结果将会保存在output_folder文件夹下的img_output文件夹下。

