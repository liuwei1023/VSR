# 解题思路：

对于图像超分，我们使用在视频超分领域表现比较好的EDVR作为baseline。

EDVR很好的利用了视频多帧之间的信息，将视频超分划分为三个主要流程：PCD,TSA和reconstruction。其中PCD主要的作用是帧间对齐，TSA用于融合多帧的特征信息，reconstruction用于图像重建以及超分。
尽管EDVR已经能取得很不错的超分效果，但是仍然有很多改进的空间。具体的，我们对EDVR做了如下改进。

1. CBAM模块
EDVR使用了简单的residual模块进行特征提取。为了更多的利用时空信息，我们加入了注意力时空注意力机制CBAM。CBAM表示卷积模块的注意力机制模块。是一种结合了空间（spatial）和通道（channel）的注意力机制模块。相比于senet只关注通道（channel）的注意力机制可以取得更好的效果。
CBAM将注意力机制分为spatial attention和channel attention。具体如图所示。

2. Non-local模块
受计算机视觉中经典的非局部均值方法启发，非局部操作计算某一位置的响应为所有位置特征的加权和，这个操作可以作为捕获远程依赖的通用模块，具体公式和网络实现如下：

3. Denoise模块
对于初赛任务涉及到噪音问题，我们加入了denoise模块，采用了深盲视频去噪方法，使用一个结合空间和时间的滤波，学习去空间去噪首帧和同时如何结合他们的时间信息，解决目标运动，亮度变化，低照度条件和时间一致，具体网络结构如图所示：



参考文献：《CBAM: Convolutional Block Attention Module》   https://arxiv.org/abs/1807.06521

2. 其他模块仍在实验中，暂时没有好的实验结果。


# 实验细节





# 项目运行环境：
1. pytorch 1.0 以上
2. python 3.6 以上
3. cuda 10.0

# 运行方法：
python test_distribution.py VSR cfg/EDVR_CBAM_bmp.json {model_path} {output_folder} 1 1

其中model_path 是训练好的模型路径， output_folder是结果输出的文件夹。
最终结果将会保存在output_folder文件夹下的img_output文件夹下。
注：需要在代码中修改测试文件的路径

