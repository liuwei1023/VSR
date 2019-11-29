# 视频超分
天池竞赛/全国人工智能大赛 视频超分项目

# 修改记录

* 2019/11/29　　　　`v1.9`

    1. 增加stage2代码。

* 2019/11/22　　　　`v1.8`

    1. 增加CBAM_PP的相关代码和配置文件
    2. 增加CBAM_384的配置文件
    3. 增加CBAM_512的配置文件
    4. 增加CBAM_Denoise_Nonlocal的相关代码和配置文件
    5. 增加CBAM_Denoise的相关代码和配置文件
    6. 增加相关test_distribution代码

* 2019/11/22　　　　`v1.7`

    1. 修改config文件，添加配置文件中non local的设置。

* 2019/11/22　　　　`v1.6`

    1. 增加Non local模块。
    2. 增加Non local的配置文件。

* 2019/11/12　　　　`v1.5`

    1. 修改train_VSR.py中adjust_learning_rate增加learning_mode的bug

* 2019/11/12　　　　`v1.4`

    1. config.py中增加learning_mode。
    2. train_VSR.py中adjust_learning_rate增加learning_mode。

* 2019/11/12　　　　`v1.3`

    1. 修改网络EDVR_Denoise。

* 2019/11/12　　　　`v1.2`

    1. 增加去噪模块Predenoise。
    2. 增加去噪网络EDVR_Denoise。
    3. 修改train_vsr中dataLoader的depth。
    4. 训练和测试函数中增加网络EDVR_Denoise。

* 2019/11/12　　　　`v1.1`

    1. 增加test_distribution。现在可以分布测试。
    2. 修改dataLoader。将过滤bmp文件。

* 2019/11/11　　　　`v1.0`

    1. 修改tran_fun中vsr部分，添加model_path。可以加载预训练模型继续训练。
    2. 修改test_batch，现在只能用于kesci数据集的训练。
    3. 添加EDVR_BMP和EDVR_PP的config文件。



# 训练
训练的代码包括VSR 和 SISR，所有的训练log都会写在log.log文件中。

训练命令：
```
python train.py VSR cfg/config.json
```
