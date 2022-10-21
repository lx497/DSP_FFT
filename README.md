# DSP_FFT
python pyqt5设计傅里叶变换fft 功率谱 AR模型

![image](https://user-images.githubusercontent.com/81954499/197120713-4e56a0de-2416-4e56-9d64-39f9f49eecd3.png)

时域信号
![image](https://user-images.githubusercontent.com/81954499/197120954-06e5af95-8091-446a-bcb1-c765ac1e1b12.png)

FFT变换
![image](https://user-images.githubusercontent.com/81954499/197120971-c1c4e01a-8bdc-4eb6-9f4d-2aeaa398bbab.png)

功率谱，先求自相关，再FFT
![image](https://user-images.githubusercontent.com/81954499/197120993-cc521ba9-9abd-4984-8f5b-1b4495aad8b3.png)

AR模型估计功率谱
![image](https://user-images.githubusercontent.com/81954499/197121022-ed0383ff-c443-445b-a854-9c214eea41a3.png)


AR模型预测功率谱，用的数据为test.txt，若要修改为设置信号
f = open('test.txt')
        txt = f.read()
        a = txt.split('\t')
        b = []
        for i in a:
            b.append(float(i))
        u = b
   注释掉以上内容。272行# u = self.end_signal取消注释即可
