# DSP_FFT
python pyqt5设计傅里叶变换fft 功率谱 AR模型

![image](https://user-images.githubusercontent.com/81954499/197120713-4e56a0de-2416-4e56-9d64-39f9f49eecd3.png)


AR模型预测功率谱，用的数据为test.txt，若要修改为设置信号
f = open('test.txt')
        txt = f.read()
        a = txt.split('\t')
        b = []
        for i in a:
            b.append(float(i))
        u = b
   注释掉以上内容。272行# u = self.end_signal取消注释即可
