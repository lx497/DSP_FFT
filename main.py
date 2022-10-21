"""
数字信号处理 余成波老师
design by lx497
2022.10.19
address:CQUT
email:lx59497@163.com
"""
import math
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
import sys
import numpy as np
from MainWin import Ui_MainWindow
import matplotlib
from scipy.fftpack import fft
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

matplotlib.use("Qt5Agg")  # 声明使用QT5


class MyMatplotlibFigure(FigureCanvasQTAgg):
    """
    创建一个画布类，并把画布放到FigureCanvasQTAgg
    """

    def __init__(self, width=10, height=10, dpi=100):
        # plt.rcParams['figure.facecolor'] = 'r'  # 设置窗体颜色
        # plt.rcParams['axes.facecolor'] = 'b'  # 设置绘图区颜色
        # 创建一个Figure,该Figure为matplotlib下的Figure，不是matplotlib.pyplot下面的Figure
        self.figs = Figure(figsize=(width, height), dpi=dpi)
        super(MyMatplotlibFigure, self).__init__(self.figs)  # 在父类种激活self.fig，
        self.axes = self.figs.add_subplot(111)  # 添加绘图区

    def mat_plot_draw_axes(self, t, s):
        """
        用清除画布刷新的方法绘图
        :return: 0
        """
        self.axes.cla()  # 清除绘图区
        self.axes.spines['top'].set_visible(False)  # 顶边界不可见
        self.axes.spines['right'].set_visible(False)  # 右边界不可见
        # 设置左、下边界在（0，0）处相交
        # self.axes.spines['bottom'].set_position(('data', 0)) # 设置y轴线原点数据为 0
        self.axes.spines['left'].set_position(('data', 0))  # 设置x轴线原点数据为 0
        self.axes.plot(t, s)
        self.figs.canvas.draw()  # 这里注意是画布重绘，self.figs.canvas
        self.figs.canvas.flush_events()  # 画布刷新self.figs.canvas


class MainDialogImgBW(QtWidgets.QMainWindow, QDialog, Ui_MainWindow):
    def __init__(self):
        super(MainDialogImgBW, self).__init__()

        self.setupUi(self)
        self.setWindowTitle("数字信号处理")
        self.setMinimumSize(0, 0)

        # 定义MyFigure类的一个实例
        self.F = MyMatplotlibFigure(width=3, height=2, dpi=100)
        # 在GUI的groupBox中创建一个布局，用于添加MyFigure类的实例（即图形）后其他部件。
        self.gridlayout = QGridLayout(self.groupBox)  # 继承容器groupBox
        self.gridlayout.addWidget(self.F, 0, 1)

        # 参数
        self.y_fft = []
        self.Fs = 1000  # 采样频率
        self.T = 1 / self.Fs
        self.N = 1000  # 采样点数
        # self.n = np.arange(0.0, self.runtime, 1.0 / self.Fs)  # 计算出n的值
        self.nn = np.linspace(0.0, self.N * self.T, self.N)  # (0.0   ，采样点*周期数   ，   采样点)

        # sin参数
        self.sin_flag = 0  # 选择sin信号
        self.cos_flag = 0  # 选择sin信号
        self.sin_cos_xn = []  # 存sin信号的点
        self.sin_cos_F = 1  # sin 的幅值
        self.sin_cos_A = 1  # sin的频率
        self.noise = []
        self.noise_ave = 0
        self.noise_var = 1

        # 最终的信号
        self.end_signal = np.zeros(len(self.nn), dtype=float)

        # 标志位
        self.show_hn_flag = 0

        # 初始化函数
        self.button_connect()
        self.set_default_parameters()

    def set_sin_cos_AF(self):
        """
        获取sin cos信号的幅值和
        :return:
        """
        self.sin_cos_A = float(self.lineEdit_sinA.text())  # 从Edit提取出值
        self.sin_cos_F = float(self.lineEdit_sinF.text())  # 从Edit提取出值
        if self.checkBox_sin.checkState() == self.checkBox_cos.checkState():  # 判断先择的正弦信号还是余弦信号
            self.show_message()
            return
        if self.checkBox_sin.checkState():  # 判断正弦信号
            self.sin_cos_xn = np.sin(2 * np.pi * self.nn * self.sin_cos_F) * self.sin_cos_A
            if self.show_hn_flag:
                self.textBrowser_end_signal.insertPlainText('+'
                                                            + str(self.sin_cos_A) + '*'
                                                            + 'sin' + '(' + '2*pi*' + str(
                    self.sin_cos_F) + ')')  # 进行公式的显示
            else:
                self.textBrowser_end_signal.insertPlainText('h(n)='
                                                            + str(self.sin_cos_A) + '*'
                                                            + 'sin' + '(' + '2*pi**' + str(
                    self.sin_cos_F) + ')')  # 进行公式的显示
                self.show_hn_flag = 1
        else:  # 判断余弦信号
            self.sin_cos_xn = np.cos(2 * np.pi * self.nn * self.sin_cos_F) * self.sin_cos_A
            if self.show_hn_flag:
                self.textBrowser_end_signal.insertPlainText('+'
                                                            + str(self.sin_cos_A) + '*'
                                                            + 'cos' + '(' + '2*pi*' + str(
                    self.sin_cos_F) + ')')  # 进行公式的显示
            else:
                self.textBrowser_end_signal.insertPlainText('h(n)='
                                                            + str(self.sin_cos_A) + '*'
                                                            + 'cos' + '(' + '2*pi*' + str(
                    self.sin_cos_F) + ')')  # 进行公式的显示
                self.show_hn_flag = 1
        self.F.mat_plot_draw_axes(self.nn, self.sin_cos_xn)
        self.end_signal = self.end_signal + self.sin_cos_xn

    def set_noise(self):
        """
        设置白噪声
        :return:
        """
        if self.lineEdit_ns_ave.text():  # 判断lineEdit中是否有值
            self.noise_ave = float(self.lineEdit_ns_ave.text())  # 均值
        else:
            self.show_message()  # 没有值，就显示错误
            return 0
        if self.lineEdit_ns_var.text():  # 与上同理
            self.noise_var = float(self.lineEdit_ns_var.text())  # 方差
        else:
            self.show_message()
            return 0
        self.noise = (self.noise_ave + np.random.randn(len(self.nn))) * self.noise_var  # 参生方差、均值的白噪声
        self.F.mat_plot_draw_axes(self.nn, self.noise)  # 画图
        self.end_signal = self.end_signal + self.noise
        if self.show_hn_flag:
            self.textBrowser_end_signal.insertPlainText('+' +
                                                        str(self.noise_ave) + '+' + 'randn*' + str(self.noise_var))
        else:
            self.textBrowser_end_signal.insertPlainText('h(n)=' +
                                                        str(self.noise_ave) + '+' + 'randn*' + str(self.noise_var))
            self.show_hn_flag = 1

    def Fs_N_setting(self):
        """
        设置参数采样频率以及采样点数
        :return:0
        """
        if self.lineEdit_Fs.text():  # 加上if判断防止为空，运行错误
            self.Fs = int(self.lineEdit_Fs.text())
        if self.lineEdit_N.text():
            self.N = int(self.lineEdit_N.text())
        self.nn = np.linspace(0.0, self.N * self.T, self.N)  # (0.0   ，采样点*周期数   ，   采样点)
        self.end_signal = np.zeros(len(self.nn), dtype=float)
        self.textBrowser_Fs_N_end.clear()
        self.textBrowser_Fs_N_end.setText('Fs=' + str(self.Fs) + '  ' + 'N=' + str(self.N))

    def set_reset(self):
        self.set_default_parameters()
        # 参数
        self.y_fft = []
        self.Fs = 2000  # 采样频率
        self.T = 1 / self.Fs
        self.N = 1000  # 采样点数
        self.nn = np.linspace(0.0, self.N * self.T, self.N)  # (0.0   ，采样点*周期数   ，   采样点)

        # sin参数
        self.sin_flag = 0  # 选择sin信号
        self.cos_flag = 0  # 选择sin信号
        self.sin_cos_xn = []  # 存sin信号的点
        self.sin_cos_F = 1  # sin 的幅值
        self.sin_cos_A = 1  # sin的频率
        self.noise = []
        self.noise_ave = 0
        self.noise_var = 1

        # 最终的信号
        self.end_signal = np.zeros(len(self.nn), dtype=float)

        # 标志位
        self.show_hn_flag = 0
        self.textBrowser_Fs_N_end.clear()
        self.textBrowser_Fs_N_end.setText('Fs=' + str(self.Fs) + '  ' + 'N=' + str(self.N))
        self.textBrowser_end_signal.clear()

    def set_default_parameters(self):
        """
        设置初始化参数
        :return:
        """
        self.lineEdit_Fs.setText(str(self.Fs))
        self.lineEdit_N.setText(str(self.N))
        self.lineEdit_sinA.setText(str(self.sin_cos_A))
        self.lineEdit_sinF.setText(str(self.sin_cos_F))
        self.lineEdit_ns_var.setText(str(self.noise_var))
        self.lineEdit_ns_ave.setText(str(self.noise_ave))
        self.textBrowser_Fs_N_end.clear()
        self.textBrowser_Fs_N_end.setText('Fs=' + str(self.Fs) + '  ' + 'N=' + str(self.N))

    def button_connect(self):
        """
        连接各个按钮函数
        :return:
        """
        self.pushButton_canshu_set.clicked.connect(self.Fs_N_setting)
        self.pushButton_sinadd.clicked.connect(self.set_sin_cos_AF)
        self.pushButton_set_ns.clicked.connect(self.set_noise)
        self.pushButton_hn.clicked.connect(self.show_signal_end)
        self.pushButton_fft.clicked.connect(self.show_fft)
        self.pushButton_pfft.clicked.connect(self.power_spectrum)
        self.pushButton_ar.clicked.connect(self.AR_model)
        self.pushButton_default.clicked.connect(self.set_reset)

    def show_signal_end(self):
        """
        显示最终的信号
        :return:
        """
        if len(self.end_signal):
            self.F.mat_plot_draw_axes(self.nn, self.end_signal)

        else:
            self.show_message()

    def show_fft(self):
        """
        对信号进行FFT变换
        :return:
        """
        yf = fft(self.end_signal)

        xf = np.linspace(0.0, 1.0 / (2.0 * self.T), self.N // 2)  # 目的是获取频率  print("N//2",N//2) 保证是整数，频谱图的频率一般是不会高于采样率的。
        self.y_fft = np.abs(yf[0:self.N // 2]) * 2.0 / self.N
        self.F.mat_plot_draw_axes(xf, self.y_fft)

    def power_spectrum(self):
        """
        求功率谱
        :return:
        """
        cor_x = np.correlate(self.end_signal, self.end_signal, 'same')
        cor_X = fft(cor_x)
        ps_cor = np.abs(cor_X)
        ps_cor = ps_cor / np.max(ps_cor)
        self.ps = np.abs(ps_cor[0:self.N // 2])
        xf = np.linspace(0.0, 1.0 / (2.0 * self.T), self.N // 2)
        self.F.mat_plot_draw_axes(xf, self.ps)

    def AR_model(self):
        f = open('test.txt')
        txt = f.read()
        a = txt.split('\t')
        b = []
        for i in a:
            b.append(float(i))
        u = b

        # u = self.end_signal
        if len(u) == 0:  # 没有数据显示错误
            self.show_message()
            return 0
        a, P = param_init(u)

        H = cal_PSD(4, a, P)
        l = 512
        self.F.mat_plot_draw_axes(np.arange(0, 0.5, 0.5 / l), H)
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel('PSD (dB/Hz)')
        # plt.title('The curve of power spectrum(p=3,4,5)')
        # # plt.plot(abs(cor_X))
        # plt.show()

    def show_message(self):
        QMessageBox.information(self, "警告", "输入有误！！！",
                                QMessageBox.Yes)  # 最后的Yes表示弹框的按钮显示为Yes，默认按钮显示为OK,不填QMessageBox.Yes即为默认


def param_init(u):
    """
    求参数
    :param u: 输入信号
    :return: a p给cal_PSD
    """
    N = len(u)
    k = 5  # 阶数
    # 数据初始化
    f = u[:]  # 用于更新的误差变量
    b = u[:]
    a = np.array(np.zeros((k + 1, k + 1)))  # 模型参数初始化
    for i in range(k + 1):
        a[i][0] = 1
    # 计算P0 1/N*sum(u*2)
    P0 = 0
    for i in range(N):
        P0 += u[i] ** 2
    P0 /= N
    P = [P0]
    # Burg 算法更新模型参数
    for p in range(1, k + 1):
        Ka = 0  # 反射系数的分子
        Kb = 0  # 反射系数的分母F
        for n in range(p, N):
            Ka += f[n] * b[n - 1]
            Kb = Kb + f[n] ** 2 + b[n - 1] ** 2
        K = 2 * Ka / Kb
        # 更新前向误差和反向误差
        fO = f[:]
        bO = b[:]
        for n in range(p, N):
            b[n] = -K * fO[n] + bO[n - 1]
            f[n] = fO[n] - K * bO[n - 1]
        # 更新此时的模型参数
        for i in range(1, p + 1):
            if (i == p):
                a[p][i] = -K
            else:
                a[p][i] = a[p - 1][i] - K * a[p - 1][p - i]
        P.append((1 - K ** 2) * P[p - 1])

    return a, P


# 计算第k阶的功率谱
def cal_PSD(k, a, P, l=512):
    """
    计算功率谱
    :param k:阶数
    :param a:
    :param P:
    :param l:
    :return:
    """
    H = np.array(np.zeros(l), dtype=complex)
    for f in range(l):
        f1 = f * 0.5 / l  # 频率值
        for i in range(1, k + 1):
            H[f] += complex(a[k][i] * np.cos(2 * np.pi * f1 * i), -a[k][i] * np.sin(2 * np.pi * f1 * i))
        H[f] += 1
        H[f] = 1 / H[f]  # 系统函数的表达式
        H[f] = 10 * math.log10(np.abs(H[f]) ** 2 * P[k])

    return H


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainDialogImgBW()
    main.show()
    sys.exit(app.exec_())
