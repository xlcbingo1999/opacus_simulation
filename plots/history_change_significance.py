# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

henzuobiaos = np.array([0, 50, 100, 200, 300, 500, 800, 1000])
his_gamma_0_0 = np.array([106.3,89.4,95.8,196.9,183.2,180.3,208,213.2])
his_gamma_0_01 = np.array([105.4,104,95.9,188.5,201.1,178.8,219.5,211.7])
his_gamma_0_02 = np.array([104.5,90.8,93.2,193.7,198.2,178.8,217.5,213.8])
his_gamma_0_05 = np.array([105.3,113.6,103.4,200.3,185.7,161.6,207.5,212.8])
his_gamma_0_1 = np.array([123.4,117,105.9,204.8,184.4,166.7,200.5,210.5])
his_gamma_0_2 = np.array([134,148.5,126.7,193.4,162.1,156.4,184.5,187.6])
his_gamma_0_3 = np.array([162.2,147,139.8,175.6,147.6,130,156,164.9])
his_gamma_0_4 = np.array([148.3,128,116.3,154.2,123.3,105.9,137,145.7])
pbg = np.array([80,65.5,55.1,87,68,63.4,96,91.2])
sage = np.array([114.3,89.3,82.5,118.2,102,96.9,134,131.7])



# label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1
fig = plt.figure(figsize=(10, 5))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框

plt.plot(henzuobiaos, his_gamma_0_0, marker='o', color="r", label=r"HIS:$\gamma$-0.0", linewidth=1.5)
plt.plot(henzuobiaos, his_gamma_0_05, marker='o', color="g", label=r"HIS:$\gamma$-0.05", linewidth=1.5)
plt.plot(henzuobiaos, his_gamma_0_1, marker='o', color="b", label=r"HIS:$\gamma$-0.1", linewidth=1.5)
plt.plot(henzuobiaos, his_gamma_0_2, marker='o', color="k", label=r"HIS:$\gamma$-0.2", linewidth=1.5)
plt.plot(henzuobiaos, his_gamma_0_4, marker='o', color="y", label=r"HIS:$\gamma$-0.4", linewidth=1.5)

plt.plot(henzuobiaos, pbg, marker='o', color="c", label=r"PBG", linewidth=1.5)
plt.plot(henzuobiaos, sage, marker='o', color="xkcd:violet", label=r"SAGE", linewidth=1.5)


group_labels = list(str(hen) for hen in henzuobiaos)  # x轴刻度的标识
plt.xticks(henzuobiaos, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
plt.yticks(fontsize=12, fontweight='bold')
# plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
plt.xlabel("Number of history jobs", fontsize=13, fontweight='bold')
plt.ylabel("Significance of allocated jobs", fontsize=13, fontweight='bold')
# plt.xlim(0.9, 6.1)  # 设置x轴的范围
# plt.ylim(1.5, 16)

# plt.legend()          #显示各曲线的图例
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
plt.legend(loc=2, bbox_to_anchor=(0.98,1.0),borderaxespad = 0.)
plt.subplots_adjust(left=0.1, right=0.88)

path = '/home/netlab/DL_lab/opacus_simulation/plots/history_change_significance'
plt.savefig(path + '.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()
pp = PdfPages(path + '.pdf')
pp.savefig(fig)
pp.close()