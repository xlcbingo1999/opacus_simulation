# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

henzuobiaos = np.array([500, 1000, 1500, 2000])
his_gamma_0_0 = np.array([272,379,394,410])
his_gamma_0_01 = np.array([277,387,390,390])
his_gamma_0_02 = np.array([278,383,382,418])
his_gamma_0_05 = np.array([251,390,390,414])
his_gamma_0_1 = np.array([253,376,438,430])
his_gamma_0_2 = np.array([236,336,445,495])
his_gamma_0_3 = np.array([203,286,394,488])
his_gamma_0_4 = np.array([172,266,345,409])
pbg = np.array([136,165,147,140])
sage = np.array([200,228,217,199])


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
plt.xlabel(r"Number of test jobs $n$", fontsize=13, fontweight='bold')
plt.ylabel("Number of allocated jobs", fontsize=13, fontweight='bold')
# plt.xlim(0.9, 6.1)  # 设置x轴的范围
# plt.ylim(1.5, 16)

# plt.legend()          #显示各曲线的图例
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
plt.legend(loc=2, bbox_to_anchor=(0.98,1.0),borderaxespad = 0.)
plt.subplots_adjust(left=0.1, right=0.88)

path = '/home/netlab/DL_lab/opacus_simulation/plots/test_change_jobnum'
plt.savefig(path + '.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()
pp = PdfPages(path + '.pdf')
pp.savefig(fig)
pp.close()