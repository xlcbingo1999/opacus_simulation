# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

henzuobiaos = np.array([0, 50, 100, 200, 300, 500, 800, 1000])
his_gamma_0_0 = np.array([181, 174, 194, 257, 286, 272, 271, 279])
his_gamma_0_01 = np.array([169,192,195,253,302,277,282,280])
his_gamma_0_02 = np.array([180,176,192,251,301,278,279,276])
his_gamma_0_05 = np.array([181,204,202,263,283,251,267,282])
his_gamma_0_1 = np.array([195,210,211,268,287,253,258,275])
his_gamma_0_2 = np.array([208,241,243,257,251,236,240,249])
his_gamma_0_3 = np.array([240,246,246,232,233,203,204,216])
his_gamma_0_4 = np.array([229,217,203,197,190,172,180,190])
pbg = np.array([166,148,140,134,147,136,143,140])
sage = np.array([231,205,204,190,214,200,201,203])



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

path = '/home/netlab/DL_lab/opacus_simulation/plots/history_change_jobnum'
plt.savefig(path + '.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()
pp = PdfPages(path + '.pdf')
pp.savefig(fig)
pp.close()