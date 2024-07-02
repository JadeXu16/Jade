# The original pd data are summarized
# The partial derivative graphs of one and two dimensions are drawn

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./result/pd_all.csv', header=0, index_col=0)
data = data.groupby('feature')

group_1 = data.get_group('R_14_pct')
group_2 = data.get_group('R_16_pct')
group_3 = data.get_group('R_21_pct')
group_4 = data.get_group('R_22_pct')
group_5 = data.get_group('R_24_pct')
group_6 = data.get_group('R_9_pct')

group_1 = group_1.groupby('d')['ret'].mean()
group_2 = group_2.groupby('d')['ret'].mean()
group_3 = group_3.groupby('d')['ret'].mean()
group_4 = group_4.groupby('d')['ret'].mean()
group_5 = group_5.groupby('d')['ret'].mean()
group_6 = group_6.groupby('d')['ret'].mean()

# normalization
R_14 = (group_1-group_1.min()) / (group_1.max()-group_1.min()) * 2 - 1
R_16 = (group_2-group_2.min()) / (group_2.max()-group_2.min()) * 2 - 1
R_21 = (group_3-group_3.min()) / (group_3.max()-group_3.min()) * 2 - 1
R_22 = (group_4-group_4.min()) / (group_4.max()-group_4.min()) * 2 - 1
R_24 = (group_5-group_5.min()) / (group_5.max()-group_5.min()) * 2 - 1
R_9 = (group_6-group_6.min()) / (group_6.max()-group_6.min()) * 2 - 1

pd_sum = pd.DataFrame({'R_14': R_14,
                       'R_16': R_16,
                       'R_21': R_21,
                       'R_22': R_22,
                       'R_24': R_24,
                       'R_9': R_9
                       })

pd_sum.to_csv('./result/partial_derivative.csv')

data = pd.read_csv('/result/pd_imp.csv', index_col=0, header=0)

x = data.index
y1 = data['R_9']
y2 = data['R_22']
y3 = data['R_24']
y4 = data['R_21']
y5 = data['R_14']
y6 = data['R_16']

fig, axs = plt.subplots(2, 3)
plt.subplots_adjust(hspace=0.5)

axs[0, 0].bar(x, y1, width=0.5, color='black', alpha=0.7)
axs[0, 0].set_title('R(9,1)')
axs[0, 0].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
axs[0, 0].set_xticklabels(['', 'Low', '', '', '', '', '', '', '', '', 'High'])

axs[0, 1].bar(x, y2, width=0.5, color='black', alpha=0.7)
axs[0, 1].set_title('R(22,1)')
axs[0, 1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
axs[0, 1].set_xticklabels(['', 'Low', '', '', '', '', '', '', '', '', 'High'])

axs[0, 2].bar(x, y3, width=0.5, color='black', alpha=0.7)
axs[0, 2].set_title('R(24,1)')
axs[0, 2].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
axs[0, 2].set_xticklabels(['', 'Low', '', '', '', '', '', '', '', '', 'High'])

axs[1, 0].bar(x, y4, width=0.5, color='black', alpha=0.7)
axs[1, 0].set_title('R(21,1)')
axs[1, 0].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
axs[1, 0].set_xticklabels(['', 'Low', '', '', '', '', '', '', '', '', 'High'])

axs[1, 1].bar(x, y5, width=0.5, color='black', alpha=0.7)
axs[1, 1].set_title('R(14,1)')
axs[1, 1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
axs[1, 1].set_xticklabels(['', 'Low', '', '', '', '', '', '', '', '', 'High'])

axs[1, 2].bar(x, y6, width=0.5, color='black', alpha=0.7)
axs[1, 2].set_title('R(16,1)')
axs[1, 2].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
axs[1, 2].set_xticklabels(['', 'Low', '', '', '', '', '', '', '', '', 'High'])

plt.show()

data = pd.read_csv('./result/pd_2.csv', index_col=0, header=0)
data = data.groupby(['r1', 'r2', 'd1', 'd2'])['ret'].mean().reset_index()
data = data.groupby(['r1', 'r2'])

fig, axs = plt.subplots(2, 3)
i = 0
j = 0
for name, group in data:
    print(name)
    x = group['d1'].unique()
    y = group['d2'].unique()
    z = group.pivot(index='d1', columns='d2', values='ret')
    val_z = group['ret'].unique()
    contour = axs[i, j].contourf(x, y, z, levels=15, cmap='Greys')
    contour_lines = axs[i, j].contour(x, y, z, levels=15, colors='black', alpha=0.8, linewidths=0.5)
    axs[i, j].set_xlabel(name[0])
    axs[i, j].set_ylabel(name[1])
    j += 1
    if j == 3:
        i += 1
        j = 0

for ax in axs.flat:
    fig.colorbar(contour, ax=ax)

plt.tight_layout()

plt.show()
