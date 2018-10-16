import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

corr = train.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

plt.show()