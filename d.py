# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv('../input/diabetes.csv')

outcome=data["Outcome"]

list = ["Outcome"]
data_lean = data.drop(list, axis=1)
data_lean.describe()
data_standard= ( (data_lean - data_lean.mean()) / data_lean.std() )
data_concact = pd.concat([outcome, data_standard], axis=1)
data_melt = pd.melt(data_concact,id_vars="Outcome", var_name="features", value_name='value')

plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="Outcome", data=data_melt,split=True, inner="quart")
plt.xticks(rotation=60)

sns.boxplot(x="features", y="value", hue="Outcome", data=data_melt)
plt.xticks(rotation=60)


sns.jointplot(data_melt.loc[:,'concavity_worst'], data_melt.loc[:,'concave points_worst'], kind="regg", color="#ce1414")
# Any results you write to the current directory are saved as output.
sns.swarmplot(x="features", y="value", hue="Outcome", data=data_melt)


plt.figure(figsize=(10,6))
ax = sns.heatmap(data_melt, fmt='d', linewidths=.5)
