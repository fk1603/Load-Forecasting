import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("RFR.csv")
df2 = pd.read_csv("XGB.csv")
df3 = pd.read_csv("TFT.csv")
df4 = pd.read_csv("ENTSO-E.csv")

values1 = df1['abs_pct_error']
values2 = df2['abs_pct_error']
values3 = df3['daily_mape']
values4 = df4['abs_pct_error']

# Create box plot
plt.boxplot([values1, values2, values3, values4], labels=['RFR', 'XGB', 'TFT', 'ENTSO-E'])

# Customize the plot
plt.ylabel("MAPE [%]", fontsize=16)
plt.xticks(fontsize=16)
plt.grid(True)

# Show the plot
plt.savefig("boxplot_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()
