import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("RFR.csv")
df2 = pd.read_csv("XGB.csv")
df3 = pd.read_csv("TFT.csv")
df4 = pd.read_csv("ENTSO-E.csv")

values1 = df1['abs_pct_error']
values2 = df2['abs_pct_error']
values3 = df3['abs_pct_error']
values4 = df4['abs_pct_error']

# Create box plot
plt.boxplot([values1, values2, values3, values4], labels=['RFR', 'XGB', 'TFT', 'ENTSO-E'])

# Customize the plot
plt.title("Box Plot of Entire Test Range")
plt.ylabel("Value")
plt.grid(True)

# Show the plot
plt.show()

'''
plt.figure(figsize=(10, 5))
plt.boxplot(daily_mape.values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='gray'),
                capprops=dict(color='gray'),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=5, linestyle='none'))

plt.ylabel("MAPE (%)")
plt.title("Distribution of Daily MAPE")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
'''