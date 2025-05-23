import matplotlib.pyplot as plt

years = list(range(2015, 2025))
solar_capacity = [2.52, 5.04, 7.56, 10.08, 12.6, 18.6, 29.2, 41.53, 69.55, 85.0]

plt.figure(figsize=(9, 4))

# Plot estimated start (2015–2018)
plt.plot(years, solar_capacity, marker='o', linestyle='-', color='blue', label='Estimated')

# Plot actual data (2019–2023)
plt.plot(years[4:9], solar_capacity[4:9], marker='s', linestyle='-', color='orange', label='Actual')

# Plot estimated end (2024), connecting to previous
#plt.plot(years[8:10], solar_capacity[8:10], marker='s', linestyle='-', color='gray')

# Annotate all points
for year, capacity in zip(years, solar_capacity):
    plt.text(year, capacity + 1.5, f'{capacity:.2f}', ha='center', fontsize=9)

plt.xlabel('Year', fontsize=16)
plt.ylabel('Installed Solar \nPower Capacity (MW)', fontsize=16)
plt.grid(True)
plt.xticks(years, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("solar_power.pdf", format="pdf", bbox_inches="tight")
plt.show()
