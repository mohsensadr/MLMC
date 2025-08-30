import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mlmc_vs_mc.csv")

plt.loglog(df["eps"], df["mlmc_cost"], "o-", label="MLMC")
plt.loglog(df["eps"], df["mc_cost"], "s-", label="MC")
plt.xlabel("ε")
plt.ylabel("Cost")
plt.legend()
plt.grid(True, which="both")
plt.show()

