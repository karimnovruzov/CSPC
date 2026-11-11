import os
from ucimlrepo import fetch_ucirepo



concrete = fetch_ucirepo(id=165)
X = concrete.data.features
y = concrete.data.targets


print(" METADATA ")
print(concrete.metadata)
print("\n VARIABLES ")
print(concrete.variables)





X = X.copy()
X.columns = [
    "cement",
    "slag",
    "ash",
    "coarse_agg",
    "fine_agg",
    "water",
    "superplastic",
    "age",
]




y = y.iloc[:, 0].rename("strength")




os.makedirs("data", exist_ok=True)
df_combined = X.assign(strength=y)
df_combined.to_csv("data/concrete.csv", index=False)
print("\n DATA SAVED ")