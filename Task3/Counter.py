import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

df = pd.read_csv("quotes_dataset.csv")

df["Author"] = df["Author"].str.strip()
df["Tags"] = df["Tags"].fillna("").str.strip()

top_authors = df["Author"].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_authors.values, y=top_authors.index, palette="viridis")
plt.title("Top 10 Most Quoted Authors")
plt.xlabel("Number of Quotes")
plt.ylabel("Author")
plt.show()

all_tags = []
for t in df["Tags"]:
    all_tags.extend(t.split(", "))

tag_counts = Counter(all_tags).most_common(10)
tags, counts = zip(*tag_counts)

plt.figure(figsize=(10,6))
sns.barplot(x=counts, y=tags, palette="magma")
plt.title("Top 10 Most Common Tags")
plt.xlabel("Frequency")
plt.ylabel("Tag")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Author"].value_counts(), bins=20, kde=True, color="teal")
plt.title("Distribution of Quotes per Author")
plt.xlabel("Number of Quotes")
plt.ylabel("Number of Authors")
plt.show()