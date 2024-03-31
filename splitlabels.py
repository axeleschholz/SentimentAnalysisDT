import pandas as pd

df = pd.read_csv("preprocessed_asos.csv")

labelled_df = df[df['Label'].notna()]
unlabelled_df = df[df["Label"].isna()]
binary_df = df[df['Label'] != 0]
labelled_binary_df = labelled_df[labelled_df['Label'] != 0]

labelled_df.to_csv("asos_with_labels.csv", index=False)
unlabelled_df.to_csv("asos_without_labels.csv", index=False)
binary_df.to_csv("binary_asos.csv", index=False)
labelled_binary_df.to_csv("binary_asos_with_labels.csv", index=False)
