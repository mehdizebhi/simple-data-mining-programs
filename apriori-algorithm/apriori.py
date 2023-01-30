import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def main():
    # read the dataset
    df = pd.read_csv("store_data.csv", header=None)

    # preprocess the data
    df_binary = pd.get_dummies(df)

    # run the apriori algorithm
    frequent_itemsets = apriori(df_binary, min_support=0.0045, use_colnames=True)

    # extract association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

    # filter rules by lift
    rules = rules[(rules['lift'] >= 3)]

    print("******************************************************")
    print("All Rules with lift >= 3 and min confidence = 0.2 : \n")
    print(rules)

if __name__ == "__main__":
    main()