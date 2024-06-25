The Automobile dataset consists of 193 cars from the 1985 model year. The goal for this dataset is to predict a car's price (the target) from 23 of the car's features, such as make,
body_style, and horsepower. In this example, we'll rank the features with mutual information and investigate the results by data visualization.

This hidden cell imports some libraries and loads the dataset.

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    plt.style.use("seaborn-whitegrid")

    df = pd.read_csv("../input/fe-course-data/autos.csv")
    df.head()

        symboling	make	    fuel_type	aspiration	num_of_doors	body_style	drive_wheels	engine_location	wheel_base	length	...	engine_size	fuel_system	bore	stroke	compression_ratio	horsepower	peak_rpm	city_mpg	highway_mpg	price
    0	3	        alfa-romero	gas	        std	        2	            convertible	rwd	            front	        88.6	    168.8	...	130	        mpfi	    3.47	2.68	9	                111	        5000	    21	        27	        13495
    1	3	        alfa-romero	gas	        std	        2	            convertible	rwd	            front	        88.6	    168.8	...	130	        mpfi	    3.47	2.68	9	                111	        5000	    21	        27	        16500
    2	1	        alfa-romero	gas	        std	        2	            hatchback	rwd	            front	        94.5	    171.2	...	152	        mpfi	    2.68	3.47	9	                154	        5000	    19	        26	        16500
    3	2	        audi	    gas	        std	        4	            sedan	    fwd	            front	        99.8	    176.6	...	109	        mpfi	    3.19	3.40	10	                102	        5500	    24	        30	        13950
    4	2	        audi	    gas	        std	        4	            sedan	    4wd	            front	        99.4	    176.6	...	136	        mpfi	    3.19	3.40	8	                115	        5500	    18	        22	        17450
    5 rows × 25 columns


The scikit-learn algorithm for MI treats discrete features differently from continuous features. Consequently, you need to tell it which are which. As a rule of thumb, anything that must
have a float dtype is not discrete. Categoricals (object or categorial dtype) can be treated as discrete by giving them a label encoding (you can review label encodings in our Categorical
Variables lesson).

    X = df.copy()
    y = X.pop("price")

    # Label encoding for categoricals
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()

    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X.dtypes == int


scikit-learn has two mutual information metrics in its feature_selection module: one for real-valued targets (mutual_info_regression) and one for categorical targets (mutual_info_class
if). Our target, price, is real-valued. The next cell computes the MI scores for our features and wraps them up in a nice dataframe.

    from sklearn.feature_selection import mutual_info_regression

    def make_mi_scores(X, y, discrete_features):
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    mi_scores = make_mi_scores(X, y, discrete_features)
    mi_scores[::3]  # show a few features with their MI scores

    curb_weight          1.540126
    highway_mpg          0.951700
    length               0.621566
    fuel_system          0.485085
    stroke               0.389321
    num_of_cylinders     0.330988
    compression_ratio    0.133927
    fuel_type            0.048139
    Name: MI Scores, dtype: float64


And now a bar plot to make comparisions easier.

    def plot_mi_scores(scores):
        scores = scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")

    plt.figure(dpi=100, figsize=(8, 5))
    plot_mi_scores(mi_scores)

![__results___7_0](https://github.com/JamesSuryaPutra/Intermediate-Kaggle-Part-4-Feature-Engineering/assets/155945814/a39ded19-445e-4318-870b-faa82f5d3a28)


Data visualization is a great follow-up to a utility ranking. Let's take a closer look at a couple of these. As we might expect, the high-scoring curb_weight feature exhibits a strong
relationship with price, the target.

    sns.relplot(x="curb_weight", y="price", data=df);

![__results___9_0](https://github.com/JamesSuryaPutra/Intermediate-Kaggle-Part-4-Feature-Engineering/assets/155945814/fce1252c-cb4b-4abc-bf2e-bfec63e66cfa)


The fuel_type feature has a fairly low MI score, but as we can see from the figure, it clearly separates two price populations with different trends within the horsepower feature. This
indicates that fuel_type contributes an interaction effect and might not be unimportant after all. Before deciding a feature is unimportant from its MI score, it's good to investigate
any possible interaction effects -- domain knowledge can offer a lot of guidance here.

    sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df);

![__results___11_0](https://github.com/JamesSuryaPutra/Intermediate-Kaggle-Part-4-Feature-Engineering/assets/155945814/758d1ec4-df75-4a8d-b655-cc6eda34a876)


Data visualization is a great addition to your feature-engineering toolbox. Along with utility metrics like mutual information, visualizations like these can help you discover important
relationships in your data.
