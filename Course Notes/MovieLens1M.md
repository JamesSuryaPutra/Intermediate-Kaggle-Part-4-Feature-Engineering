The MovieLens1M dataset contains one-million movie ratings by users of the MovieLens website, with features describing each user and movie.

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import warnings

    plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True)
    plt.rc(
        "axes",
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=14,
        titlepad=10,
    )
    warnings.filterwarnings('ignore')

    df = pd.read_csv("../input/fe-course-data/movielens1m.csv")
    df = df.astype(np.uint8, errors='ignore') # reduce memory footprint
    print("Number of Unique Zipcodes: {}".format(df["Zipcode"].nunique()))

    Number of Unique Zipcodes: 3439


With over 3000 categories, the Zipcode feature makes a good candidate for target encoding, and the size of this dataset (over one-million rows) means we can spare some data to create the
encoding.

We'll start by creating a 25% split to train the target encoder.

    X = df.copy()
    y = X.pop('Rating')

    X_encode = X.sample(frac=0.25)
    y_encode = y[X_encode.index]
    X_pretrain = X.drop(X_encode.index)
    y_train = y[X_pretrain.index]


The category_encoders package in scikit-learn-contrib implements an m-estimate encoder, which we'll use to encode our Zipcode feature.

    from category_encoders import MEstimateEncoder

    # Create the encoder instance. Choose m to control noise.
    encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)

    # Fit the encoder on the encoding split.
    encoder.fit(X_encode, y_encode)

    # Encode the Zipcode column to create the final training data
    X_train = encoder.transform(X_pretrain)


Let's compare the encoded values to the target to see how informative our encoding might be.

    plt.figure(dpi=90)
    ax = sns.distplot(y, kde=False, norm_hist=True)
    ax = sns.kdeplot(X_train.Zipcode, color='r', ax=ax)
    ax.set_xlabel("Rating")
    ax.legend(labels=['Zipcode', 'Rating']);

![__results___11_0](https://github.com/JamesSuryaPutra/Intermediate-Kaggle-Part-4-Feature-Engineering/assets/155945814/a46d182c-2b82-40f8-adcf-50ffee960691)


The distribution of the encoded Zipcode feature roughly follows the distribution of the actual ratings, meaning that movie-watchers differed enough in their ratings from zipcode to
zipcode that our target encoding was able to capture useful information.
