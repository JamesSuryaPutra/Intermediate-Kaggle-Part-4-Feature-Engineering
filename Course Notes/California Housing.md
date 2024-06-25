As spatial features, California Housing's 'Latitude' and 'Longitude' make natural candidates for k-means clustering. In this example we'll cluster these with 'MedInc' (median income) to
create economic segments in different regions of California.

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.cluster import KMeans

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

    df = pd.read_csv("../input/fe-course-data/housing.csv")
    X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]
    X.head()

    /opt/conda/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning:
    A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.5
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
    /tmp/ipykernel_20/577165241.py:6: MatplotlibDeprecationWarning:
    The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<
    style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")

        MedInc	Latitude Longitude
    0	8.3252	37.88	 -122.23
    1	8.3014	37.86	 -122.22
    2	7.2574	37.85	 -122.24
    3	5.6431	37.85	 -122.25
    4	3.8462	37.85	 -122.25


Since k-means clustering is sensitive to scale, it can be a good idea rescale or normalize data with extreme values. Our features are already roughly on the same scale, so we'll leave
them as-is.

    # Create cluster feature
    kmeans = KMeans(n_clusters=6)
    X["Cluster"] = kmeans.fit_predict(X)
    X["Cluster"] = X["Cluster"].astype("category")

    X.head()

    /opt/conda/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning:
    The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(

        MedInc	Latitude	Longitude	Cluster
    0	8.3252	37.88	    -122.23	    5
    1	8.3014	37.86	    -122.22	    5
    2	7.2574	37.85	    -122.24	    5
    3	5.6431	37.85	    -122.25	    5
    4	3.8462	37.85	    -122.25	    2


Now let's look at a couple plots to see how effective this was. First, a scatter plot that shows the geographic distribution of the clusters. It seems like the algorithm has created
separate segments for higher-income areas on the coasts.

    sns.relplot(
        x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
    );

    /opt/conda/lib/python3.10/site-packages/seaborn/axisgrid.py:118: UserWarning: The figure layout has changed to tight
      self._figure.tight_layout(*args, **kwargs)

![__results___6_1](https://github.com/JamesSuryaPutra/Intermediate-Kaggle-Part-4-Feature-Engineering/assets/155945814/d81060b3-9193-4435-8d79-fce45896ec3b)


The target in this dataset is MedHouseVal (median house value). These box-plots show the distribution of the target within each cluster. If the clustering is informative, these
distributions should, for the most part, separate across MedHouseVal, which is indeed what we see.

    X["MedHouseVal"] = df["MedHouseVal"]
    sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6);

    /opt/conda/lib/python3.10/site-packages/seaborn/axisgrid.py:118: UserWarning: The figure layout has changed to tight
      self._figure.tight_layout(*args, **kwargs)

![__results___8_1](https://github.com/JamesSuryaPutra/Intermediate-Kaggle-Part-4-Feature-Engineering/assets/155945814/4aea7fa9-bc44-45f1-8327-78e6fbd107cc)
