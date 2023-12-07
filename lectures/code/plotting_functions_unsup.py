import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap
from scipy.spatial import distance
from sklearn.metrics import euclidean_distances
# from sklearn.neighbors import DistanceMetric
from sklearn.manifold import MDS
from scipy.spatial import distance
from sklearn.datasets import make_blobs
from matplotlib.patches import Circle
from sklearn.linear_model import Lasso
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import scipy
from scipy.cluster.hierarchy import (
    average,
    complete,
    dendrogram,
    fcluster,
    single,
    ward,
)
from scipy.spatial.distance import cdist

import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Adapted from the mglearn package
# https://github.com/amueller/mglearn/blob/106cf48ef03710ef1402813997746741aa6467da/mglearn/plot_helpers.py#L27

data = np.random.rand(4, 4)
fig, (ax1, ax2) = plt.subplots(2)
ax1.imshow(data)
ax1.set_title("Default colormap")
plt.rc('image', cmap='viridis')
ax2.imshow(data)
ax2.set_title("Set default colormap")
colors = ['xkcd:azure', 'yellowgreen', 'tomato', 'teal', 'indigo', 'aqua', 'orangered', 'orchid', 'black', 'xkcd:turquoise', 'xkcd:violet', 'aquamarine', 'chocolate', 'darkgreen', 'sienna', 'pink', 'lightblue', 'yellow', 'lavender', 'wheat', 'linen']

####################################
# Common visualization functions
####################################


def discrete_scatter(x1, x2, y=None, markers=None, s=8, ax=None,
                     labels=None, padding=.2, alpha=1, c=None, markeredgewidth=0.6, 
                     label_points=False, x1_annot=-0.1, x2_annot=0.1):
    """Adaption of matplotlib.pyplot.scatter to plot classes or clusters.
    Parameters
    ----------
    x1 : nd-array
        input data, first axis
    x2 : nd-array
        input data, second axis
    y : nd-array
        input data, discrete labels
    cmap : colormap
        Colormap to use.
    markers : list of string
        List of markers to use, or None (which defaults to 'o').
    s : int or float
        Size of the marker
    padding : float
        Fraction of the dataset range to use for padding the axes.
    alpha : float
        Alpha value for all points.
    """
    if ax is None:
        ax = plt.gca()

    if y is None:
        y = np.zeros(len(x1))        

    # unique_y = np.unique(y)
    unique_y, inds = np.unique(y, return_index=True)    

    if markers is None:
        markers = ['o', '^', 'v', 'D', 's', '*', 'p', 'h', 'H', '8', '<', '>'] * 10

    if len(markers) == 1:
        markers = markers * len(unique_y)

    if labels is None:
        labels = unique_y

    # lines in the matplotlib sense, not actual lines
    lines = []


    if len(unique_y) == 1: 
        cr = [-1]
    else: 
        cr = sorted([y[index] for index in sorted(inds)])

    if c is not None and len(c) == 1: 
        cr = c
    
    for (i, (yy, color_ind)) in enumerate(zip(unique_y, cr)):
        mask = y == yy
        # print(f'color_ind= {color_ind} and i = {i}')
        # if c is none, use color cycle
        color = colors[color_ind]
        # print('color: ', color)
        # use light edge for dark markers
        if np.mean(colorConverter.to_rgb(color)) < .2:
            markeredgecolor = "grey"
        else:
            markeredgecolor = "black"

        lines.append(ax.plot(x1[mask], x2[mask], markers[i], markersize=s,
                             label=labels[i], alpha=alpha, c=color,                             
                             markeredgewidth=markeredgewidth,
                             markeredgecolor=markeredgecolor)[0])
    if label_points: 
        labs = [str(label) for label in list(range(0,len(x1)))]
        for i, txt in enumerate(labs):
            font_size=10
            ax.annotate(txt, (x1[i], x2[i]), xytext= (x1[i]+x1_annot, x2[i]+x2_annot), c='k', size = font_size)

    return lines    


def corr_heatmat(cor, w=6, h=4): 
    plt.figure(figsize=(w, h))
    sns.set(font_scale=1)
    sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)

def simple_bar_plot(x, y, x_title = "x", y_title ="y"):
    fig = px.bar(x=x, y=y)
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis = dict(
            tickmode = 'linear',        
        )
    )
    return fig 


def plot_lda_w_vectors(W, component_labels, feature_names, width=800, height=600): 
    
    fig = px.imshow(
        W,
        y=component_labels,
        x=feature_names,
        color_continuous_scale="viridis",
    )

    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Topics",
        xaxis = {'side': 'top',  'tickangle':300}, 
    )
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
    )    

    return fig


####################################
# Word2vec visualization functions
####################################
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.manifold import TSNE

def plot_perplexity_tsne(
    digits, perplexity_range=[2, 5, 10, 15, 30, 200], title="Digits data"
):

    fig, ax = plt.subplots(
        2, 3, figsize=(24, 12), subplot_kw={"xticks": (), "yticks": ()}
    )
    colors = [
        "#476A2A",
        "#7851B8",
        "#BD3430",
        "#4A2D4E",
        "#875525",
        "#A83683",
        "#4E655E",
        "#853541",
        "#3A3120",
        "#535D8E",
    ]
    k = 0
    for i in range(2):
        for j in range(3):
            tsne = TSNE(perplexity=perplexity_range[k], random_state=42)
            digits_Z = tsne.fit_transform(digits.data)
            ax[i, j].set_xlim(digits_Z[:, 0].min(), digits_Z[:, 0].max())
            ax[i, j].set_ylim(digits_Z[:, 1].min(), digits_Z[:, 1].max())
            for dig in range(len(digits.data)):
                # actually plot the digits as text instead of using scatter
                ax[i, j].text(
                    digits_Z[dig, 0],
                    digits_Z[dig, 1],
                    str(digits.target[dig]),
                    color=colors[digits.target[dig]],
                    fontdict={"weight": "bold", "size": 9},
                )
            ax[i, j].set_title(title + " perplexity = %s"%(perplexity_range[k]))
            ax[i, j].set_xlabel("Transformed feat 0", fontsize=12)
            ax[i, j].set_ylabel("Transformed feat 1", fontsize=12)
            k += 1

def plot_digits(digits, digits_Z, title="Digits data"):
    colors = [
        "#476A2A",
        "#7851B8",
        "#BD3430",
        "#4A2D4E",
        "#875525",
        "#A83683",
        "#4E655E",
        "#853541",
        "#3A3120",
        "#535D8E",
    ]
    plt.figure(figsize=(8, 6))
    plt.xlim(digits_Z[:, 0].min(), digits_Z[:, 0].max())
    plt.ylim(digits_Z[:, 1].min(), digits_Z[:, 1].max())
    for i in range(len(digits.data)):
        # actually plot the digits as text instead of using scatter
        plt.text(
            digits_Z[i, 0],
            digits_Z[i, 1],
            str(digits.target[i]),
            color=colors[digits.target[i]],
            fontdict={"weight": "bold", "size": 9},
        )
    plt.title(title)
    plt.xlabel("Transformed feat 0", fontsize=12)
    plt.ylabel("Transformed feat 1", fontsize=12)

def plot_swiss_roll(X, colour, swiss_tsne):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1,2,1, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colour, cmap=plt.cm.Spectral)

    ax.set_title("Original data", fontsize=12)
    ax = fig.add_subplot(122)
    ax.scatter(swiss_tsne[:, 0], swiss_tsne[:, 1], c=colour, cmap=plt.cm.Spectral)
    plt.axis("tight")
    plt.xticks([]), plt.yticks([])
    plt.title("Projected data with t-SNE", fontsize=12)
    plt.show()            
    
####################################
# PCA visualization functions
####################################


def plot_pca_w_vectors(W, component_labels, feature_names, width=800, height=600): 
    
    fig = px.imshow(
        W,
        y=component_labels,
        x=feature_names,
        color_continuous_scale="viridis",
    )

    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Principal Components",
        xaxis = {'side': 'top',  'tickangle':300}, 
    )
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
    )    

    return fig


def reconstruction_error(X, X_hat):
    error = np.sum((np.array(X) - np.array(X_hat)) ** 2, axis=1)
    #error = pd.Series(data=error)
    #error = (error - np.min(error)) / (np.max(error) - np.min(error))  # normalize
    return error

def plot_pca_model_search(X, fig, alpha, w=10, h=8):
    W = np.array([[np.cos(np.radians(alpha)), np.sin(np.radians(alpha))]])
    Z = X @ (W.T @ W)

    err = np.round(reconstruction_error(X, Z).sum(), 4)    
    #mglearn.discrete_scatter(X[:, 0], X[:, 1], s=8)
    discrete_scatter(X[:, 0], X[:, 1], s=12, label_points=True)
    #plt.scatter(X[:, 0], X[:, 1], c=X[:, 0], cmap="viridis", s=60, linewidths=3)    
    plt.plot(W[0, 0] * 3.5 * np.array([-1, 1]), W[0, 1] * 3.5 * np.array([-1, 1]), "k")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    discrete_scatter(Z[:, 0], Z[:, 1], s=12, c=[0])
    #plt.scatter(Z[:, 0], Z[:, 1], c="r", s=60)
    for i in range(len(X)):
        plt.plot(
            [X[i, 0], Z[i, 0]],
            [X[i, 1], Z[i, 1]],
            color="red",
            linestyle="-.",
            alpha=0.8,
            linewidth=1.5,
        )

    n = X.shape[0]
    w = W.T @ W
    plt.title(f"Which line?\nReconstruction error: {err}", fontsize=14)
    plt.close()
    return fig

def plot_pca_reconstructions(X, pca):
    Z = pca.transform(X)
    W = pca.components_
    X_hat = pca.inverse_transform(Z)

    shapes = [
    dict(
        type="line",
        x0=X[i, 0],
        y0=X[i, 1],
        x1=X_hat[i, 0],
        y1=X_hat[i, 1],
        line=dict(color="grey", width=2),
    ) for i in range(X.shape[0])]
    
    grid = np.linspace(min(X[:,0]) - 0.3, max(X[:,1]) + 0.3, 1000)
    gridplot = (grid - pca.mean_[0]) / W[0, 0] * W[0, 1] + pca.mean_[1]
    # gridplot = (grid) / W[0, 0] * W[0, 1]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original data", "Reconstructions (X_hat)"))

    fig.add_trace(
        go.Scatter(
            x=X[:,0],
            y=X[:,1],
            mode="markers",
            marker=dict(size=8, color="linen"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=X[:,0],
            y=X[:,1],
            mode="markers",
            marker=dict(size=8, color="linen"),
            name="Original data (X)",
        ),
        row=1,
        col=2,
    )
        
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=gridplot,
            line_color="green",
            mode="lines",
            line=dict(width=2),
            name="PCA model (W vector)",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=X_hat[:, 0],
            y=X_hat[:, 1],
            mode="markers",
            marker=dict(size=8, color="blue"),
            name="Reconstructions (X_hat)",
        ),
        row=1,
        col=2,
    )        
    fig.update_layout(
        autosize=False,
        width=1000,
        height=500,)
    
    fig.update_traces(marker=dict(size=10,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))    
    
    for trace in fig["data"]:
        if trace["name"] == None:
            trace["showlegend"] = False


    for shape in shapes:
        fig.add_shape(shape, row=1, col=2)

    #fig.update_shapes(dict(xref='x', yref='y'), row =1, col= 2)
    return fig

def plot_pca_model_reconstructions(X, pca):
    Z = pca.transform(X)
    W = pca.components_
    X_hat = pca.inverse_transform(Z)

    shapes = [
    dict(
        type="line",
        x0=X.iloc[i, 0],
        y0=X.iloc[i, 1],
        x1=X_hat[i, 0],
        y1=X_hat[i, 1],
        line=dict(color="grey", width=2),
    ) for i in range(X.shape[0])]
    
    grid = np.linspace(min(X.iloc[:,0]) - 0.3, max(X.iloc[:,1]) + 0.3, 1000)
    gridplot = (grid - pca.mean_[0]) / W[0, 0] * W[0, 1] + pca.mean_[1]
    # gridplot = (grid) / W[0, 0] * W[0, 1]

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Original data X", "",  "Transformed data Z", "Reconstructions X_hat"))

    fig.add_trace(
        go.Scatter(
            x=X.iloc[:,0],
            y=X.iloc[:,1],
            mode="markers",
            marker=dict(size=8, color="linen"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=Z[:,0],
            y=np.zeros(Z.shape[0]),
            mode="markers",
            marker=dict(size=8, color="blue"),
        ),
        row=2,
        col=1,
    )    

    fig.add_trace(
        go.Scatter(
            x=X.iloc[:,0],
            y=X.iloc[:,1],
            mode="markers",
            marker=dict(size=8, color="linen"),
            name="X",
        ),
        row=2,
        col=2,
    )
        
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=gridplot,
            line_color="green",
            mode="lines",
            line=dict(width=2),
            name="W vector",
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=X_hat[:, 0],
            y=X_hat[:, 1],
            mode="markers",
            marker=dict(size=8, color="blue"),
            name="X_hat",
        ),
        row=2,
        col=2,
    )        

    fig.update_traces(marker=dict(size=8,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))        
    
    for trace in fig["data"]:
        if trace["name"] == None:
            trace["showlegend"] = False

    fig.update_layout(
        autosize=False,
        width=800,
        height=700,)
            
    for shape in shapes:
        fig.add_shape(shape, row=2, col=2)

    #fig.update_shapes(dict(xref='x', yref='y'), row =1, col= 2)
    return fig


def plot_interactive_3d(X):
    trace = go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers',
                               marker=dict(
                                        color='linen',
                                        size=8,
                                        line=dict(
                                            color='black',
                                            width=3)))
    layout = go.Layout(showlegend=False, autosize=False, width=900, height=500, 
                       scene=dict(xaxis={'title':'x1'},yaxis={'title':'x2'},zaxis={'title':'x3'}))
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
    
def plot_3d_1k(X, pca): 
    # get grid for visualizing plane
    Zgrid = np.linspace(-7, 7, 100)[:, None]
    Xgrid = pca.inverse_transform(Zgrid)
    n = X.shape[0]
    # get reconstructions of original points
    Xhat = pca.inverse_transform(pca.transform(X))

    traces1 = []
    for i in range(n):
        traces1.append(
            go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers',
                                       marker=dict(
                                                color='linen',
                                                size=8,
                                                line=dict(
                                                    color='black',
                                                    width=3)))
        )

    trace2 = go.Scatter3d(
        x=Xgrid[:, 0], y=Xgrid[:, 1], z=Xgrid[:, 2], mode="lines" #, marker={"color": "black"}
    )

    trace3 = go.Scatter3d(x=Xhat[:, 0], y=Xhat[:, 1], z=Xhat[:, 2], mode="markers", marker=dict(color='blue',size=8))

    data = traces1 + [trace2, trace3]

    layout = go.Layout(
        showlegend=False,
        autosize=False, width=900, height=500, 
        scene=dict(xaxis={"title": "x1"}, yaxis={"title": "x2"}, zaxis={"title": "x3"}),
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    

def plot_3d_2k(X, pca): 
    # get grid for visualizing plane
    n = X.shape[0]    
    z1 = np.linspace(-7, 7, 100)
    z2 = np.linspace(-7, 7, 100)
    z1grid, z2grid = np.meshgrid(z1, z2)
    Zgrid = np.concatenate((z1grid.flatten()[:, None], z2grid.flatten()[:, None]), axis=1)
    Xgrid = pca.inverse_transform(Zgrid)
    Xgrid_re = np.reshape(Xgrid, (100, 100, 3))

    # get reconstructions of original points
    Z = pca.transform(X)
    Xhat = pca.inverse_transform(Z)

    traces1 = []
    for i in range(n):
        traces1.append(
            go.Scatter3d(
                x=(X[i, 0], Xhat[i, 0]),
                y=(X[i, 1], Xhat[i, 1]),
                z=(X[i, 2], Xhat[i, 2]),
                marker=dict(color="linen", size=8,
                                        line=dict(
                                            color='black',
                                            width=3)),
                name="original points"
            )
        )

    trace2 = go.Surface(
        x=Xgrid_re[:, :, 0],
        y=Xgrid_re[:, :, 1],
        z=Xgrid_re[:, :, 2],
        showscale=False,
        colorscale=[[0, "rgb(200,300,200)"], [1, "rgb(200,300,200)"]],
        opacity=0.9,
        name="reconstructions"
    )
    
    trace3 = go.Scatter3d(x=Xhat[:, 0], y=Xhat[:, 1], z=Xhat[:, 2], mode="markers",marker=dict(color='blue',size=8))

    data = traces1 + [trace2, trace3]
    
    layout = go.Layout(
        showlegend=False,
        autosize=False, width=800, height=500, 
        scene=dict(xaxis={"title": "x1"}, yaxis={"title": "x2"}, zaxis={"title": "x3"}),
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(marker=dict(size=10,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))    
    
    iplot(fig)    
    
def plot_multicollinearity_3d(X_mc, pca):    
    W = pca.components_
    trace1 = go.Scatter3d(x=X_mc[:,0], y=X_mc[:,1], z=X_mc[:,2], mode='markers',
                               marker=dict(
                                        color='linen',
                                        size=5,
                                        line=dict(
                                            color='black',
                                            width=3)))    
    trace2 = go.Cone(x=[0, 0, 0], y=[0, 0, 0], z=[0, 0, 0], 
                     u=[W[0][0], W[1][0], W[2][0]], v=[W[0][1], W[1][1], W[2][1]], w=[W[0][2], W[1][2], W[2][2]], 
                     sizemode="absolute", sizeref=2, anchor="tip")
    layout = go.Layout(showlegend=False, autosize=False, width=900, height=500, 
                       scene=dict(xaxis={'title':'x1'},yaxis={'title':'x2'},zaxis={'title':'x3'}))    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    iplot(fig)
    
    
def plot_feature_selection(X, w=15, h=6, drop=0):
    dimension = set([0, 1])
    dimension = list(dimension.difference([drop]))[0]

    fig, ax = plt.subplots(1, 2, figsize=(w, h))
    discrete_scatter(X[:, 0], X[:, 1], s = 10, label_points=True, x1_annot=-0.05, x2_annot=0.1, ax = ax[0])
    # This for will draw the red lines into the plot
    for i in range(0, X.shape[0]):
        if dimension == 0:
            ax[0].plot(
                [X[i, 0], X[i, 0]],
                [0, X[i, 1]],
                color="red",
                linestyle="-.",
                alpha=0.8,
            )
        if dimension == 1:
            ax[0].plot(
                [0, X[i, 0]],
                [X[i, 1], X[i, 1]],
                color="red",
                linestyle="-.",
                alpha=0.8,
            )

    # This line plots the black line
    if dimension == 0:
        ax[0].plot(
            [np.min(X[:, 0]), np.max(X[:, 0])], [0, 0], color="black", linewidth=1.5
        )
    if dimension == 1:
        ax[0].plot(
            [0, 0], [np.min(X[:, 1]), np.max(X[:, 1])], color="black", linewidth=1.5
        )

    # This line plots the red points
    if dimension == 0:
        discrete_scatter(X[:, 0], np.zeros_like(X[:, 0]), c=[0], s=10, ax = ax[0])
        #ax[0].scatter(X[:, 0], np.zeros_like(X[:, 0]), s=80, color="blue")
    if dimension == 1:
        discrete_scatter(np.zeros_like(X[:, 1]), X[:, 1], c=[0], s=10, ax = ax[0])        
        #ax[0].scatter(np.zeros_like(X[:, 1]), X[:, 1], s=80, color="blue")

    ax[0].set_title("Dropping the column " + str(drop), fontsize=14)
    ax[0].set_xlabel("feature 0", fontdict={"fontsize": 14})
    ax[0].set_ylabel("feature 1", fontdict={"fontsize": 14})    
    # The functions called here are the same as above,
    # but this time is for the plot on the right
    discrete_scatter(X[:, dimension], np.zeros_like(X[:, 0]), c=[0], s=10, label_points=True, x1_annot=0, x2_annot=0.002, ax = ax[1])
    #ax[1].scatter(X[:, dimension], np.zeros_like(X[:, 0]), s=70, color="blue")
    ax[1].set_title("Projected points", fontsize=14)
    ax[1].set_yticks([])
    ax[1].set_xlabel("$Z_1$ = feature " + str(dimension), fontdict={"fontsize": 14})    
    
    
def plot_pca_regression(X, error_type='both', ax=None, title='PCA vs. Regression'):
    """
    Plot different lines for projections;
    """
   
    theta=45    

    theta = np.deg2rad(theta)
    #fig, ax = plt.subplots(1, 1, figsize=(w, h))    
    if ax is None:
        ax = plt.gca()    
    # Plot the points
    discrete_scatter(X[:, 0], X[:, 1], ax=ax)
    #ax.scatter(x_std[:, 0], x_std[:, 1])

    r = [[np.cos(theta)], [np.sin(theta)]] / \
        np.linalg.norm([np.cos(theta), np.sin(theta)])
    z = X[:, 0:2] @ r
    v = z @ r.reshape(1, -1)
    
    m = (np.max(v[:, 1]) - np.min(v[:, 1]))/(np.max(v[:, 0]) - np.min(v[:, 0]))
    error = np.round(np.sum((X[:, 0:2] - v)**2), 2)
    
    # This line plots the black line
    if np.rad2deg(theta) <= 90:
        ax.plot([np.min((v[:, 0], X[:,0])), np.max((v[:, 0], X[:,0]))],
                   [m*np.min((v[:, 0], X[:,0])), m*np.max((v[:, 0], X[:,0]))], color="green", label='model')
    else:
        ax.plot([np.min((v[:, 0], X[:,0])), np.max((v[:, 0], X[:,0]))],
                   [-m*np.min((v[:, 0], X[:,0])), -m*np.max((v[:, 0], X[:,0]))], color="green", label='model')

    # This for will draw the red lines into the plot
    #if theta > 90:
    #    m = -m
    for i in range(0, X.shape[0]):
        if error_type=='pca' or error_type=='both':
            ax.plot([X[i, 0], v[i, 0]], [X[i, 1], v[i, 1]],
                       color="black", linestyle="-")
        if error_type=='lm' or error_type=='both':
            ax.plot([X[i, 0], X[i, 0]], [X[i, 1], m*X[i, 0]],
                       color="red", linestyle="-.")

    # Removing frame sides
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.set_xlabel("feature 0", fontsize=12)
    ax.set_ylabel("feature 1", fontsize=12)    
    ax.set_title(title, fontsize=14)
    

# Directly copied from https://github.com/amueller/mglearn/blob/master/mglearn/plot_pca.py
def plot_pca_illustration():
    rnd = np.random.RandomState(5)
    X_ = rnd.normal(size=(300, 2))
    X_blob = np.dot(X_, rnd.normal(size=(2, 2))) + rnd.normal(size=2)

    pca = PCA()
    pca.fit(X_blob)
    X_pca = pca.transform(X_blob)

    S = X_pca.std(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    axes[0].set_title("Original data")
    axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=X_pca[:, 0], linewidths=0,
                    s=60, cmap='viridis')
    axes[0].set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    axes[0].arrow(pca.mean_[0], pca.mean_[1], S[0] * pca.components_[0, 0],
                  S[0] * pca.components_[0, 1], width=.1, head_width=.3,
                  color='k')
    axes[0].arrow(pca.mean_[0], pca.mean_[1], S[1] * pca.components_[1, 0],
                  S[1] * pca.components_[1, 1], width=.1, head_width=.3,
                  color='k')
    axes[0].text(-1.5, -.5, "Component 2", size=14)
    axes[0].text(-4, -4, "Component 1", size=14)
    axes[0].set_aspect('equal')

    axes[1].set_title("Transformed data")
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=X_pca[:, 0], linewidths=0,
                    s=60, cmap='viridis')
    axes[1].set_xlabel("First principal component")
    axes[1].set_ylabel("Second principal component")
    axes[1].set_aspect('equal')
    axes[1].set_ylim(-8, 8)

    pca = PCA(n_components=1)
    pca.fit(X_blob)
    X_inverse = pca.inverse_transform(pca.transform(X_blob))

    axes[2].set_title("Transformed data w/ second component dropped")
    axes[2].scatter(X_pca[:, 0], np.zeros(X_pca.shape[0]), c=X_pca[:, 0],
                    linewidths=0, s=60, cmap='viridis')
    axes[2].set_xlabel("First principal component")
    axes[2].set_aspect('equal')
    axes[2].set_ylim(-8, 8)

    axes[3].set_title("Back-rotation using only first component")
    axes[3].scatter(X_inverse[:, 0], X_inverse[:, 1], c=X_pca[:, 0],
                    linewidths=0, s=60, cmap='viridis')
    axes[3].set_xlabel("feature 1")
    axes[3].set_ylabel("feature 2")
    axes[3].set_aspect('equal')
    axes[3].set_xlim(-8, 4)
    axes[3].set_ylim(-8, 4)    
    

# Adapted from here: https://github.com/amueller/mglearn/blob/master/mglearn/plot_pca.py
def pca_faces(X_faces):
    reduced_images = []
    for n_components in [10, 50, 100, 200, 300]:
        pca = PCA(n_components=n_components)
        pca.fit(X_faces)
        X_pca = pca.transform(X_faces)
        X_hat = pca.inverse_transform(X_pca)
        reduced_images.append(X_hat)
    return reduced_images    
    

def plot_pca_faces(X_faces, image_shape, index=30):
    reduced_images = pca_faces(X_faces)

    # plot the first three images in the test set:
    fix, axes = plt.subplots(3, 6, figsize=(15, 8),
                             subplot_kw={'xticks': (), 'yticks': ()})
    for i, ax in enumerate(axes):
        # plot original image
        ax[0].imshow(X_faces[i+index].reshape(image_shape),
                     vmin=0, vmax=1)
        # plot the four back-transformed images
        for a, X_hat in zip(ax[1:], reduced_images):
            a.imshow(X_hat[i+index].reshape(image_shape), vmin=0, vmax=1)

    # label the top row
    axes[0, 0].set_title("original image")
    for ax, n_components in zip(axes[0, 1:], [10, 50, 100, 200, 300]):
        ax.set_title("%d components" % n_components)
    plt.show()    

    
def plot_strong_comp_images(X_faces, Z, W, image_shape, compn=1):
    inds = np.argsort(Z[:, compn])[::-1]
    fig, axes = plt.subplots(
        2, 5, figsize=(8, 3), subplot_kw={"xticks": (), "yticks": ()}
    )
    fig.suptitle("Large component %d" % (compn))
    for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
        if i == 0: 
            ax.imshow(W[compn].reshape(image_shape))
        else:     
            ax.imshow(X_faces[ind].reshape(image_shape))
    plt.show()


# Source: https://github.com/amueller/mglearn/blob/master/mglearn/tools.py
def print_topics(topics, feature_names, sorting, topics_per_chunk=6,
                 n_words=20):
    for i in range(0, len(topics), topics_per_chunk):
        # for each chunk:
        these_topics = topics[i: i + topics_per_chunk]
        # maybe we have less than topics_per_chunk left
        len_this_chunk = len(these_topics)
        # print topic headers
        print(("topic {:<8}" * len_this_chunk).format(*these_topics))
        print(("-------- {0:<5}" * len_this_chunk).format(""))
        # print top n_words frequent words
        for i in range(n_words):
            try:
                print(("{:<14}" * len_this_chunk).format(
                    *feature_names[sorting[these_topics, i]]))
            except:
                pass
        print("\n")

def plot_components(W, image_shape): 
    fig, axes = plt.subplots(2, 5, figsize=(10, 4), subplot_kw={"xticks": (), "yticks": ()})
    for i, (component, ax) in enumerate(zip(W, axes.ravel())):
        ax.imshow(component.reshape(image_shape), cmap="viridis")
        ax.set_title("{}. component".format((i)))
    plt.show()    
    
# Copied from here: https://github.com/amueller/mglearn/blob/master/mglearn/plot_nmf.py
from sklearn.decomposition import NMF
def plot_nmf_illustration():
    rnd = np.random.RandomState(5)
    X_ = rnd.normal(size=(300, 2))
    # Add 8 to make sure every point lies in the positive part of the space
    X_blob = np.dot(X_, rnd.normal(size=(2, 2))) + rnd.normal(size=2) + 8

    nmf = NMF(random_state=0)
    nmf.fit(X_blob)
    X_nmf = nmf.transform(X_blob)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=X_nmf[:, 0], linewidths=0,
                    s=60, cmap='viridis')
    axes[0].set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    axes[0].set_xlim(0, 12)
    axes[0].set_ylim(0, 12)
    axes[0].arrow(0, 0, nmf.components_[0, 0], nmf.components_[0, 1], width=.1,
                  head_width=.3, color='k')
    axes[0].arrow(0, 0, nmf.components_[1, 0], nmf.components_[1, 1], width=.1,
                  head_width=.3, color='k')
    axes[0].set_aspect('equal')
    axes[0].set_title("NMF with two components")

    # second plot
    nmf = NMF(random_state=0, n_components=1)
    nmf.fit(X_blob)

    axes[1].scatter(X_blob[:, 0], X_blob[:, 1], c=X_nmf[:, 0], linewidths=0,
                    s=60, cmap='viridis')
    axes[1].set_xlabel("feature 1")
    axes[1].set_ylabel("feature 2")
    axes[1].set_xlim(0, 12)
    axes[1].set_ylim(0, 12)
    axes[1].arrow(0, 0, nmf.components_[0, 0], nmf.components_[0, 1], width=.1,
                  head_width=.3, color='k')

    axes[1].set_aspect('equal')
    axes[1].set_title("NMF with one component")
    plt.show()        

    
def plot_orig_reconstructed_faces(X, reconstructed_images, image_shape=(200,200)):    
    fig, axes = plt.subplots(1, 5, figsize=(10, 6), subplot_kw={"xticks": (), "yticks": ()})
    axes[0].set_ylabel('Original')
    for image, ax in zip(X, axes.ravel()):
        ax.imshow(image.reshape(image_shape))
    plt.show()

    fig, axes = plt.subplots(1, 5, figsize=(10, 6), subplot_kw={"xticks": (), "yticks": ()})
    axes[0].set_ylabel('Reconstructed')
    for image, ax in zip(reconstructed_images, axes.ravel()):
        ax.imshow(image.reshape(image_shape))
    plt.show()    

def plot_orig_compressed(orig, compressed, n_components):
    fig, ax = plt.subplots(1, 2, figsize=[6,4], subplot_kw={'xticks': (), 'yticks': ()})
    ax[0].set_title('Original image')
    ax[0].imshow(orig, cmap=plt.cm.gray)
    ax[1].set_title("Compressed image\n n_components:{}".format(n_components))
    ax[1].imshow(compressed, cmap=plt.cm.gray)
    plt.show()    

    
    
####################################
# Clustering visualization functions
####################################
def update_Z(X, centers):
    """
    returns distances and updated cluster assignments
    """
    dist = euclidean_distances(X, centers)
    return dist, np.argmin(dist, axis=1)


def update_centers(X, Z, old_centers, k):
    """
    returns new centers
    """
    new_centers = old_centers.copy()
    for kk in range(k):
        new_centers[kk] = np.mean(X[Z == kk], axis=0)
    return new_centers



def plot_example_dist(data, centroids, fig, fontsize = 16, point_ind=None, ax=None):
    """
    Plot the distance of a point to the centroids.

    Parameters:
    -----------
    data: pd.DataFrame
        A pandas dataframe with X1 and X2 coordinate. If more than two
        coordinates, only the first two will be used.
    centroids: pd.DataFrame
        A pandas dataframe composed by k rows of data, chosen randomly. (where k 
        stands for the number of clusters)
    w: int
        width of the plot
    h: int
        height of the plot
    point: int
        the index of the point to be used to calculate the distance
    """
    if ax is None:
        ax = plt.gca()
    k = centroids.shape[0]
    if point_ind is None:
        point = np.random.choice(range(0, data.shape[0]), size=1)

    point = data[point_ind, 0:2]
    centroids = centroids[:, 0:2]

    discrete_scatter(data[:, 0], data[:, 1], s=14, label_points=True, ax=ax)
    discrete_scatter(centroids[:, 0], centroids[:, 1], y=[0,1,2], s=18,
                markers='*', ax=ax)
    # ax.set_xlabel(data.columns[0], fontdict={'fontsize': fontsize})
    # ax.set_ylabel(data.columns[1], fontdict={'fontsize': fontsize})
    #ax.scatter(point[0], point[1])
    
    dist = np.zeros(k)
    for i in range(0, k):
        l = np.row_stack((point, centroids[i, :]))
        dist[i] = np.sum((point-centroids[i, :])**2)**0.5                 
        ax.plot(l[:, 0], l[:, 1], c=colors[i], linewidth=2.0, linestyle='-.')
        if (l[0, 1] <= l[1, 1]):
            ax.text(l[1, 0]+.20, l[1, 1]+.2,
                     f"d = {np.round(dist[i], 2)}", color=colors[i],
                     fontdict={'fontsize': fontsize})
        else:
            ax.text(l[1, 0]+.15, l[1, 1]+.2,
                     f"d = {np.round(dist[i], 2)}", color=colors[i],
                     fontdict={'fontsize': fontsize})

    i = np.argmin(dist)
    l = np.row_stack((point, centroids[i, :]))
    ax.plot(l[:, 0], l[:, 1], c=colors[i], linewidth=3.5, linestyle='-')
    title = f"Point {point_ind} will be assigned to {colors[np.argmin(dist)]} cluster (min dist = {np.round(np.min(dist),2)})"
    ax.set_title(title, fontdict={'fontsize': fontsize});
    plt.close()
    return fig
    
def plot_km_initialization(X, centers):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)) 
    discrete_scatter(X[:, 0], X[:, 1], markers="o", ax=ax[0]);
    ax[0].set_title("Before clustering");    
    discrete_scatter(X[:, 0], X[:, 1], markers="o", ax=ax[1])
    discrete_scatter(
        centers[:, 0], centers[:, 1], y=np.arange(len(centers)), markers="*", s=14, ax=ax[1]
    );    
    ax[1].set_title("Initial centers");    
    
    
def plot_km_iteration(X, Z, centers, new_centers, iteration, fig, ax, fontsize=18):
    discrete_scatter(X[:,0], X[:,1], y=Z.tolist(), markers='o', s=12, ax = ax[0])
    discrete_scatter(centers[:,0], centers[:,1], y=np.arange(len(centers)), markers='*',s=18, ax = ax[0])
    ax[0].set_title(f'Iteration: {iteration}: Update Z', fontdict={'fontsize': fontsize})    
    discrete_scatter(X[:,0], X[:,1], y=Z.tolist(), markers='o', s=12, label_points=True, ax = ax[1])
    discrete_scatter(new_centers[:,0], new_centers[:,1], y=np.arange(len(centers)), markers='*',s=18, ax = ax[1])    
    aux = new_centers-(centers+(new_centers-centers)*0.9)
    aux = np.linalg.norm(aux, axis=1)    
    for i in range(0, 3):
        ax[1].arrow(centers[i, 0], centers[i, 1],
                  (new_centers[i, 0]-centers[i, 0])*0.8,
                  (new_centers[i, 1]-centers[i, 1])*0.8,
                  head_width=.1, head_length=aux[i], fc=colors[i], ec=colors[i])    
    ax[1].set_title(f'Iteration: {iteration}: Update cluster centers', fontdict={'fontsize': fontsize})
    plt.close()    
    return fig


def plot_km_iterative(X, starting_centroid, iterations=5, k=3):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)) 
    k = starting_centroid.shape[0]
    x = X[:, 0:2]
    centroids = starting_centroid.copy()
    dist, Z = update_Z(x, centroids)        
    y, inds = np.unique(Z, return_index=True)    
    # c = [Z[index] for index in sorted(inds)]
    discrete_scatter(x[:, 0], x[:, 1], y=Z, markers="o", ax=ax[0])              
    discrete_scatter(centroids[:, 0], centroids[:, 1], y=[0,1,2], markers='*', s=16, ax=ax[0])

    ax[0].set_title('Initial centers and cluster assignments')
    
    for i in range(iterations):
        discrete_scatter(x[:, 0], x[:, 1], y=Z, c=Z, markers="o", ax=ax[1])        
        new_centroids = update_centers(x, Z, centroids, k)            
        dist, Z = update_Z(x, new_centroids)
        discrete_scatter(new_centroids[:, 0], new_centroids[:, 1], y=[0,1,2], markers='*', s=16, ax=ax[1])

        aux = new_centroids-(centroids+(new_centroids-centroids)*0.9)
        aux = np.linalg.norm(aux, axis=1)
        for i in range(0, 3):
            if aux[i] > .005:
                plt.arrow(centroids[i, 0], centroids[i, 1],
                          (new_centroids[i, 0]-centroids[i, 0])*0.8,
                          (new_centroids[i, 1]-centroids[i, 1])*0.8,
                          head_width=.25, head_length=aux[i], fc=colors[i], ec=colors[i])
        centroids = new_centroids
        
    
    #plt.xlabel(data.columns[0], fontdict={'fontsize': w})
    #plt.ylabel(data.columns[1], fontdict={'fontsize': w})
    ax[1].set_title(f"Centers and cluster assignments after {iterations} iteration(s)")

def plot_silhouette_dist(w, h):

    n = 30
    df, target = make_blobs(n_samples=n,
                            n_features=2,
                            centers=[[0, 0], [1, 1], [2.5, 0]],
                            cluster_std=.15,
                            random_state=1)

    colors = np.array(['black', 'blue', 'red'])

    plt.figure(figsize=(w, h))
    ax = plt.gca()
    ax.set_ylim(-.45, 1.4)
    ax.set_xlim(-.25, 2.8)
    plt.scatter(df[:, 0], df[:, 1], c=colors[target])

    p = 1
    for i in range(0, n):
        plt.plot((df[p, 0], df[i, 0]), (df[p, 1], df[i, 1]),
                 linewidth=.7, c=colors[target[i]])

    plt.scatter(df[p, 0], df[p, 1], c="green", zorder=10, s=200)

    c1 = Circle((.1, -.12), 0.27, fill=False, linewidth=2, color='black')
    c2 = Circle((1.03, 1.04), 0.27, fill=False, linewidth=2, color='blue')
    c3 = Circle((2.48, 0.1), 0.27, fill=False, linewidth=2, color='red')
    ax.add_artist(c1)
    ax.add_artist(c2)
    ax.add_artist(c3)
    plt.xlabel("X1", fontdict={'fontsize': w})
    plt.ylabel("X2", fontdict={'fontsize': w})
    plt.title("Distances for silhouette", fontdict={'fontsize': w+h})


def Gaussian_mixture_1d(ϕ1, ϕ2, fig, μ1=0.0, μ2=5.0, Σ1=1, Σ2=3):
    """
    Plot a Gaussian Mixture with two components. 
    
    Parameters:
    -----------
    μ1: float
       the mean of the first Gaussian
    μ2: float
       the mean of the second Gaussian 
    Σ1: float
       the variance of the first Gaussian
    Σ2: float
       the variance of the second Gaussian
    ϕ1: float > 0
       the weight of the first component
    ϕ2: float > 0
       the weight of the second component
    w: int
       The width of the plot
    h: int
       the height of the plot       
    """

    # Creating the DataFrame
    data = pd.DataFrame({'x': np.arange(np.min([μ1 - 4*Σ1, μ2 - 4*Σ2]), np.max([μ1 + 4*Σ1, μ2 + 4*Σ2]), 1/1000)})
    data['f1(x|mu1,Sigma1)'] = scipy.stats.norm.pdf(data['x'], μ1, Σ1)
    data['f2(x|mu2,Sigma2)'] = scipy.stats.norm.pdf(data['x'], μ2, Σ2)
    data['mixture'] = ϕ1 * data['f1(x|mu1,Sigma1)'] + ϕ2 * data['f2(x|mu2,Sigma2)']


    ## Plotting
    plt.plot('x', 'mixture', data=data, linewidth=4, label='Mixture')
    plt.plot('x', 'f1(x|mu1,Sigma1)', linestyle='--', linewidth=3, alpha = .50, data=data, color='green', label=f'$f_1(x|\mu_1={μ1},\Sigma_1={Σ1})$')
    plt.plot('x', 'f2(x|mu2,Sigma2)', linestyle='--', linewidth=3, alpha = .50, data=data, color='red', label=f'$f_2(x|\mu_2={μ2},\Sigma_2={Σ2})$')
    plt.legend(fontsize=16)
    plt.title(f'Gaussian Mixture: $\Phi_1$ = {ϕ1} and $\Phi_2$ = {ϕ2}', fontdict={'fontsize':18})
    plt.close()    
    return fig

def plot_kmeans_circles(kmeans, X, n_clusters=3, ax=None):
    km_labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    discrete_scatter(X[:,0], X[:,1], km_labels, c=km_labels, markers='o', markeredgewidth=0.2, ax=ax);
    discrete_scatter(
        centers[:, 0], centers[:, 1], y=[0,1,2], markers="*", s=18
    );
    radii = [cdist(X[km_labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for center, radius in zip(centers, radii):
        ax.add_patch(plt.Circle(center, radius, fc='gray', alpha=0.4))
        
        
def plot_cov_types(X_train, gmm_full_labels, gmm_tied_labels, gmm_diag_labels, gmm_spherical_labels): 
    fig, ax = plt.subplots(2, 2, figsize=(12, 8)) 
    discrete_scatter(X_train[:, 0], X_train[:, 1], gmm_spherical_labels, c=gmm_spherical_labels, markers="o", ax=ax[0][0]);
    ax[0][0].set_title('Spherical');
    discrete_scatter(X_train[:, 0], X_train[:, 1], gmm_diag_labels, c=gmm_diag_labels, markers="o", ax=ax[0][1]);
    ax[0][1].set_title('diag')
    discrete_scatter(X_train[:, 0], X_train[:, 1], gmm_tied_labels, c=gmm_tied_labels, markers="o", ax=ax[1][0]);
    ax[1][0].set_title('tied')
    discrete_scatter(X_train[:, 0], X_train[:, 1], gmm_full_labels, c=gmm_full_labels, markers="o", ax=ax[1][1]);
    ax[1][1].set_title('full')
    
def make_ellipses(gmm, ax):
    colors = ['xkcd:azure', 'yellowgreen', 'tomato']    
    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], 2*v[0], 2*v[1], angle=180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")    
        
def plot_gmm_cov_types(estimators, X_train): 
    n_estimators = len(estimators)

    plt.figure(figsize=(3 * n_estimators // 2, 5))
    plt.subplots_adjust(
        bottom=0.01, top=0.95, hspace=0.15, wspace=0.05, left=0.01, right=0.99
    )

    for index, (name, estimator) in enumerate(estimators.items()):
        # Train the other parameters using the EM algorithm.
        estimator.fit(X_train)
        labels = estimator.predict(X_train)
        h = plt.subplot(2, n_estimators // 2, index + 1)
        discrete_scatter(X_train[:, 0], X_train[:, 1], labels, c=labels, markers="o", markeredgewidth=0.2, ax=h);    
        make_ellipses(estimator, h)

        plt.xticks(())
        plt.yticks(())
        plt.title(name)

    plt.legend(scatterpoints=1, loc="upper right", prop=dict(size=12))

    
def get_cluster_images(model, Z, inputs, cluster=0, n_img=5):
    fig, axes = plt.subplots(1, n_img + 1, subplot_kw={'xticks': (), 'yticks': ()},
                             figsize=(10, 10), gridspec_kw={"hspace": .3})
    img_shape = [3,200,200]
    transpose_axes = (1,2,0)      
    
    if type(model).__name__ == 'KMeans': 
        center = model.cluster_centers_[cluster]
        mask = model.labels_ == cluster
        dists = np.sum((Z - center) ** 2, axis=1)
        dists[~mask] = np.inf
        inds = np.argsort(dists)[:n_img]        
        if Z.shape[1] == 1024: 
            axes[0].imshow(center.reshape((32,32)))
        else:
            axes[0].imshow(np.transpose(center.reshape(img_shape) / 2 + 0.5, transpose_axes))
        axes[0].set_title('Cluster center %d'%(cluster))       
    if type(model).__name__ == 'GaussianMixture':         
        cluster_probs = model.predict_proba(Z)[:,cluster]
        inds = np.argsort(cluster_probs)[-n_img:]        
        axes[0].imshow(np.transpose(inputs[inds[0]].reshape(img_shape) / 2 + 0.5, transpose_axes))
        axes[0].set_title('Cluster %d'%(cluster))   
        
    i = 1
    print('Image indices: ', inds)
    for image in inputs[inds]:
        axes[i].imshow(np.transpose(image/2 + 0.5 , transpose_axes))
        i+=1
    plt.show()    
    
def plot_original_clustered(X, model, labels):
    k = np.unique(labels).shape[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))    
    ax[0].set_title("Original dataset")
    ax[0].set_xlabel("Feature 0")
    ax[0].set_ylabel("Feature 1")    
    discrete_scatter(X[:, 0], X[:, 1], ax=ax[0]);
    # cluster the data into three clusters
    # plot the cluster assignments and cluster centers
    ax[1].set_title(f"{type(model).__name__} clusters")    
    ax[1].set_xlabel("Feature 0")
    ax[1].set_ylabel("Feature 1")

    discrete_scatter(X[:, 0], X[:, 1], labels, c=labels, markers='o', ax=ax[1]); 
    if type(model).__name__ == "KMeans": 
        discrete_scatter(
            model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], y=np.arange(0,k), s=15, 
            markers='*', markeredgewidth=1.0, ax=ax[1])
        
def plot_kmeans_gmm(X, k):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))    
    ax[0].set_title("Original dataset")
    ax[0].set_xlabel("Feature 0")
    ax[0].set_ylabel("Feature 1")    
    discrete_scatter(X[:, 0], X[:, 1], ax=ax[0]);
    # cluster the data into three clusters
    # plot the cluster assignments and cluster centers

    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X)
    ax[1].set_title(f"KMeans clusters n_clusters={k}")    
    ax[1].set_xlabel("Feature 0")
    ax[1].set_ylabel("Feature 1")        
    discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o', ax=ax[1])
    discrete_scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], range(0,k), s=15, 
        markers='*', markeredgewidth=1.0, ax=ax[1])   
    
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)
    ax[2].set_title(f"Gaussian Mixture clusters n_components={k}")    
    ax[2].set_xlabel("Feature 0")
    ax[2].set_ylabel("Feature 1")        
    discrete_scatter(X[:, 0], X[:, 1], gmm.predict(X), markers='o', ax=ax[2])


def plot_dbscan_with_labels(X, fig, eps=1.0, min_samples = 2, font_size=14):
    model = DBSCAN(eps=eps, min_samples=min_samples) 
    model.fit(X)   
    if np.any(model.labels_ == -1):
        n_clusters = len(set(model.labels_)) - 1 
    else: 
        n_clusters = len(set(model.labels_))
    plt.title('Number of clusters: %d'%(n_clusters))
    db_colors = ['xkcd:azure', 'yellowgreen', 'tomato', 'teal', 'orangered', 'orchid', 'black', 'xkcd:turquoise' , 'wheat']    
    # colours = []
    # for i in range(n_clusters + 1):
    #     colours.append("#%06X" % np.random.randint(0, 0xFFFFFF))
        
    if np.any(model.labels_ == -1):
        db_colors = db_colors + ["white"]
    discrete_scatter(
        X[:, 0], X[:, 1], model.labels_, c = db_colors, markers="o", markeredgewidth=1.0
    );
    plt.legend()
    labels = [str(label) for label in list(range(0,len(X)))]
    for i, txt in enumerate(labels):
        plt.annotate(txt, X[i], xytext=X[i] + 0.2, size = font_size)
    plt.close()
    return fig

def plot_k_means_dbscan_comparison(X, k=3, eps=1.0, min_samples=2):
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    discrete_scatter(X[:, 0], X[:, 1], ax=ax[0], markeredgewidth=1.0)

    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X)
    colours = list(range(0,k)) 
    # plot the cluster assignments and cluster centers
    ax[1].set_title("K-Means clusters (K=%d)"%(k))
    ax[1].set_xlabel("Feature 0")
    ax[1].set_ylabel("Feature 1")        
    discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o', ax=ax[1], markeredgewidth=1.0)

    discrete_scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], colours,   
        markers='*', markeredgewidth=1.5, ax=ax[1])

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    n_clusters = len(set(dbscan.labels_))
    # colours = []
    # for i in range(n_clusters):
    #     colours.append("#%06X" % np.random.randint(0, 0xFFFFFF))
    colours = np.arange(0,n_clusters)
    print(colours)
    # if np.any(dbscan.labels_ == -1):
    #     colours = ["w"] + colours
    # plot the cluster assignments and cluster centers
    ax[2].set_title("DBSCAN clusters eps=%0.2f and min_samples=%d"%(eps,min_samples))
    ax[2].set_xlabel("Feature 0")
    ax[2].set_ylabel("Feature 1")        
    discrete_scatter(
        X[:, 0], X[:, 1], y=dbscan.labels_, c=colours, markers='o', markeredgewidth=1.0, ax=ax[2]
    );    
    plt.legend()
    
#cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])

# Credit: Based on https://github.com/amueller/mglearn/blob/master/mglearn/plot_dbscan.py
def plot_dbscan():
    X, y = make_blobs(random_state=0, n_samples=12)

    dbscan = DBSCAN()
    clusters = dbscan.fit_predict(X)
    clusters

    fig, axes = plt.subplots(3, 4, figsize=(11, 8),
                             subplot_kw={'xticks': (), 'yticks': ()})
    # Plot clusters as red, green and blue, and outliers (-1) as white
    markers = ['o', 'o', 'o']

    # iterate over settings of min_samples and eps
    for i, min_samples in enumerate([2, 3, 5]):
        for j, eps in enumerate([1, 1.5, 2, 3]):
            # instantiate DBSCAN with a particular setting
            dbscan = DBSCAN(min_samples=min_samples, eps=eps)
            # get cluster assignments
            clusters = dbscan.fit_predict(X)
            print("min_samples: %d eps: %f  cluster: %s"
                  % (min_samples, eps, clusters))
            if np.any(clusters == -1):
                c =  [-2]
                m =  markers + ['o']
            else:
                c = colors
                m = markers
            discrete_scatter(X[:, 0], X[:, 1], clusters, ax=axes[i, j], s=5,  c=c, markers=m)
            inds = dbscan.core_sample_indices_
            # vizualize core samples and clusters.
            if len(inds):
                discrete_scatter(X[inds, 0], X[inds, 1], clusters[inds],
                                 ax=axes[i, j], s=8, c=colors,
                                 markers=markers)
            axes[i, j].set_title("min_samples: %d eps: %.1f"
                                 % (min_samples, eps))
    fig.tight_layout()    
    

def plot_X_dendrogram(X, linkage_array, font_size=14, label_n_clusters=False, title='Dendrogram'): 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    discrete_scatter(X[:, 0], X[:, 1], markeredgewidth=1.0, ax = axes[0]); 
    axes[0].set_title('Original data')
    labels = [str(label) for label in list(range(0,len(X)))]
    for i, txt in enumerate(labels):
        axes[0].annotate(txt, X[i], xytext=X[i] + 0.2, size = font_size)
        
    # Credit: Based on the code in Introduction to Machine Learning with Python        
    dendrogram(linkage_array, ax=axes[1])
    axes[1] = plt.gca()    
    axes[1].set_title(title)    
    if label_n_clusters: 
        bounds = axes[1].get_xbound()
        axes[1].plot(bounds, [7.0, 7.0], "--", c="k")
        axes[1].plot(bounds, [4, 4], "--", c="k")
        axes[1].plot(bounds, [2, 2], "--", c="k")    
        #axes[1].plot(bounds, [4, 4], "--", c="k")
        #axes[1].plot(bounds, [2, 2], "--", c="k")        
        #axes[1].plot(bounds, [1.3, 1.3], "--", c="k")            
        axes[1].text(bounds[1], 7.25, " two clusters", va="center", fontdict={"size": 15})
        axes[1].text(bounds[1], 4, " three clusters", va="center", fontdict={"size": 15})
        axes[1].text(bounds[1], 2, " four clusters", va="center", fontdict={"size": 15})    
        #axes[1].text(bounds[1], 1.3, " four clusters", va="center", fontdict={"size": 15})            
    plt.xlabel("Examples")
    plt.ylabel("Cluster distance");     
    
def hc_truncation_toy_demo(linkage_array):
    # Credit: adapted from here: https://stackoverflow.com/questions/66180002/scipy-cluster-hierarchy-dendrogram-exactly-what-does-truncate-mode-level-do
    fig, ax_rows = plt.subplots(ncols=5, nrows=2, sharey=True, figsize=(14, 5))

    for ax_row, truncate_mode in zip(ax_rows, ['level', 'lastp']):
        dendrogram(linkage_array, p=4, truncate_mode="level", ax=ax_row[0])
        ax_row[0].set_title('default, no truncation')
        for ind, ax in enumerate(ax_row[1:]):
            if truncate_mode == 'level':
                p = len(ax_row) - ind - 1
            else:
                p = len(ax_row) - ind
            dendrogram(linkage_array, truncate_mode=truncate_mode, p=p, ax=ax)
            ax.set_title(f"truncate_mode='{truncate_mode}', p={p}")
    plt.tight_layout()
    plt.show()        

def print_dbscan_noise_images(Z, inputs, labels, transpose_axes=(1,2,0)):
    noise = inputs[labels == -1]

    fig, axes = plt.subplots(
        2, 9, subplot_kw={"xticks": (), "yticks": ()}, figsize=(12, 4)
    )
    for image, ax in zip(noise, axes.ravel()):
        ax.imshow(np.transpose(image.numpy()/2 + 0.5 , transpose_axes))

def print_dbscan_clusters(Z, inputs, labels, transpose_axes=(1,2,0)):
    i = 0
    for cluster in range(max(labels) + 1):
        mask = labels == cluster
        n_images = np.sum(mask)
        fig, axes = plt.subplots(
            1,
            n_images,
            figsize=(n_images * 1.5, 4),
            subplot_kw={"xticks": (), "yticks": ()},
        )
        for image, ax in zip(inputs[mask], axes):
            ax.imshow(np.transpose(image.numpy()/2 + 0.5 , transpose_axes))
            ax.set_title("cluster %d" % (i))
        i += 1            

def print_hierarchical_clusters(inputs, Z, cluster_labels, unique_cluster_labels=[2, 3, 7, 8, 13, 15, 17, 19, 20, 21, 22, 27, 29, 30]):
    transpose_axes = (1,2,0)
    for cluster in unique_cluster_labels: # hand-picked "interesting" clusters
        mask = cluster_labels == cluster
        fig, axes = plt.subplots(
            1, 15, subplot_kw={"xticks": (), "yticks": ()}, figsize=(15, 8)
        )
        cluster_size = np.sum(mask)

        for image, label, ax in zip(
            inputs[mask], cluster_labels[mask], axes
        ):            
            ax.imshow(np.transpose(image.numpy()/2 + 0.5 , transpose_axes))
            ax.set_title(label, fontdict={"fontsize": 9})
        for i in range(cluster_size, 15):
            axes[i].set_visible(False)

def plot_dendrogram_clusters(X, linkage_array, hier_labels, linkage_type='single', title=None): 
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)) 
    dendrogram(linkage_array, ax=ax[0])
    ax[0].set_xlabel("Sample index")
    ax[0].set_ylabel("Cluster distance");
    ax[0].set_title(f"{linkage_type} linkage")
    discrete_scatter(X[:, 0], X[:,1], hier_labels, markers='o', label_points=True, ax=ax[1]);
    ax[1].set_title(title)            
            
def plot_linkage_criteria(X, n_clusters):
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))    
    criteria = [single, complete, average, ward]
    for (i, criterion) in enumerate(criteria): 
        linkage_array = criterion(X)
        hier_labels = fcluster(linkage_array, n_clusters, criterion="maxclust")       
        discrete_scatter(X[:, 0], X[:, 1], hier_labels, c=hier_labels, markers='o', ax=ax[i], alpha=0.6);
        cluster_sizes = np.bincount(hier_labels-1)
        ax[i].set_title(criterion.__name__ + f' linkage \n cluster sizes: {cluster_sizes}')
        
def plot_sup_x_unsup(data, w, h):
    """
        Function to generate a supervised vs unsupervised plot.
        Parameters:
        -----------
        data: pd.DataFrame
            A pandas dataframe with X1 and X2 coordinate, and a target column
            for the classes.
        w: int
            Width of the plot
        h: int
            height of the plot
    """
    # Colors to be used (upt to 5 classes)
    colors = np.array(['black', 'blue', 'red', 'green', 'purple'])

    # Getting the column and classes' names
    col_names = data.columns.to_numpy()
    target_names = data['target'].to_numpy()

    # Getting numerical values for the classes labels
    target = np.unique(data['target'].to_numpy(), return_inverse=True)

    # Getting X1 and X2
    data = data.iloc[:, 0:2].to_numpy()

    # Creates the Figure
    plt.figure(0, figsize=(w, h))

    # Create two subplots
    plt.subplots_adjust(right=2.5)

    # Get the first subplot, which is the Supervised one.
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    for i, label in enumerate(target[0]):
        plt.scatter(data[target_names == label, 0],
                    data[target_names == label, 1],
                    c=colors[i], label=label)

    # Creates the legend
    plt.legend(loc='best', fontsize=22, frameon=True)

    # Name the axes and creates title
    plt.xlabel(col_names[0], fontsize=1.5*(w + h))
    plt.ylabel(col_names[1], fontsize=1.5*(w + h))
    plt.title("Supervised", fontdict={'fontsize': 2 * (w + h)})

#     ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize': w + h})
#     ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize': w + h})

    # Creates the unsupervised subplot.
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    plt.scatter(data[:, 0], data[:, 1])
    plt.title("Unsupervised", fontdict={'fontsize': 2 * (w + h)})
    plt.xlabel(col_names[0], fontsize=1.5*(w + h))
    plt.ylabel(col_names[1], fontsize=1.5*(w + h))
#     ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize': w + h})
#     ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize': w + h})
    
    
            