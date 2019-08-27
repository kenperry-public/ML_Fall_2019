
# coding: utf-8

# # PCA
# Find the SLC (e.g. a projection) with maximum variance: 
# $$\max_{a:||a||=1}Var(a^TX) = \max_{a:||a||=1}a^TVar(X)a$$
# 
# Setting $a=\gamma$, with $\gamma$ the Eigenvector of the largest Eigenvalue $\lambda$ of $Var(X)$ will satisfy this OP.
# 
# $$Var(X) \gamma=\lambda \gamma$$
# $$(Var(X)-\lambda I)\gamma=0$$
# $$ |Var(X)-\lambda I|=0$$
# 
# Yields Eigenvalues $\lambda$ of Var(X). Plugging back into second equation above gives Eigenvectors. The result can be written as:
# 
# $$\boldsymbol{\lambda}=\boldsymbol{\gamma}^TVar(X)\boldsymbol{\gamma}$$
# 
# Where $\boldsymbol{\lambda}$ is the diagonal matrix of Eigenvalues and $\boldsymbol{\gamma}$ is the corresponding matrix of Eigenvectors. Rearanging gives the spectral decomposition of the covarianvce matrix.
# 
# $$Var(X)=\boldsymbol{\gamma}\boldsymbol{\lambda}\boldsymbol{\gamma}^T$$
# 
# The transformation of X onto the orthonormal basis spanned by $\gamma$ is:
# $$X_{PCA}=\boldsymbol{\gamma}^TX$$

# # Libs & Defs

# In[35]:

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import math

from sklearn.decomposition import PCA, KernelPCA


# # Load Dataset

# In[2]:

df = pd.read_csv('Marktzinsen_mod.csv', sep=',')

df['Datum'] = pd.to_datetime(df['Datum'],infer_datetime_format=True)

df.set_index('Datum', drop=True, inplace=True)

df.index.names = [None]

df.drop('Index', axis=1, inplace=True)

dt = df.transpose()


# # Visualizing the Dataset

# In[39]:

plt.figure(figsize=(20,15))

plt.plot(df.index, df)
plt.xlim(df.index.min(), df.index.max())
# plt.ylim(0, 0.1)
plt.axhline(y=0,c="grey",linewidth=0.5,zorder=0)
for i in range(df.index.min().year, df.index.max().year+1):
    plt.axvline(x=df.index[df.index.searchsorted(pd.datetime(i,1,1))-1],
                c="grey", linewidth=0.5, zorder=0)


# In[42]:

cols = 6
num_years = df.index.max().year-df.index.min().year
rows = math.ceil(num_years/cols)

plt.figure(figsize=(24,(24/cols)*rows))

plt.subplot2grid((rows,cols), (0,0), colspan=cols, rowspan=fig1_rows)


colnum = 0
rownum = 0
for year in range(df.index.min().year,df.index.max().year+1):
    year_start = df.index[df.index.searchsorted(pd.datetime(year,1,1))]
    year_end = df.index[df.index.searchsorted(pd.datetime(year,12,31))]
    
    plt.subplot2grid((rows,cols), (rownum,colnum), colspan=1, rowspan=1)
    plt.title('{0}'.format(year))
    plt.xlim(0, len(dt.index)-1)
    plt.ylim(np.min(dt.values), np.max(dt.values))
    plt.xticks(range(len(dt.index)), dt.index, size='small')
    
    plt.plot(dt.ix[:,year_start:year_end].values)
    
    if colnum != cols-1:
        colnum += 1
    else:
        colnum = 0
        rownum += 1

None


# # Projection onto Principal Components

# In[71]:

# calculate the PCA (Eigenvectors & Eigenvalues of the covariance matrix)
pcaA = PCA(n_components=3, copy=True, whiten=False)

# pcaA = KernelPCA(n_components=3,
#                  kernel='rbf',
#                  gamma=2.0, # default 1/n_features
#                  kernel_params=None,
#                  fit_inverse_transform=False,
#                  eigen_solver='auto',
#                  tol=0,
#                  max_iter=None)

# transform the dataset onto the first two eigenvectors
pcaA.fit(df)
dpca = pd.DataFrame(pcaA.transform(df))
dpca.index = df.index


# In[85]:

for i,pc in enumerate(pcaA.explained_variance_ratio_):
    print('{0}.\t{1:2.2f}%'.format(i+1,pc*100.0))


# In[53]:

fig = plt.figure(figsize=(16,10))
plt.title('First {0} PCA components'.format(np.shape(np.transpose(pcaA.components_))[-1]))

plt.plot(np.transpose(pcaA.components_), label=['1. PC', '2. PC'])
plt.legend('upper right')

None


# In[54]:

# plot the result
merged_years = 1
pc1 = 0
pc2 = 1
fig = plt.figure(figsize=(16,12))
plt.title('Projection on {0}. and {1}. PC'.format(pc1+1,pc2+1))
plt.axhline(y=0,c="grey",linewidth=1.0,zorder=0)
plt.axvline(x=0,c="grey",linewidth=1.0,zorder=0)
    
sc = plt.scatter(dpca.loc[:,pc1],dpca.loc[:,pc2], c=[d.year for d in dpca.index], cmap='rainbow')
cb = plt.colorbar(sc)
cb.set_ticks(ticks=np.unique([d.year for d in dpca.index])[::1])
cb.set_ticklabels(np.unique([d.year for d in dpca.index])[::1])

for year in range(df.index.min().year,df.index.max().year+1,merged_years):
    year_start = df.index[df.index.searchsorted(pd.datetime(year,1,1))]
    year_end = df.index[df.index.searchsorted(pd.datetime(year+merged_years-1,12,31))]
    
    plt.annotate('{0}'.format(year), xy=(dpca.loc[year_start,pc1],dpca.loc[year_start,pc2]), xytext=(dpca.loc[year_start,pc1],dpca.loc[year_start,pc2]))

None


# In[55]:

# plot the result
merged_years = 1
pc1 = 0
pc2 = 2
fig = plt.figure(figsize=(16,12))
plt.title('Projection on {0}. and {1}. PC'.format(pc1+1,pc2+1))
plt.axhline(y=0,c="grey",linewidth=1.0,zorder=0)
plt.axvline(x=0,c="grey",linewidth=1.0,zorder=0)
    
sc = plt.scatter(dpca.loc[:,pc1],dpca.loc[:,pc2], c=[d.year for d in dpca.index], cmap='rainbow')
cb = plt.colorbar(sc)
cb.set_ticks(ticks=np.unique([d.year for d in dpca.index])[::1])
cb.set_ticklabels(np.unique([d.year for d in dpca.index])[::1])

for year in range(df.index.min().year,df.index.max().year+1,merged_years):
    year_start = df.index[df.index.searchsorted(pd.datetime(year,1,1))]
    year_end = df.index[df.index.searchsorted(pd.datetime(year+merged_years-1,12,31))]
    
    plt.annotate('{0}'.format(year), xy=(dpca.loc[year_start,pc1],dpca.loc[year_start,pc2]), xytext=(dpca.loc[year_start,pc1],dpca.loc[year_start,pc2]))

None


# In[56]:

# plot the result
merged_years = 1
pc1 = 1
pc2 = 2
fig = plt.figure(figsize=(16,12))
plt.title('Projection on {0}. and {1}. PC'.format(pc1+1,pc2+1))
plt.axhline(y=0,c="grey",linewidth=1.0,zorder=0)
plt.axvline(x=0,c="grey",linewidth=1.0,zorder=0)
    
sc = plt.scatter(dpca.loc[:,pc1],dpca.loc[:,pc2], c=[d.year for d in dpca.index], cmap='rainbow')
cb = plt.colorbar(sc)
cb.set_ticks(ticks=np.unique([d.year for d in dpca.index])[::1])
cb.set_ticklabels(np.unique([d.year for d in dpca.index])[::1])

for year in range(df.index.min().year,df.index.max().year+1,merged_years):
    year_start = df.index[df.index.searchsorted(pd.datetime(year,1,1))]
    year_end = df.index[df.index.searchsorted(pd.datetime(year+merged_years-1,12,31))]
    
    plt.annotate('{0}'.format(year), xy=(dpca.loc[year_start,pc1],dpca.loc[year_start,pc2]), xytext=(dpca.loc[year_start,pc1],dpca.loc[year_start,pc2]))

None


# # Principal Components by Year

# In[70]:

pca = PCA(n_components=2, copy=True, whiten=False)

merged_years = 4

cols = 3
num_years = df.index.max().year-df.index.min().year
rows = math.ceil(num_years/cols)

plt.figure(figsize=(24,(24/cols)*rows))

colnum = 0
rownum = 0
for year in range(df.index.min().year,df.index.max().year+1,merged_years):
    year_start = df.index[df.index.searchsorted(pd.datetime(year,1,1))]
    year_end = df.index[df.index.searchsorted(pd.datetime(year+merged_years-1,12,31))]
    
    pca.fit(df.ix[year_start:year_end,:].values)
    pca_components = np.transpose(pca.components_)

    plt.subplot2grid((rows,cols), (rownum,colnum), colspan=1, rowspan=1)
    plt.title('{0} - {1}'.format(year_start.year, year_end.year))
    plt.xlim(0, len(pca_components)-1)
    plt.ylim(-0.5, 0.6)
    plt.xticks(range(len(pca_components)), dt.index, size='small')
    
    for i, comp in enumerate(pca.components_):
        plt.plot(pcaA.components_[i], label='{0}. PC'.format(i+1), color='#dddddd')
        plt.plot(comp, label='{0}. PC'.format(i+1))
    plt.legend(loc='upper right')
    
    if colnum != cols-1:
        colnum += 1
    else:
        colnum = 0
        rownum += 1

None


# In[72]:

pca = PCA(n_components=2, copy=True, whiten=False)

merged_years = 4

cols = 3
num_years = df.index.max().year-df.index.min().year
rows = math.ceil(num_years/cols)

plt.figure(figsize=(24,(24/cols)*rows))

colnum = 0
rownum = 0
for year in range(df.index.min().year,df.index.max().year+1,merged_years):
    year_start = df.index[df.index.searchsorted(pd.datetime(year,1,1))]
    year_end = df.index[df.index.searchsorted(pd.datetime(year+merged_years-1,12,31))]
    
    pca.fit(df.ix[year_start:year_end,:].values)
    pca_components = np.transpose(pca.components_)

    plt.subplot2grid((rows,cols), (rownum,colnum), colspan=1, rowspan=1)
    plt.title('{0} - {1}'.format(year_start.year, year_end.year))
    plt.xlim(0, len(pca_components)-1)
    plt.ylim(-0.8, 0.8)
    plt.xticks(range(len(pca_components)), dt.index, size='small')
    plt.axhline(y=0,c="grey",linewidth=1.0,zorder=0)
    
    for i, comp in enumerate(pca.components_):
        plt.plot(pcaA.components_[i]-comp, label='{0}. PC'.format(i+1))
        
    plt.legend(loc='upper right')
    
    if colnum != cols-1:
        colnum += 1
    else:
        colnum = 0
        rownum += 1

None


# # Kernel PCA

# In[67]:

# calculate the PCA (Eigenvectors & Eigenvalues of the covariance matrix)
# pcaA = PCA(n_components=3, copy=True, whiten=False)

pcaA = KernelPCA(n_components=3,
                 kernel='rbf',
                 gamma=4, # default 1/n_features
                 kernel_params=None,
                 fit_inverse_transform=False,
                 eigen_solver='auto',
                 tol=0,
                 max_iter=None)

# transform the dataset onto the first two eigenvectors
pcaA.fit(df)
dpca = pd.DataFrame(pcaA.transform(df))
dpca.index = df.index


# In[68]:

# plot the result
merged_years = 1
pc1 = 0
pc2 = 1
fig = plt.figure(figsize=(16,12))
plt.title('Projection on {0}. and {1}. PC'.format(pc1+1,pc2+1))
plt.axhline(y=0,c="grey",linewidth=1.0,zorder=0)
plt.axvline(x=0,c="grey",linewidth=1.0,zorder=0)
    
sc = plt.scatter(dpca.loc[:,pc1],dpca.loc[:,pc2], c=[d.year for d in dpca.index], cmap='rainbow')
cb = plt.colorbar(sc)
cb.set_ticks(ticks=np.unique([d.year for d in dpca.index])[::1])
cb.set_ticklabels(np.unique([d.year for d in dpca.index])[::1])

for year in range(df.index.min().year,df.index.max().year+1,merged_years):
    year_start = df.index[df.index.searchsorted(pd.datetime(year,1,1))]
    year_end = df.index[df.index.searchsorted(pd.datetime(year+merged_years-1,12,31))]
    
    plt.annotate('{0}'.format(year), xy=(dpca.loc[year_start,pc1],dpca.loc[year_start,pc2]), xytext=(dpca.loc[year_start,pc1],dpca.loc[year_start,pc2]))

None


# In[69]:

# plot the result
merged_years = 1
pc1 = 0
pc2 = 2
fig = plt.figure(figsize=(16,12))
plt.title('Projection on {0}. and {1}. PC'.format(pc1+1,pc2+1))
plt.axhline(y=0,c="grey",linewidth=1.0,zorder=0)
plt.axvline(x=0,c="grey",linewidth=1.0,zorder=0)
    
sc = plt.scatter(dpca.loc[:,pc1],dpca.loc[:,pc2], c=[d.year for d in dpca.index], cmap='rainbow')
cb = plt.colorbar(sc)
cb.set_ticks(ticks=np.unique([d.year for d in dpca.index])[::1])
cb.set_ticklabels(np.unique([d.year for d in dpca.index])[::1])

for year in range(df.index.min().year,df.index.max().year+1,merged_years):
    year_start = df.index[df.index.searchsorted(pd.datetime(year,1,1))]
    year_end = df.index[df.index.searchsorted(pd.datetime(year+merged_years-1,12,31))]
    
    plt.annotate('{0}'.format(year), xy=(dpca.loc[year_start,pc1],dpca.loc[year_start,pc2]), xytext=(dpca.loc[year_start,pc1],dpca.loc[year_start,pc2]))

None


# In[70]:

# plot the result
merged_years = 1
pc1 = 1
pc2 = 2
fig = plt.figure(figsize=(16,12))
plt.title('Projection on {0}. and {1}. PC'.format(pc1+1,pc2+1))
plt.axhline(y=0,c="grey",linewidth=1.0,zorder=0)
plt.axvline(x=0,c="grey",linewidth=1.0,zorder=0)
    
sc = plt.scatter(dpca.loc[:,pc1],dpca.loc[:,pc2], c=[d.year for d in dpca.index], cmap='rainbow')
cb = plt.colorbar(sc)
cb.set_ticks(ticks=np.unique([d.year for d in dpca.index])[::1])
cb.set_ticklabels(np.unique([d.year for d in dpca.index])[::1])

for year in range(df.index.min().year,df.index.max().year+1,merged_years):
    year_start = df.index[df.index.searchsorted(pd.datetime(year,1,1))]
    year_end = df.index[df.index.searchsorted(pd.datetime(year+merged_years-1,12,31))]
    
    plt.annotate('{0}'.format(year), xy=(dpca.loc[year_start,pc1],dpca.loc[year_start,pc2]), xytext=(dpca.loc[year_start,pc1],dpca.loc[year_start,pc2]))

None

