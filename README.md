This analysis allows you to study the following things (if given roster plots as innput either from MEA or any imaging analysis)

1. Basic network analysis (including time-dependent for each of the features below and including Lag Window (10 ms and 20 ms)):
   
    - correlation coefficient distibution
    - clustering coefficient distribution
    - degree distribution
    - mean community size
    - number of communities
    - number of isolated nodes
    - normalized number of pairs in log scale vs correlation coefficient and the fit function, whose params are plotted as a function of time separately
      
2. Higher order correlation analysis (see the draft for more details)
   
   - Analyzing trees (called H0 objects)
   - Analayzing loops (called H1 objects)
   - Analyzis of spheres (called H2 objects in plots)
  
   - Birth and death rate for each of the objects (trees, loops, spheres)
   - Number of objects for each data set
   - Number of objects as a function of time
   - Studying specific objects within a group:
       - H0 includes all possible trees: pair-wise correlations, 3-wise, 4-wise,, n -wise (and they do not form a loop)
       - H1 includes all possible loops: loops of 3 nodes, 4 nodes, etc
       - H2 includes all possible spheres: pyramid objects etc 
