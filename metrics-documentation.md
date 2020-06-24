## Metrics Documentation

**Betweenness:** 
The dashboard implements [betweenness metric from NetworkX](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html).  For all pairs in a network, betweenness is calculated from the number of shortest paths that pass through a given node.  

**Closeness:**
The dashboard implements [closeness metric from NetworkX](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.closeness_centrality.html).  Closeness centrality is a node's average inverse distance to all other nodes in the network.

**Girvan-Newman algorithm:**
The dashboard implements the [Girvan-Newman algorithm from NetworkX](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html).  The algorithm finds the edges in a network that occur the most frequently amongst pairs of nodes.  It does this by finding the the edges with the highest betweenness value and recursively removing them.  What remains are the subset communities in the network. 

**K-Core algorithm:** 
With an undirected graph, the k-core is the subgraph of G where all nodes are adjacent to at least k nodes.  This creates subsets i.e. k-core clusters, where each node in each subset has at least k connections.

**Simple degree:** 
The sum of adjacencies a node has.

**Weighted degree:**
The dashboard implements [weighted degree metric from NetworkX](https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.algorithms.centrality.degree_centrality.html).  A node's weighted degree value is the percentage of nodes it has connections with.

### Dashboard framework

[Dash](https://dash.plotly.com/introduction) is a Python framework for constructing web based applications.  It is designed to accommodate interactive data analyses and visualization, and allows for the creation of a user interface to be developed in python.   
