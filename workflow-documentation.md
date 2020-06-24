## MultiNet Toolkit Dashboard User Workflow Documentation

### Pull data from Web of Science (WoS) in one of the following export format: 

  Full Record and Cited References' records in Tab-delimited (Mac, UTF-8)
  
  Full Record and Cited References' records in Tab-delimited (Win, UTF-8)
  

The text box in the top right will populate with current directory path, and allows the user to load data from a folder or file path of their choosing.

Once the 'Get data' button is pressed, user file(s) are loaded, and the data is preprocessed.  

Be patient, the network may take a few moments to parse.    


In addition to the visualization output, an initial table displaying an edgelist will populate at the bottom of the page.  It is anticipated that there may be many records, so this feature is set for vertical scrolling so as to not clutter up the page.


After the data is loaded, and preprocessing is complete, navigate to the metrics tab to create co-authorship and co-citation network analyses.

Note: If co-citation is the network chosen, in addition to the co-citation edgelist table output, a paper citation report will generate in table format as well.  This is a network of all records in the dataset with each record's citations mapped to said record.  This network is not available for visualization, only in table format and available for export.


Color clustering by node is mapped for Girvan-Newman and K-core subsets. 

The Girvan-Newman table will list nodes with numerical cluster assignment.

The K-core table will list the nodes that satify the minimum number of edges to be included in a k-core cluster.  Currently K = 3.


Node size is correlated to weighted degree.  The larger the node, the higher the weighted-degree.

Hovering over a node will provide the node's identity (author name or doi reference number), and number of node adjacencies aka simple degree.


To observe centrality values for a single node, the user can choose from the following centralities: betweenness, weighted degree, closeness. The top 10 actors will be generated in the table at the bottom of the page.

All analyses is available for export. An export button is visible at the top of each table generated.  Exports will download in CSV format.






