## MultiNet Toolkit Dashboard User Workflow Documentation

Pull data from Web of Science (WoS) in one of the following export format: 

  Full Record and Cited References' records in Tab-delimited (Mac, UTF-8)
  Full Record and Cited References' records in Tab-delimited (Win, UTF-8)

The text box in the top right will populate with current directory path, and allows the user to load data from a folder or file path of their choosing.

Once the 'Get data' button is pressed, the user's file(s) are loaded, and the data is preprocessed.  
Be patient, the network may take a few moments to parse.    

In addition to the visualization output, an initial table displaying an edgelist will populate at the bottom of the page.  It is anticipated that there may be many records, so this feature is set for vertical scrolling so as to not clutter up the page.

After the data is loaded, and preprocessing is complete, navigate to the metrics tab to create co-authorship and co-citation network analyses.

Color clustering is available for Girvan-Newman and K-core subsets.
  The dashboard implements the Girvan-Newman algorithm from NetworkX.  The algorithm finds the edges in a network that occur the most frequently amongst pairs of nodes.  It does this by finding the the edges with the highest betweenness value and recursively removing them.  What remains are the subset communities in the network.  

Use the Centralities dropdown to observe betweenness, weighted degree, and closeness centralities. The top 10 actors will be generated in the table at the bottom of the page.

All tables are available for export. Click the export button and for a CSV download of the chosen analysis.

## Documentation


