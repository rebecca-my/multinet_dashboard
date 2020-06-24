from collections import defaultdict
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import datetime
from flask_caching import Cache
import os
from colour import Color
import pandas as pd
from time import time
import uuid
import numpy as np
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
import json
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from flask import send_file
import re
from scipy.sparse import csr_matrix

external_stylesheets = [
    # Dash CSS
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    # Loading screen CSS
    'https://codepen.io/chriddyp/pen/brPBPO.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=False)

cache = Cache(app.server, config={
    'CACHE_TYPE': 'redis',
    # Note that filesystem cache doesn't work on systems with ephemeral
    # filesystems like Heroku.
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    # should be equal to maximum number of users on the app at a single time
    # higher numbers will store more data in the filesystem / redis cache
    'CACHE_THRESHOLD': 200
})
def read_inputs(file_path):
    if os.path.isdir(file_path):
        dfs = []
        for name in os.listdir(file_path):
            filename = os.path.join(file_path, name)
            text = []
            with open(filename, encoding='latin1') as f:
                for line in f.readlines():
                    text.append(line.replace('\x00','').replace('\n', '').split('\t'))
        
            dfs.append(pd.DataFrame(data=text[1:-1], columns=text[0]+['mystery']))
        df = pd.concat(dfs).reset_index()
        del dfs
    else:
        text = []
        with open(file_path, encoding='latin1') as f:
            for line in f.readlines():
                text.append(line.replace('\x00','').replace('\n', '').split('\t'))
    
        df = pd.DataFrame(data=text[1:-1], columns=text[0]+['mystery'])

    #df.to_csv('some_file_name', index=False)
    df.drop('mystery', axis=1, inplace=True)
    return df

def get_pairs(session_id, file_name, mode):
    @cache.memoize()
    def load_preprocess_co_auth_user_data(session_id, file_name):
        df = read_inputs(file_name)
        authors = list(set(df['AU'].str.split('; ').sum()))
        author_id = dict([(name, i) for i, name in enumerate(authors)])
        cstr_rows = df['AU'].str.split('; ').apply(lambda x: sorted(map(author_id.get, x))).sum()
        cstr_columns = (pd.Series(df.index).map(lambda x: [x]) * df['AU'].str.split('; ').apply(len)).sum()


        row = np.array(cstr_rows)
        col = np.array(cstr_columns)
        data = np.ones(len(row))
        author_matrix = csr_matrix((data, (row, col)), shape=(len(authors), len(df)))

        pairs = []
        for auth_1 in range(len(authors) - 1):
            for auth_2 in range(auth_1 + 1, len(authors)):
                auth_1_data = author_matrix.indices[author_matrix.indptr[auth_1]:author_matrix.indptr[auth_1+1]]
                auth_2_data = author_matrix.indices[author_matrix.indptr[auth_2]:author_matrix.indptr[auth_2+1]]
                weight = len(set(auth_1_data).intersection(set(auth_2_data)))
                if weight > 0:
                    pairs.append((authors[auth_1], authors[auth_2], weight))
        data = {'edges' : pairs, 
                'node_list' : authors,
                'reverse_lookup' : author_id}
        return json.dumps(data)

    @cache.memoize()
    def load_preprocess_citation_user_data(session_id, file_name):
        df = read_inputs(file_name)
        # citation preprocessing
        cited_ref = list(set(df['CR'].str.split('; ').sum()))
        ref_index = dict([(name, i) for i, name in enumerate(cited_ref)])
        doi_ref = df['CR'].str.findall('DOI\s+\S+;?').str.join(',').str.replace(';', '').str.replace('DOI\s+', '').str.replace('[', '').str.replace(']', '').str.split(',')

        def get_non_doi(cites):
            # replace x.strip for better parsing.
            # this just strips out the extra white space, and uses raw citation as is.  this can be turned into something more matchable in the future.
            return [x.strip() for x in cites.split(';') if re.match(r'^((?!DOI).)*$', x)]

        non_doi_ref = df['CR'].apply(get_non_doi)
        all_ref = doi_ref + non_doi_ref
        papers = list(set(all_ref.sum()))
        papers_id = dict([(name, i) for i, name in enumerate(papers)])

        co_refs = defaultdict(int)
        for cites in all_ref:
            for a, b in zip(cites[1:], cites[:-1]):
                #crude filtering for weird non_doi references
                if len(a) <= 3 or len(b) <= 3:
                    continue
                co_refs[(a, b)] += 1

        ref_pairs = [(k[0], k[1], v) for (k, v) in co_refs.items() if v > 1]

        if len(ref_pairs) == 0:
            raise ValueError('No co-citations in the input')

        data = {'edges' : ref_pairs, 
                'node_list' : papers,
                'reverse_lookup' : papers_id,
                'doi_list' : list(df['DI']),
                'all_ref_list' : list(all_ref)}
        return json.dumps(data)

    if mode == 'Co-authorship':
        return load_preprocess_co_auth_user_data(session_id, file_name)
    elif mode == 'Co-citation':
        return load_preprocess_citation_user_data(session_id, file_name)
    else:
        raise ValueError('Bad mode')


def adjacencies(G):
    node_adjacencies = []
    node_text = []
    for (adjacencies, node) in zip(G.adjacency(), G.nodes):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(str(node) + ' # of connections: ' + str(len(adjacencies[1])))
    return node_adjacencies, node_text

def generate_reports(G):
    node_adjacencies,_ = adjacencies(G)
    simple_degree = dict(zip(G.nodes, node_adjacencies))
    betweenness = nx.betweenness_centrality(G)
    degree = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G)
    cent_df = pd.DataFrame.from_dict([betweenness, degree, closeness, simple_degree])
    cent_df = pd.DataFrame.transpose(cent_df)
    cent_df.columns = ['betweenness', 'degree', 'closeness', 'simple_degree']
    cent_df['node'] = cent_df.index
    cent_df = cent_df[['node', 'betweenness', 'degree', 'closeness', 'simple_degree']]
    return cent_df


def calc_graph(points, min_edges=1):
    data = json.loads(points)
    w_subset = [edge for edge in data['edges'] if edge[2] > min_edges]
    G = nx.Graph()
    G.add_weighted_edges_from(w_subset)
    return G

# def calc_filtered_graph(points):
#     dbl_pairs = np.array(pairs + [(edge[1], edge[0], edge[2]) for edge in pairs])
#     data = dbl_pairs[:,2]
#     row = dbl_pairs[:,0]
#     col = dbl_pairs[:,1]
#     coauth_matrix = csr_matrix((data, (row, col)), shape=(len(authors), len(authors)))
#     filter_pairs = []
#     for auth in range(len(authors)):
#         auth_col = coauth_matrix.indices[coauth_matrix.indptr[auth]:coauth_matrix.indptr[auth+1]]
#         auth_weights = coauth_matrix.data[coauth_matrix.indptr[auth]:coauth_matrix.indptr[auth+1]]
#         if sum(auth_weights) > 5:
#             filter_pairs += [(auth, other_auth, weight) for other_auth, weight in zip(auth_col, auth_weights)]  
#     G = nx.Graph()
#     G.add_weighted_edges_from(filter_pairs)
#     return G

def k_cores(G, k):
    k_core = G.copy()
    updated = True
    while updated:
        updated = False
        # why did we use copy here?  refer 6/10 lesson
        for node in list(k_core.nodes).copy():
            if len(k_core[node]) < k:
                k_core.remove_node(node)
                updated = True
    return k_core

def community_detection(G, number_of_clusters):
    girvan_mod = girvan_newman(G)
    for girvan_tuple in girvan_mod:
        if len(girvan_tuple) >= number_of_clusters:
            return girvan_tuple  
    return girvan_tuple

def rebuild_G(G_json):
    # this goes from json back to G
    data = json.loads(G_json)
    G = nx.Graph()
    # data['links'] is a dictionary that has source, target, and weight in it.  a graph is built by handing it edges in the 
    # order source, target, weight.  From reading the dictionary for the items one at a time, and as a tuple.  
    edges = [(e['source'], e['target'], e['weight']) for e in data['links']]
    G.add_weighted_edges_from(edges)
    return G

def visualize_graph(G, mode, coloring_tuple):
    start_time = time()
    print('start processing')

    pos=nx.drawing.layout.spring_layout(G)

    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])

    traceRecode = []

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        weight = 1
        trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                           mode='lines',
                           line={'width': weight},
                           marker=dict(color='black'),
                           opacity=1)
        traceRecode.append(trace)
        index = index + 1

    node_x = []
    node_y = []

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',

        marker=dict(
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=3,
            line_width=1))

    node_adjacencies, node_text = adjacencies(G)
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    colors = list(Color('lightcoral').range_to(Color('darkblue'), len(coloring_tuple)))
    colored = {}
    for color, cluster in zip(colors, coloring_tuple):
        for node in cluster:
            colored[node] = 'rgb' + str(color.rgb)

    node_trace.marker.color = [colored[node] for node in G.nodes]

    deg_cent = nx.degree_centrality(G)
    node_size =  [np.sqrt(v)*125 for v in deg_cent.values()]
    node_trace.marker.size = node_size

    traceRecode.append(node_trace)

    figure = {
    "data": traceRecode,
    "layout": go.Layout(showlegend=False, hovermode='closest',
                        margin={'b': 30, 'l': 20, 'r': 20, 't': 30},
                        xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                        yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                        height=600,
                        width=700,
                        clickmode='event+select',
                        annotations=[
                            dict(
                                ax=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 20,
                                ay=(G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 40, 
                                x=(G.nodes[edge[1]]['pos'][0] + G.nodes[edge[0]]['pos'][0]) / 20,
                                y=(G.nodes[edge[1]]['pos'][1] + G.nodes[edge[0]]['pos'][1]) / 40, 
                                opacity=1
                            ) for edge in G.edges]
                        )}
    print('finishing time: ', time() - start_time),
    return figure

def generate_table(dataframe,  max_rows=100):
    return  dash_table.DataTable(
            columns=[{'name' : col, 'id' : col} for col in dataframe.columns],
            data=[dict(row) for (i, row) in dataframe.iterrows()],
            export_format='xlsx',
            export_headers='display',
            merge_duplicate_headers=True,
            #style_cell={'whitespace' : 'normal', 'height' : 'auto'},
            style_cell={'overflow' : 'hidden', 'textOverflow' : 'ellipsis', 'maxWidth' : 0},
            style_table={'overflowY' : 'auto', 'height' : '300px'}
            )


def serve_layout():
    session_id = str(uuid.uuid4())

    return html.Div([
        html.Div(session_id, id='session-id', style={'display': 'none'}),
        html.Div(
            html.H3(
                className='eight columns',
                children='MultiNet Toolkit',
                style={'textAlign': 'center', 'color': 'black'}
                )
            ),
        html.Div(
            className='four columns',
            children=[
                html.Button('Get data', id='get-data-button'),
                dcc.Input(id='datafile', value=os.getcwd(), type='text', style={'size':50})],
                style={'float': 'right'}
            ),
        html.Div(
            className='seven columns',
            children=[html.H4('Filter / Explore node data',
                style={'textAlign':'left', 'text-decoration':'underline', 'color':'black'}),
                dcc.Markdown(""" Node size indicates number of connections for an author or citation reference.
                    Community grouping shown by node color implements Girvan Newman algorithm, or K-core clustering.
                    """),
                html.Div(id='graph')]
                ),
        html.Div(
            className='four columns',
            children=[
                dcc.Tabs(id='tabs_with_classes', value='tab_1', parent_className='custom_tabs', className='custom_tabs_container',
                    children=[
                        dcc.Tab(label='Workflow', value='tab_1', className='custom_tab', selected_className='custom_tab__selected',
                            children=[
                                html.A('GitHub', href='https://github.iu.edu/rsciagli/MultiNet_Toolkit', target='_blank')
                            ]),
                        dcc.Tab(label='About', value='tab_2', className='custom_tab', selected_className='custom_tab__selected',
                            children=[html.Div(
                                dcc.Markdown("""
                                    The MultiNet Toolkit is an interactive dashboard built for [IU Network Science Institute](https://iuni.iu.edu/). 

                                    It is a network science tool that can be shared amongst scientific organizations, and scholars.  

                                    The goal of the toolkit is for users with no prior knowledge of how network science is done, can collect and analyze network data centered around scientific collaboration, and be able to asses change over time.

                                    The MultiNet Toolkit provides descriptive analyses on co-authorship and citation networks.  

                                    Observe who in the analyses is working together, what the citation practices look like, and subset activity within communities. 
                                """),
                                    style={'textAlign':'left', 'font-size':'18px', 'padding-top':20, 'padding-bottom':20,
                                        'li':'before'})
                            ]),
                        dcc.Tab(label='Metrics', value='tab_3', className='custom_tab', selected_className='custom_tab__selected',
                                children=[html.Div(None),
                                html.Br(),
                                html.Div(
                                    children=[html.Div('Networks:',
                                        style={'textAlign':'center', 'background-color':'lightgrey', 'color':'black',
                                                'font-size':'18px', 'padding-top':10, 'padding-bottom':10}),
                                        dcc.RadioItems(
                                            id='network_buttons',
                                            options=[
                                                {'label': i, 'value': i} for i in ['Co-authorship', 'Co-citation']],
                                                value='Co-authorship',
                                                labelStyle = {'display':'inline-block', 'margin-right':10, 'padding-top':30, 'padding-bottom':30}
                                        )
                                    ]),
                                html.Div(
                                    children=[html.Div('Community Detection:',
                                        style={'textAlign':'center', 'background-color':'lightgrey', 'color':'black',
                                                'font-size':'18px', 'padding-top':10, 'padding-bottom':10}),
                                        dcc.RadioItems(
                                            id='modularity_buttons',
                                            options=[
                                                {'label': i, 'value': i} for i in ['Girvan-Newman', 'K-core']],
                                                value='Girvan-Newman',
                                                labelStyle = {'display':'inline-block', 'margin-right':10, 'padding-top':30, 'padding-bottom':30}
                                        )]
                                    ),
                                html.Div(
                                    children=[html.Div('Centrality Metrics',
                                        style={'textAlign':'center', 'background-color':'lightgrey', 'color':'black',
                                                'font-size':'18px', 'padding-top':10, 'padding-bottom':10}),
                                        dcc.Dropdown(id='centrality_type',
                                            options=[
                                                {'label': 'Betweenness', 'value': 'betweenness'},
                                                {'label': 'Weighted degree', 'value': 'degree'},
                                                {'label': 'Closeness', 'value': 'closeness'}
                                            ],
                                            value=None
                                        )
                                ])
                            ]) 
                        ])
                ]
            ),
        html.Div(id='tabs_content_classes'),
        html.Div(id='modularity_content'),
        html.Div(
            className='twelve columns', id='simple_degree'),
        html.Div(
            className='twelve columns', id='table'),
        html.Div(
            className='twelve columns', id='papers_network'),
        html.Div(
            className='twelve columns', id='author_edgelist'),
        html.Div(
            className='twelve columns', id='Girvan_table'),
        html.Div(id='G_holder', style={'display': 'none'}),
        html.Div(id='intermediate-value', style={'display': 'none'})
    ])

app.layout = serve_layout

@app.callback(Output('table', 'children'),
              [Input('G_holder', 'children'), Input('centrality_type', 'value')])
def make_table(G_json, centrality_type):
    # added first if to avoid error if user clicks on centralities and no data available.
    if G_json is not None:
        if centrality_type is not None:
            # rebuild G
            G = rebuild_G(G_json)
            # make the table
            df = generate_reports(G)
            df = df.sort_values(centrality_type, ascending = False)[:10]
            return generate_table(df)

@app.callback(Output('papers_network', 'children'),
              [Input('intermediate-value', 'children')])
def make_paper_table(parsed_inputs):
    if parsed_inputs is not None and parsed_inputs[1] is not None:
        return ['Paper citation network', generate_table(pd.read_json(parsed_inputs[1], orient='split'))]

@app.callback(Output('author_edgelist', 'children'),
              [Input('G_holder', 'children')])
def make_edgelist_table(G_json):
    if G_json is not None:
        G = rebuild_G(G_json)
        df = pd.DataFrame(G.edges, columns=['node_1', 'node_2'])
        return ['Edge list', generate_table(df)]


@app.callback(Output('Girvan_table', 'children'),
             [Input('G_holder', 'children'), Input('modularity_buttons', 'value')])
def make_Girvan_table(G_json, modularity_buttons):
    if G_json is not None:
        G = rebuild_G(G_json)
        if modularity_buttons == 'Girvan-Newman':
            girvan_tuple = community_detection(G, np.sqrt(len(G.nodes)))
            nodes = []
            clusters = []
            for i, cluster in enumerate(girvan_tuple):
                nodes += list(cluster)
                clusters += [i] * len(cluster)
            df = pd.DataFrame(data = zip(nodes, clusters), columns = ['node', 'cluster'])
            return ['Girvan-Newman', generate_table(df)]
        if modularity_buttons == 'K-core':
            G_core = k_cores(G, 3)
            df = pd.DataFrame(data = G_core.nodes, columns = ['k-core nodes'])
            return ['K-cores', generate_table(df)]

@app.callback(Output('G_holder', 'children'),
    [Input('intermediate-value', 'children')])
def store_G(parsed_inputs):
    if parsed_inputs is not None:
        # Return data in node-link format that is suitable for JSON serialization and use in Javascript documents.
        # the below two lines take G and encode it to json
        data = nx.readwrite.json_graph.node_link_data(calc_graph(parsed_inputs[0]))
        return json.dumps(data)


@app.callback(Output('tabs_content_classes', 'children'),
    [Input('tabs_with_classes', 'value')])
def render_tabs(tab):
    if tab == 'tab_1':
        return html.Div([
            ])
    elif tab == 'tab_2':
        return html.Div([
        ])
    elif tab == 'tab_3':
        return html.Div([
        ])

@app.callback(Output('graph', 'children'),
            [Input('G_holder', 'children'), Input('modularity_buttons', 'value')],
            [State('network_buttons', 'value')])
def update_output(G_holder, button, mode):
    if G_holder is not None:
        G = rebuild_G(G_holder)
        if button == 'Girvan-Newman':
            coloring_tuple = community_detection(G, np.sqrt(len(G.nodes)))
            return [dcc.Graph(figure=visualize_graph(G, mode, coloring_tuple))]
        if button == 'K-core':
            # k parameter = 3 here.
            G = k_cores(G, 3)
            coloring_tuple = community_detection(G, np.sqrt(len(G.nodes)))
            return [dcc.Graph(figure=visualize_graph(G, mode, coloring_tuple))]

@app.callback(Output('intermediate-value', 'children'),
              [Input('get-data-button', 'n_clicks'),
               Input('session-id', 'children'),
               Input('network_buttons', 'value')],
              [State('datafile', 'value')])
def backend_calc(num_clicks, session_id, mode, file_name):
    print(num_clicks)
    if num_clicks is not None:
        parsed_inputs = get_pairs(session_id, file_name, mode)
        data = json.loads(parsed_inputs)
        if 'doi_list' in data.keys():
            paper_citations = pd.DataFrame(data=zip(data['doi_list'], data['all_ref_list']), columns=['records', 'citations mapped to record']).to_json(orient='split')
        else:
            paper_citations = None

        return [parsed_inputs, paper_citations]


if __name__ == '__main__':
    app.run_server(debug=True)

