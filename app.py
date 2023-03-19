import pandas as pd
import random
import gzip
from plotly import graph_objs as go
from plotly.graph_objs import *
from dash import dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the data
input_file = 'data/processed/transformed_df.h5'
key = 'data'
transformed_df = pd.read_hdf(input_file, key)

#transformed_df = pd.read_csv('data/processed/transformed_df.csv', index_col=0)
col_list_dropdown = transformed_df.columns.to_list()

# Styles and bootstraping
ext_styl = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.google.com/specimen/Roboto"
    ]

SIDEBAR_STYLE = {
    "top": 0,
    "left": 0,
    "bottom": 0,
    "padding": "1rem 1rem",
}

CONTENT_STYLE = {
    "padding": "0.3rem 0.3rem",
}

# App configuration and the bootstrap setup
app = dash.Dash(
    __name__,
    external_stylesheets = [dbc.themes.FLATLY]
)

# Run the server
server = app.server

# Set the dashboard title
app.title = 'GitHub User Segmentation'

# Widgets
event_selector = dcc.Dropdown(
    id="event_filter",
    placeholder="Event Types",
    options=[
        {"label": i, "value": i} for i in transformed_df.columns.to_list()
    ],
    clearable=True,
    searchable=True,
    multi=True,
    value=col_list_dropdown
    )

component_selector = dcc.Dropdown(
    id="n_component",
    placeholder="Number of Segments",
    options=[
        {"label": i, "value": i} for i in range(5, 15)
    ],
    clearable=False,
    value=10
    )


app.layout = dbc.Container([
    html.H1("🤖 GitHub User Segmentation", style={'textAlign': 'start'}),
    html.P("A dashboard to take a deeper look into GitHub users' behavior on March 17, 2023", 
           style={'textAlign': 'start'}),
    html.Hr(),
    dbc.Row([
        dbc.Col([html.P('Select the Events:'),
                 event_selector], width=9),
        dbc.Col([html.P('Select the Number of Clusters:'),
                 component_selector], width=3)]),
    html.Hr(),
    dbc.Card([
       dbc.CardHeader("Cluster Visualization"),
       dbc.CardBody([dbc.Spinner(dcc.Graph(id="cluster_viz"))]),
       ]),
    html.Br(),
    dbc.Card([
       dbc.CardHeader("PCA Visualization"),
       dbc.CardBody([dbc.Spinner(dcc.Graph(id="pca-plot"))]),
       ]),
    html.Br(),
    html.P("Build by Mohammad Reza Nabizadeh - UBC MDS - 2023", 
           style={'textAlign': 'start'})
                    
])

# Set up callbacks/backend
@app.callback(
        [Output("cluster_viz", "figure"),
         Output("pca-plot", "figure")],
        [Input('event_filter', 'value'),
         Input('n_component', 'value')]
)


def update_charts(event_filter, n_component):

    cols_list = transformed_df.columns.to_list()

    if len(event_filter) < 3:
        subtract_list = [x for x in cols_list if x not in event_filter]
        random_index = random.randint(0, len(subtract_list) - 1)
        random_item = subtract_list[random_index]
        event_filter.append(random_item)
        filter_col_list = event_filter
    else:
        filter_col_list = event_filter

    subset = transformed_df[filter_col_list]

    # Using Principal Component Analysis (PCA) for dimensionality reduction
    pca = PCA(n_components=3)
    pca.fit(subset)
    z_pca = pca.transform(subset)
    pca_df = pd.DataFrame(z_pca,
                      columns=["Z1", "Z2", "Z3"],
                      index=subset.index)
    W = pca.components_
    component_labels = ["PC" + str(i + 1) for i in range(3)]
    weights_labels = pd.DataFrame(W, columns=subset.columns)
    weights_labels = weights_labels.abs()
    top2_cols = weights_labels.apply(lambda x: x.nlargest(2).index.tolist(), axis=1)
    Z1 = ' + '.join(top2_cols[0])
    Z2 = ' + '.join(top2_cols[1])
    Z3 = ' + '.join(top2_cols[2])
    
    pca_df = pca_df.rename(columns={'Z1':Z1,
                                    'Z2':Z2,
                                    'Z3':Z3})
    
    # Plot Principal Component Analysis

    fig_2 = px.imshow(
        W,
        y=component_labels,
        x=subset.columns.to_list(),
        color_continuous_scale="viridis",
        )
    
    fig_2.update_layout(
        xaxis_title="Features",
        yaxis_title="Principal Components",
        xaxis = {'side': 'top',  'tickangle':300}
        )
    fig_2.update_layout(autosize=True)

    # Run K-means model
    k = n_component
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(subset)
    labels = kmeans.labels_
    pca_df["label"] = pd.DataFrame(labels)[0]
    
    # Clean PCA outliers for visualizations
    col_names_pca = pca_df.columns.to_list()
    col_names_pca.remove('label')
    for col in col_names_pca:
        q1 = pca_df[col].quantile(0.05)
        q2 = pca_df[col].quantile(0.95)
        non_out_pca_df = pca_df.loc[(pca_df[col] >= q1) &
                                    (pca_df[col] <= q2)]
        
    
    cols_list = non_out_pca_df.columns.to_list()
    c1 = cols_list[0]
    c2 = cols_list[1]
    c3 = cols_list[2]
    c4 = cols_list[3]

    x = non_out_pca_df[c1]
    y = non_out_pca_df[c2]
    z = non_out_pca_df[c3]
    label = non_out_pca_df[c4]
    
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=label,
            colorscale='turbo_r',
            opacity=0.5
            ))
    
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title=c1),
            yaxis=dict(title=c2),
            zaxis=dict(title=c3),
        ),
        width=900,
        height=600)
    
    fig_1 = go.Figure(data=[trace], layout=layout)
    fig_1.update_layout(autosize=True)

    return fig_1, fig_2


if __name__ == '__main__':
    app.run_server(debug=True)