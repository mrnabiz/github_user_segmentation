import random
import pandas as pd
from plotly import graph_objs as go
import plotly.express as px
from dash import dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the data for ML
transformed_df = pd.read_csv('../data/processed/transformed_df.csv.zip',
                          index_col=0,
                          compression='zip')
col_list_dropdown = transformed_df.columns.to_list()

# Read the data for Sankey
combined_df = pd.read_csv('../data/processed/combined_sankey.csv.zip',
                          index_col=0,
                          compression='zip')
sankey_col_list = combined_df['type'].unique().tolist()
sankey_col_list.remove('Start')

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

# Widgets for ML
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

# Widgets for Sankey
event_selector_sankey = dcc.Dropdown(
    id="event_filter_sankey",
    placeholder="Event Types",
    options=[
        {"label": i, "value": i} for i in sankey_col_list
    ],
    clearable=True,
    searchable=True,
    multi=True,
    value=sankey_col_list
    )

step_selector = dcc.Dropdown(
    id="n_steps",
    placeholder="Number of Steps",
    options=[
        {"label": i, "value": i} for i in range(3, 11)
    ],
    clearable=False,
    value=5
    )

depth_selector = dcc.Dropdown(
    id="events_per_step",
    placeholder="Number of Events/Step",
    options=[
        {"label": i, "value": i} for i in range(2, 16)
    ],
    clearable=False,
    value=5
    )


app.layout = dbc.Container([
    html.H1("🤖 GitHub User Segmentation and Behavior Dashboard", style={'textAlign': 'start'}),
    html.P("A dashboard to take a deeper look into GitHub users' behavior on March 17, 2023, Why only March 17th? Because there is over 4M recorded events on only one day!😬", 
           style={'textAlign': 'start'}),
    html.Hr(),
    html.H5("User Segmentation", style={'textAlign': 'start'}),
    dbc.Row([
        dbc.Col([html.P('Select the Events:'),
                 event_selector], width=9),
        dbc.Col([html.P('Number of Clusters:'),
                 component_selector], width=3)]),
    html.Hr(),
    dbc.Card([
       dbc.CardHeader("🧶 Cluster Visualization"),
       dbc.CardBody([dbc.Spinner(dcc.Graph(id="cluster_viz"))]),
       ]),
    html.Br(),
    dbc.Card([
       dbc.CardHeader("🗿 PCA Visualization"),
       dbc.CardBody([dbc.Spinner(dcc.Graph(id="pca-plot"))]),
       ]),
    html.Br(),
    html.H5("User Behavior Analysis", style={'textAlign': 'start'}),
    dbc.Row([
        dbc.Col([html.P('Select the Events for Behavior Diagram:'),
                 event_selector_sankey], width=8),
        dbc.Col([html.P('Number of Steps:'),
                 step_selector], width=2),
        dbc.Col([html.P('Number of Events/Step:'),
                 depth_selector], width=2)]),
    html.Hr(),
    dbc.Card([
       dbc.CardHeader("🏄 Sankey User Behavior Analysis"),
       dbc.CardBody([dbc.Spinner(dcc.Graph(id="sankey-plot"))]),
       ]),
    html.Br(),
    html.P(
    [
        "Made with Love at UBC MDS 💜 - 2023  By: ",
        html.A("Mohammad Reza Nabizadeh", href = "https://nabi.me")
    ], style={'textAlign': 'start'})
                    
])

# Set up callbacks/backend
@app.callback(
        [Output("cluster_viz", "figure"),
         Output("pca-plot", "figure"),
         Output("sankey-plot", "figure")],
        [Input('event_filter', 'value'),
         Input('n_component', 'value'),
         Input('event_filter_sankey', 'value'),
         Input('n_steps', 'value'),
         Input('events_per_step', 'value')])

def update_charts(event_filter, n_component, event_filter_sankey, n_steps, events_per_step):

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
        xaxis = {'side': 'top',  'tickangle':300},
        width=1080,
        height=600
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
            opacity=0.8
            ))
    
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title=c1),
            yaxis=dict(title=c2),
            zaxis=dict(title=c3),
        ),
        width=1080,
        height=600)
    
    fig_1 = go.Figure(data=[trace], layout=layout)
    fig_1.update_layout(autosize=True)


    # Sankey Plot
    
    # Required Function
    def filter_starting_step(x, starting_step, n_steps):
        starting_step_index = x.index(starting_step)
        
        return x[starting_step_index: starting_step_index + n_steps] 

    sank_col_list = event_filter_sankey.copy()
    sank_col_list.append('Start')
    sankey_df = combined_df[combined_df['type'].isin(sank_col_list)]
    starting_step = 'Start'
    valid_ids = sankey_df[sankey_df['type'] == starting_step]['actor_id'].unique()
    
    n_steps = n_steps

    # plan out the journey per user, with each step in a separate column
    flow = combined_df[(combined_df['actor_id'].isin(valid_ids))]\
        .groupby('actor_id')\
        .type.agg(list)\
        .to_frame()['type']\
        .apply(lambda x: filter_starting_step(x, starting_step, n_steps))\
        .to_frame()\
        ['type'].apply(pd.Series)

    flow = flow.fillna('End')

    for i, col in enumerate (flow.columns):
        flow[col] = 'Step {}: '.format(i+1) + flow[col].astype(str)
        
    events_per_step = events_per_step

    for col in flow.columns:
        all_events = flow[col].value_counts().index.tolist()
        all_events = [e for e in all_events if e!=(str(col+1) + ': End' )]
        top_events = all_events[:events_per_step]
        to_replace = list(set(all_events) - set(top_events))
        flow[col].replace(to_replace, [str(col+1) +': Other'] * len(to_replace), inplace=True)

    flow = flow.groupby(list(range(n_steps)))\
        .size()\
        .to_frame()\
        .rename({0: 'Count'}, axis=1)\
        .reset_index()
    
    label_list = []
    cat_cols = flow.columns[:-1].values.tolist()
    for cat_col in cat_cols:
        label_list_temp = list(set(flow[cat_col].values))
        label_list = label_list + label_list_temp

    colors_list = label_list.copy()
    for i, item in enumerate(label_list):
        if (item.__contains__('Start') | item.__contains__('End')):
            colors_list[i] = '#8C564B'
        elif item.__contains__('Push'):
            colors_list[i] = '#636EFA'
        elif item.__contains__('Create'):
            colors_list[i] = '#EF553B'
        elif item.__contains__('Pull Request'):
            colors_list[i] = '#00CC96'
        elif item.__contains__('Watch'):
            colors_list[i] = '#FFA15A'
        elif item.__contains__('Issue'):
            colors_list[i] = '#19D3F3'
        elif item.__contains__('Issue'):
            colors_list[i] = '#FECB52'
        elif item.__contains__('Other'):
            colors_list[i] = '#7F7F7F'
        else:
            colors_list[i] = '#7F7F7F'

    for i in range(len(cat_cols) - 1):
        if i == 0:
            source_target_df = flow[[cat_cols[i], cat_cols[i + 1], 'Count']]
            source_target_df.columns = ['Source', 'Target', 'Count']
        else:
            temp_df = flow[[cat_cols[i], cat_cols[i + 1], 'Count']]
            temp_df.columns = ['Source', 'Target', 'Count']
            source_target_df = pd.concat([source_target_df, temp_df])
            source_target_df = source_target_df.groupby(['Source', 'Target']).agg({'Count': 'sum'}).reset_index()

    # add index for source-target pair
    source_target_df['Source_id'] = source_target_df['Source'].apply(lambda x: label_list.index(x))
    source_target_df['Target_id'] = source_target_df['Target'].apply(lambda x: label_list.index(x))

    # filter out the end step
    source_target_df = source_target_df[(~source_target_df['Source'].str.contains('End')) &
                                        (~source_target_df['Target'].str.contains('End'))]

    fig_3 = go.Figure(data=[go.Sankey(
    node = dict(
        pad=20,
        thickness=20,
        color=colors_list,
        line=dict(
            color="black",
            width=0.5),
        label=label_list),
    link = dict(
        source=source_target_df['Source_id'].values.tolist(),
        target=source_target_df['Target_id'].values.tolist(),
        value=source_target_df['Count'].astype(int).values.tolist(),
        hoverlabel=dict(
            bgcolor='#C2C4C7')
            ))])

    fig_3.update_layout(dict(
        height=600,
        width=1080,
        margin=dict(t=30, l=0, r=0, b=30),
        font=dict(size=10),
        autosize=True
        ))


    return fig_1, fig_2, fig_3


if __name__ == '__main__':
    app.run_server(debug=True)