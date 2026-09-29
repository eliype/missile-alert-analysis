import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from networkx.algorithms.community import girvan_newman
from pathlib import Path
import csv

def percentage_difference(csv_name):

    df = pd.read_csv(csv_name)
    df['datetime'] = pd.to_datetime(df['alertDate'], errors='coerce')
    df = df.dropna(subset=['datetime', 'data'])

    ew_type = 'בדקות הקרובות צפויות להתקבל התרעות באזורך'
    rocket_type = 'ירי רקטות וטילים'

    df_filtered = df[df['category_desc'].isin([ew_type, rocket_type])].copy()
    df_filtered = df_filtered.sort_values(['data', 'datetime'])

    total_ew = 0
    followed_by_rocket = 0
    window = pd.Timedelta(minutes=30)

    for city, group in df_filtered.groupby('data'):
        ew_times = group[group['category_desc'] == ew_type]['datetime'].tolist()
        rocket_times = group[group['category_desc'] == rocket_type]['datetime'].tolist()

        for ew_t in ew_times:
            total_ew += 1
            has_rocket = any((ew_t <= r_t <= ew_t + window) for r_t in rocket_times)
            if has_rocket:
                followed_by_rocket += 1

    not_followed = total_ew - followed_by_rocket
    percent_not_followed = (not_followed / total_ew) * 100 if total_ew > 0 else 0

    print(f"Total Early Warnings: {total_ew}")
    print(f"Followed by Rocket Alert (within 30m): {followed_by_rocket}")
    print(f"NOT Followed by Rocket Alert: {not_followed}")
    print(f"Percentage NOT Followed: {percent_not_followed:.2f}%")

def community_detection(csv_name,category,threshold,folder_name):

    # 1. LOAD DATA
    df = pd.read_csv(csv_name)
    df['datetime'] = pd.to_datetime(df['alertDate'], errors='coerce')
    df = df.dropna(subset=['datetime', 'data'])
    alarm_types = [category]
    df_alarms = df[df['category_desc'].isin(alarm_types)].copy()
    df_alarms = df_alarms.sort_values('datetime')

    # 2. BUILD GRAPH — 2-minute sliding window co-occurrence
    records = df_alarms[['datetime', 'data']].to_dict('records')
    edge_weights = {}
    window = []

    for row in records:
        while window and (row['datetime'] - window[0]['datetime']).total_seconds() > 120:
            window.pop(0)
        for prev_row in window:
            if row['data'] != prev_row['data']:
                u, v = sorted([row['data'], prev_row['data']])
                edge_weights[(u, v)] = edge_weights.get((u, v), 0) + 1
        window.append(row)


    G = nx.Graph()
    for (u, v), w in edge_weights.items():
        if w > threshold:
            G.add_edge(u, v, weight=w)

    print(f"Graph constructed with threshold > {threshold}")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(G)}")

    # 3. COMMUNITY DETECTION (Girvan-Newman, first split)
    comp = girvan_newman(G)
    first_level_communities = next(comp)
    communities_list = list(first_level_communities)

    print("\n--- Communities Detected ---")
    for i, c in enumerate(communities_list):
        print(f"Community {i+1} ({len(c)} cities): {list(c)[:10]}")

    # FIX: write community id back onto each node (was missing before)
    for i, community in enumerate(communities_list):
        for node in community:
            G.nodes[node]['community'] = i

    palette = ['#E1204C', '#43B05C', '#3B82F6', '#F59E0B', '#8B5CF6', '#14B8A6']

    # 4 NODE-LINK GRAPH — hover-only labels, no permanent text clutter
    pos = nx.spring_layout(G, seed=42, weight='weight', k=0.3)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.4, color='#ccc'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text, node_colors = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        comm_id = G.nodes[node].get('community', 0)
        deg = G.degree(node, weight='weight')
        node_text.append(f"<b>{node}</b><br>Community: {comm_id+1}<br>Weighted degree: {deg}")
        node_colors.append(palette[comm_id % len(palette)])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(color=node_colors, size=7, line=dict(width=0.5, color='DarkSlateGrey'))
    )

    fig_graph = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Alarm Co-occurrence Graph (hover for city names)',
            title_font_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    fig_graph.show()

    # 5 COMMUNITY SUMMARY — the headline figure
    comm_sizes = {i: len(c) for i, c in enumerate(communities_list)}
    cross_edges = sum(
        1 for u, v in G.edges()
        if G.nodes[u].get('community') != G.nodes[v].get('community')
    )
    bridge_nodes = [
        n for n in G.nodes()
        if len({G.nodes[nb].get('community') for nb in G.neighbors(n)}) > 1
    ]

    n_comm = len(communities_list)
    xs = list(range(n_comm))
    fig_summary = go.Figure(go.Scatter(
        x=xs, y=[0] * n_comm,
        mode='markers+text',
        marker=dict(
            size=[max(30, comm_sizes[i] ** 0.5 * 8) for i in xs],
            color=[palette[i % len(palette)] for i in xs]
        ),
        text=[f"Community {i+1}<br>{comm_sizes[i]} cities" for i in xs],
        textposition='bottom center'
    ))
    fig_summary.add_annotation(
        x=sum(xs) / len(xs), y=0.15,
        text=f"{cross_edges} cross-community edge(s) via {len(bridge_nodes)} bridge node(s)",
        showarrow=False, font=dict(size=13)
    )
    fig_summary.update_layout(
        title='Community Summary',
        xaxis=dict(visible=False, range=[-1, n_comm]),
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        plot_bgcolor='white',
        height=350
    )
    fig_summary.show()

    # 4C. ADJACENCY HEATMAP — sorted by community, scales far better
    #     than a node-link plot for hundreds of nodes
    order = [n for c in communities_list for n in c]
    A = nx.to_numpy_array(G, nodelist=order, weight='weight')

    fig_heat = go.Figure(go.Heatmap(
        z=A, x=order, y=order,
        colorscale='Reds',
        hovertemplate='%{y} ↔ %{x}<br>weight: %{z}<extra></extra>'
    ))
    fig_heat.update_layout(
        title='Adjacency Matrix (rows/cols sorted by community)',
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
        height=650, width=650
    )
    fig_heat.show()

    # 6 TOP-DEGREE LABELS — labeled hub cities only, on the full layout
    top_nodes = sorted(G.degree(weight='weight'), key=lambda x: -x[1])[:15]
    top_names = {n for n, _ in top_nodes}

    label_x, label_y, label_text = [], [], []
    for n in top_names:
        x, y = pos[n]
        label_x.append(x)
        label_y.append(y)
        label_text.append(str(n))

    fig_labeled = go.Figure(
        data=[
            edge_trace,
            node_trace,
            go.Scatter(
                x=label_x, y=label_y, mode='text',
                text=label_text, textposition='top center',
                textfont=dict(size=10, color='black'),
                hoverinfo='none'
            )
        ],
        layout=go.Layout(
            title='Graph with Top-15 Hub Cities Labeled',
            title_font_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    # fig_labeled.show()

    output_dir = Path(folder_name)
    output_dir.mkdir(parents=True, exist_ok=True)


    fig_graph.write_html(output_dir / "graph_full.html")
    fig_summary.write_html(output_dir /"community_summary.html")
    fig_heat.write_html(output_dir /"adjacency_heatmap.html")
    fig_labeled.write_html(output_dir /"graph_labeled_hubs.html")



    with open(output_dir / 'names_and_arrays.csv', mode='w', newline='',encoding='utf-8-sig') as file:

        writer = csv.writer(file)

        # Write the column headers first
        writer.writerow(['שם עיר', 'קהילה'])

        for i, c in enumerate(communities_list):

            for site in c:

                writer.writerow([site, i+1])

    print("CSV created successfully!")



community_detection("second_war.csv", 'ירי רקטות וטילים', 70,"missiles - second war")
# community_detection("second_war.csv", 'בדקות הקרובות צפויות להתקבל התרעות באזורך', 150,"warning - second war")
percentage_difference("second_war.csv")

