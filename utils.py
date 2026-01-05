import networkx as nx
import pandas as pd
import numpy as np

def graph_to_dataframe(graph: nx.Graph) -> pd.DataFrame:
    """
    PURE FUNCTION
    """
    node_ids = []
    neighbor_lists = []
    attribute_data = {}   # To store attributes as lists

    # Collect all unique attribute keys first (if attributes are not uniform)
    first_node_id = next(iter(graph.nodes))
    first_node_attrs = graph.nodes[first_node_id]
    attribute_keys = list(first_node_attrs.keys())

    for node_id, attributes in graph.nodes(data=True):
        node_ids.append(node_id)
        neighbor_lists.append(list(graph.neighbors(node_id)))

        for key in attribute_keys:
            attribute_data.setdefault(key, []).append(attributes.get(key, None))  # Use None for missing attributes

    df = pd.DataFrame({
        'node_id': node_ids,
        'neighbors': neighbor_lists,
        **attribute_data   # Unpack attribute data into DataFrame columns
    })

    df.set_index('node_id', inplace=True)
    return df



def df_slice_to_np(
    df: pd.DataFrame,
    row_indices: list | None = None,
    columns: list | None = None
) -> np.ndarray:
    """PURE FUNCTION
    INPUT: df with consumption items as columns names and row indices as node id
    OUTPUT: np.ndarray of the form [node1_consumption_list, node2_consumption_list, ...]
    """
    if columns is None:
        columns = df.columns
    if row_indices is None:
        row_indices = df.index

    return df[columns].loc[row_indices].to_numpy()