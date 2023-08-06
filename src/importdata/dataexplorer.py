import networkx as nx
import obonet
import pandas as pd


class GoDataExplorer:
    def __init__(self, path):
        self.graph = obonet.read_obo(path)
        self.id_to_name = {id_: data.get('name') for id_, data in self.graph.nodes(data=True)}
        self.name_to_id = {data['name']: id_ for id_, data in self.graph.nodes(data=True) if 'name' in data}
        self.nodes = self.graph.nodes(data=True)

    def get_nodes_properties(self):
        properties_ = []
        for item in list(self.nodes):
            node_data = item[1]
            keys_ = list(node_data.keys())
            for property in keys_:
                properties_.append(property)

        unique_properties = set(properties_)
        return unique_properties

    def nodes_properties_dataframe(self):
        columns_ = ['name', 'namespace', 'def', 'comment', 'synonym', 'subset', 'xref', 'alt_id', 'is_a',
                    'relationship']
        data_dict = {}
        for i in columns_:
            data_dict['item_ID'] = [node_id for node_id, node_data in self.nodes]
            data_dict[i] = [node_data.get(i) for node_id, node_data in self.nodes]
        dataframe = pd.DataFrame(data_dict)
        return dataframe

    def node_properties(self, id):
        for node_id, node_data in self.nodes:
            if node_id == id:
                return node_data

    def find_parent_child_relationship(self, node_id):
        # Find edges to parent terms
        for child, parent, key in self.graph.out_edges(node_id, keys=True):
            print(f'• {self.id_to_name[child]} ⟶ {key} ⟶ {self.id_to_name[parent]}')

    def find_all_paths_to_root(self, source_term, target_term):
        # target_terms are: ['molecular_function', 'cellular_component', 'biological_process']
        source_id = self.name_to_id[source_term]
        target_id = self.name_to_id[target_term]
        paths = nx.all_simple_paths(self.graph, source=source_id, target=target_id)
        for path in paths:
            return print('•', ' ⟶ '.join(self.id_to_name[node] for node in path))
