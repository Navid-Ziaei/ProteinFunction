import obonet
import pandas as pd


class GeneOntology:
    def __init__(self, obo_path):
        self.obo_path = obo_path
        self.term_dict = self.obo_parser()
        self.graph_data = obonet.read_obo(obo_path)
        self.id_to_name = {id_: data.get('name') for id_, data in self.graph_data.nodes(data=True)}
        self.name_to_id = {data['name']: id_ for id_, data in self.graph_data.nodes(data=True) if 'name' in data}

    def obo_parser(self, valid_rel=("is_a", "part_of")):
        """
        Parse a OBO file and returns a list of ontologies, one for each namespace.
        Obsolete terms are excluded as well as external namespaces.
        """
        term_dict = {}
        term_id = None
        namespace = None
        name = None
        term_def = None
        alt_id = []
        rel = []
        obsolete = True
        with open(self.obo_path) as f:
            for line in f:
                line = line.strip().split(": ")
                if line and len(line) > 1:
                    k = line[0]
                    v = ": ".join(line[1:])
                    if k == "id":
                        # Populate the dictionary with the previous entry
                        if term_id is not None and obsolete is False and namespace is not None:
                            term_dict.setdefault(namespace, {})[term_id] = {'name': name,
                                                                            'namespace': namespace,
                                                                            'def': term_def,
                                                                            'alt_id': alt_id,
                                                                            'rel': rel}
                        # Assign current term ID
                        term_id = v

                        # Reset optional fields
                        alt_id = []
                        rel = []
                        obsolete = False
                        namespace = None

                    elif k == "alt_id":
                        alt_id.append(v)
                    elif k == "name":
                        name = v
                    elif k == "namespace" and v != 'external':
                        namespace = v
                    elif k == "def":
                        term_def = v
                    elif k == 'is_obsolete':
                        obsolete = True
                    elif k == "is_a" and k in valid_rel:
                        s = v.split('!')[0].strip()
                        rel.append(s)
                    elif k == "relationship" and v.startswith("part_of") and "part_of" in valid_rel:
                        s = v.split()[1].strip()
                        rel.append(s)

            # Last record
            if obsolete is False and namespace is not None:
                term_dict.setdefault(namespace, {})[term_id] = {'name': name,
                                                                'namespace': namespace,
                                                                'def': term_def,
                                                                'alt_id': alt_id,
                                                                'rel': rel}
        return term_dict

    def go_subclass_dataframe(self):
        subclass_name = ['biological_process', 'molecular_function', 'cellular_component']

        for name in subclass_name:
            subclass_dict = self.term_dict[name]
            dataframe = pd.DataFrame.from_dict(subclass_dict, orient='index').reset_index()
            dataframe.rename(columns={'index': 'id'}, inplace=True)
            dataframe.to_csv(r"F:\Kaggle\protein\Train\{}.csv".format(name), index=False)
