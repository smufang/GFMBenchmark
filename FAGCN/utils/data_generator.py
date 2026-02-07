from utils.pygform_creator import *
from root import ROOT_DIR
import os
import warnings

warnings.filterwarnings('ignore')
dataset_path = os.path.join(ROOT_DIR, 'datasets')


def cora():
    root = os.path.join(dataset_path, 'Cora')
    def text_transform(data):
        # Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods
        data.label_descs = [
            "which pertains to research papers that concentrate on the domain of rule-based learning, also known as rule-based machine learning. Rule learning is a method in machine learning that involves the generation of a set of rules to predict the output in a decision-making system based on the patterns discovered from the data. These rules are often in an \"if-then\" format, making them interpretable and transparent. This category would encompass research involving various rule learning algorithms, their enhancements, theoretical foundations, and applications. Rule learning methods are particularly beneficial in domains where interpretability and understanding of the learned knowledge is important, such as in medical diagnosis, credit risk prediction, and more.",
            "which refers to research papers revolving around the concept of artificial neural networks (ANNs). Neural networks are a subset of machine learning algorithms modelled after the human brain, designed to ""learn"" from observational data. They are the foundation of deep learning technologies and can process complex data inputs, find patterns, and make decisions. The network consists of interconnected layers of nodes, or ""neurons"", and each connection is assigned a weight that shapes the data and helps produce a meaningful output. Topics covered under this category could range from the architecture and function of different neural network models, advancements in training techniques, to their application in a multitude of fields such as image and speech recognition, natural language processing, and medical diagnosis.",
            "which refers to research papers focusing on case-based reasoning (CBR) in the field of artificial intelligence. Case-based reasoning is a problem-solving approach that utilizes specific knowledge of previously encountered, concrete problem situations (cases). In this method, a new problem is solved by finding similar past cases and reusing them in the new situation. The approach relies on the idea of learning from past experiences to solve new problems, which makes it relevant in many applications including medical diagnosis, legal decision-making, and others. Thus, the ""Case Based"" category would include papers that primarily focus on this particular methodology and its various aspects.",
            "which would include research papers related to genetic algorithms (GAs). Genetic algorithms are a type of optimization and search algorithms inspired by the process of natural selection and genetics. These algorithms generate solutions to optimization problems using techniques inspired by natural evolution, such as inheritance, mutation, selection, and crossover. In practice, genetic algorithms can be used to find solutions to complex problems that are difficult to solve with traditional methods, particularly in domains where the search space is large, complex, or poorly understood. This category would cover various aspects of genetic algorithms, including their design, analysis, implementation, theoretical background, and diverse applications.",
            "which likely refers to research papers that delve into the theoretical aspects of machine learning and artificial intelligence. This includes a broad array of topics such as theoretical foundations of various machine learning algorithms, performance analysis, studies on learning theory, statistical learning, information theory, and optimization methods. Additionally, it could encompass the development of new theoretical frameworks, investigations into the essence of intelligence, the potential for artificial general intelligence, as well as the ethical implications surrounding AI. Essentially, the ""Theory"" category encapsulates papers that primarily focus on theoretical concepts and discussions, contrasting with more application-oriented research which centers on specific techniques and their practical implementation.",
            "which pertains to research papers that concentrate on the domain of rule-based learning, also known as rule-based machine learning. Rule learning is a method in machine learning that involves the generation of a set of rules to predict the output in a decision-making system based on the patterns discovered from the data. These rules are often in an ""if-then"" format, making them interpretable and transparent. This category would encompass research involving various rule learning algorithms, their enhancements, theoretical foundations, and applications. Rule learning methods are particularly beneficial in domains where interpretability and understanding of the learned knowledge is important, such as in medical diagnosis, credit risk prediction, and more.",
            "which pertains to research papers that focus on probabilistic methods and models in machine learning and artificial intelligence. Probabilistic methods use the mathematics of probability to make predictions and decisions. They provide a framework to handle and quantify the uncertainty and incomplete information, which is a common scenario in real-world problems. This category could include topics like Bayesian networks, Gaussian processes, Markov decision processes, and statistical techniques for prediction and inference. These methods have applications in various areas such as computer vision, natural language processing, robotics, and data analysis, among others, due to their ability to model complex, uncertain systems and make probabilistic predictions."
        ]
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['cites']
        return data
    dataset = Cora(root=root, transform=text_transform)
    return dataset


def pubmed():
    root = os.path.join(dataset_path, 'Pubmed')
    def text_transform(data):
        # Diabetes Mellitus Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2
        data.label_descs = [
            ' which is a category of scientific literature found on PubMed that encompasses research related to experimental studies on diabetes mellitus. This category includes studies conducted in laboratory settings, often using animal models or cell cultures, to investigate various aspects of diabetes, such as its pathophysiology, treatment strategies, and potential interventions. Researchers in this field aim to better understand the underlying mechanisms of diabetes and develop experimental approaches to prevent or manage the disease. Experimental studies in this category may explore topics like insulin resistance, beta cell function, glucose metabolism, and the development of novel therapies for diabetes.',
            ' which focuses on scientific research related specifically to Type 1 diabetes mellitus. This category encompasses a wide range of studies, including clinical trials, epidemiological investigations, and basic research, all centered on understanding, diagnosing, managing, and potentially curing Type 1 diabetes. Researchers in this field explore areas such as the autoimmune processes underlying the disease, insulin therapy, glucose monitoring, pancreatic islet transplantation, and novel treatments aimed at improving the lives of individuals with Type 1 diabetes. It serves as a valuable resource for healthcare professionals, scientists, and policymakers interested in advancements related to Type 1 diabetes management and research.',
            ' which focuses on research related to Type 2 diabetes (T2D), and it can be differentiated from Diabetes Mellitus Type 1 (T1D) in the following ways: Etiology (Cause): Type 2 Diabetes (T2D): T2D is primarily characterized by insulin resistance, where the body\'s cells do not respond effectively to insulin, and relative insulin deficiency that develops over time. It is not primarily an autoimmune condition.'
        ]
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['cites']
        return data
    dataset = PubMed(root=root, transform=text_transform)
    return dataset


def ogbn_arxiv():
    root = os.path.join(dataset_path, 'ogbn-arxiv')
    def text_transform(data):
        # Diabetes Mellitus Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2
        data = undirected_transform(data)
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['cites']
        return data
    dataset = Arxiv(root=root, transform=text_transform)
    return dataset


def ogbn_mag():
    from torch_geometric.datasets import OGB_MAG
    root = os.path.join(dataset_path, 'ogbn-mag')
    dataset = OGB_MAG(root=root, transform=get_mag_metapath2vec_transform(root,is_reverse_added=True))
    return dataset


def acm():
    from torch_geometric.datasets import HGBDataset
    root = os.path.join(dataset_path, 'ACM')
    def text_transform(data):
        data['paper'].label_names = ['Database', 'Wireless Communication', 'Data Mining']
        data['paper'].label_descs = [
            "Research papers focusing on database systems, including storage structures, indexing, query optimization, data integration and transaction management.",
            "Research papers focusing on wireless communication, covering topics such as mobile networks, wireless signal processing, network protocols, and emerging wireless technologies.",
            "Research papers focusing on data mining and knowledge discovery, including pattern mining, machine learning for large-scale data, clustering, classification, and anomaly detection."
        ]
        edge_map = {
            ('paper', 'cite', 'paper'): ('paper', 'cites', 'paper'),
            ('paper', 'ref', 'paper'): ('paper', 'rev_cites', 'paper'),
            ('paper', 'to', 'author'): ('paper', 'rev_writes', 'author'),
            ('author', 'to', 'paper'): ('author', 'writes', 'paper'),
            ('paper', 'to', 'subject'): ('paper', 'belongs_to', 'subject'),
            ('subject', 'to', 'paper'): ('subject', 'rev_belongs_to', 'paper'),
            ('paper', 'to', 'term'): ('paper', 'mentions', 'term'),
            ('term', 'to', 'paper'): ('term', 'rev_mentions', 'paper'),
        }
        for old_key, new_key in edge_map.items():
            data[new_key].edge_index = data[old_key].edge_index
            del data[old_key]
        return data
    dataset = HGBDataset(root=root, name='ACM', transform=text_transform)
    return dataset


def dblp():
    from torch_geometric.datasets import HGBDataset
    root = os.path.join(dataset_path, 'DBLP')
    def text_transform(data):
        data['paper'].label_names = ['Database', 'Data Mining', 'Artificial Intelligence', 'Information Retrieval']
        data['paper'].label_descs = [
            "which refers to research papers focused on database systems, including data storage, indexing, query processing, transaction management and related architectures.",
            "which refers to research papers in the area of data mining, including pattern discovery, clustering, classification, large-scale data analysis and knowledge extraction.",
            "which refers to research papers in the field of artificial intelligence, including machine learning, reasoning, computer vision, natural language processing, and general AI methods.",
            "which refers to research papers in the domain of information retrieval, including search engines, ranking algorithms, user modelling, large-scale text retrieval, and web/information systems."
        ]
        edge_map = {
            ('author', 'to', 'paper'): ('author', 'writes', 'paper'),
            ('paper', 'to', 'author'): ('paper', 'rev_writes', 'author'),
            ('paper', 'to', 'term'): ('paper', 'mentions', 'term'),
            ('term', 'to', 'paper'): ('term', 'rev_mentions', 'paper'),
            ('paper', 'to', 'venue'): ('paper', 'published_in', 'venue'),
            ('venue', 'to', 'paper'): ('venue', 'rev_published_in', 'paper'),
        }
        for old_key, new_key in edge_map.items():
            data[new_key].edge_index = data[old_key].edge_index
            del data[old_key]
        return data
    dataset = HGBDataset(root=root, name='DBLP', transform=text_transform)
    return dataset


def genre():
    root = os.path.join(dataset_path, 'Genre')
    dataset = Genre(root=root)
    return dataset


def reddit():
    from torch_geometric.datasets import JODIEDataset
    root = os.path.join(dataset_path, 'Reddit')
    def text_transform(data):
        # predicting if a user will be banned. Till a user is banned, the label of the user is ‘0’, and their last interaction has the label ‘1’. 
        # For users that are not banned, the label is always ‘0’.
        data.label_names = ['Normal Users', 'Banned Users']
        data.label_descs = [
            "which refers to users who have not been banned from the platform.",
            "which refers to users who have been banned from the platform.",
        ]
        data.src_texts = ['user']
        data.dst_texts = ['post']
        data.relation_texts = ['interacts']
        return data
    dataset = JODIEDataset(root=root, name='Reddit', transform=text_transform)
    return dataset


def wikipedia():
    from torch_geometric.datasets import JODIEDataset
    root = os.path.join(dataset_path, 'Wikipedia')
    def text_transform(data):
        data.label_names = ['Normal Users', 'Banned Users']
        data.label_descs = [
            "which refers to users who have not been banned from the platform.",
            "which refers to users who have been banned from the platform.",
        ]
        data.src_texts = ['user']
        data.dst_texts = ['page']
        data.relation_texts = ['edits']
        return data
    dataset = JODIEDataset(root=root, name='Wikipedia', transform=text_transform)
    return dataset


def actor():
    root = os.path.join(dataset_path, 'Actor')
    def text_transform(data):
        data = HeTGB_undirected_transform(data)
        # American film actors (only), American film actors and American television actors, American television actors and American stage actors, English actors, Canadian actors
        data.label_descs = [
            'Actors exclusively labeled as American film actors, typically appearing only in motion pictures produced in the United States.',
            'Actors labeled as both American film actors and American television actors, having appeared in both U.S. films and television productions.',
            'Actors labeled as both American television actors and American stage actors, with significant roles in television series and live theater performances.',
            'Actors associated with England, including English film, television, or stage actors, reflecting performance work primarily in the UK.',
            'Actors associated with Canada, including film, television, or stage actors primarily active in Canadian productions.'
        ]
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['co-occurrence']
        return data
    dataset = Actor(root=root, transform=text_transform)
    return dataset


def texas():
    # from torch_geometric.datasets import WebKB
    root = os.path.join(dataset_path, 'Texas')
    def text_transform(data):
        data = HeTGB_undirected_transform(data)
        # student, course, project, staff, faculty
        data.label_descs = [
            'Web pages corresponding to students, typically containing personal or academic information related to the individual.',
            'Web pages corresponding to courses offered by the department, including course descriptions, schedules, and materials.',
            'Web pages related to research or academic projects, describing objectives, participants, and outcomes.',
            'Web pages corresponding to staff members, including administrative personnel and support staff associated with the department.',
            'Web pages corresponding to faculty members, including professors, lecturers, and researchers affiliated with the department.'
        ]
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['links_to']
        return data
    dataset = Texas(root=root, transform=text_transform)
    return dataset

def wisconsin():
    # from torch_geometric.datasets import WebKB
    root = os.path.join(dataset_path, 'Wisconsin')
    def text_transform(data):
        data = HeTGB_undirected_transform(data)
        # student, course, project, staff, faculty
        data.label_descs = [
            'Web pages corresponding to students, typically containing personal or academic information related to the individual.',
            'Web pages corresponding to courses offered by the department, including course descriptions, schedules, and materials.',
            'Web pages related to research or academic projects, describing objectives, participants, and outcomes.',
            'Web pages corresponding to staff members, including administrative personnel and support staff associated with the department.',
            'Web pages corresponding to faculty members, including professors, lecturers, and researchers affiliated with the department.'
        ]
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['links_to']
        return data
    dataset = Wisconsin(root=root, transform=text_transform)
    return dataset


def chameleon():
    from torch_geometric.datasets import WikipediaNetwork
    root = os.path.join(dataset_path, 'Chameleon')
    def text_transform(data):
        data = HeTGB_undirected_transform(data)
        # course, project, staff, faculty, student
        data.raw_texts = ['web page'] * data.num_nodes
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['links_to']
        return data
    dataset = WikipediaNetwork(root=root, name='chameleon', transform=text_transform)
    return dataset


def cornell():
    # from torch_geometric.datasets import WebKB
    root = os.path.join(dataset_path, 'Cornell')
    def text_transform(data):
        data = HeTGB_undirected_transform(data)
        # student, course, project, staff, faculty
        data.label_descs = [
            'Web pages corresponding to students, typically containing personal or academic information related to the individual.',
            'Web pages corresponding to courses offered by the department, including course descriptions, schedules, and materials.',
            'Web pages related to research or academic projects, describing objectives, participants, and outcomes.',
            'Web pages corresponding to staff members, including administrative personnel and support staff associated with the department.',
            'Web pages corresponding to faculty members, including professors, lecturers, and researchers affiliated with the department.'
        ]
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['links_to']
        return data
    dataset = Cornell(root=root, transform=text_transform)
    return dataset


def imdb():
    from torch_geometric.datasets import HGBDataset
    root = os.path.join(dataset_path, 'IMDB')
    def text_transform(data):
        data['movie'].label_names = ['action', 'comedy', 'drama', 'romance', 'thriller']
        data['movie'].label_descs = [
            "movies that primarily focus on action‑packed sequences, physical feats, and high‑energy scenes designed to thrill and excite the audience.",
            "movies that are designed to entertain and amuse the audience through humor, light‑hearted plots, and comedic situations.",
            "movies that focus on emotional narratives, character development, and realistic portrayals of life situations, often dealing with serious themes.",
            "movies oriented around romantic relationships, emotional arc and love stories.",
            "movies designed with suspense, tension, unexpected plot twists, and high stakes."
        ]
        edge_map = {
            ('movie', 'to', 'director'): ('movie', 'directed_by', 'director'),
            ('director', 'to', 'movie'): ('director', 'rev_directed_by', 'movie'),
            ('actor', 'to', 'movie'): ('actor', 'rstars', 'movie'),
            ('movie', 'to', 'keyword'): ('movie', 'described_by', 'keyword'),
            ('keyword', 'to', 'movie'): ('keyword', 'rev_described_by', 'movie'),
        }
        for old_key, new_key in edge_map.items():
            data[new_key].edge_index = data[old_key].edge_index
            del data[old_key]
        return data
    dataset = HGBDataset(root=root, name='IMDB', transform=text_transform)
    return dataset


def yelp():
    from torch_geometric.datasets import Yelp
    root = os.path.join(dataset_path, 'Yelp')
    dataset = Yelp(root=root)
    return dataset


def yelpSemRec():
    root = os.path.join(dataset_path, 'YelpSemRec')
    dataset = YelpSemRec(root=root)
    return dataset


def photo():
    from torch_geometric.datasets import Amazon
    root = os.path.join(dataset_path, 'Photo')
    def text_transform(data):
        data.raw_texts = ['product'] * data.num_nodes
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['co-purchasing']
        return data
    dataset = Amazon(root=root, name='Photo', transform=text_transform)
    return dataset


def computers():
    from torch_geometric.datasets import Amazon
    root = os.path.join(dataset_path, 'Computers')
    def text_transform(data):
        data.raw_texts = ['product'] * data.num_nodes
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['co-purchasing']
        return data
    dataset = Amazon(root=root, name='Computers', transform=text_transform)
    return dataset


def amazon():
    root = os.path.join(dataset_path, 'Amazon')
    dataset = Amazon(root=root)
    return dataset


def amazonh():
    root = os.path.join(dataset_path, 'Amazon-HeTGB')
    def text_transform(data):
        data = HeTGB_undirected_transform(data)
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['co-purchasing']
        return data
    dataset = AmazonH(root=root, transform=text_transform)
    return dataset


def products():
    root = os.path.join(dataset_path, 'Products')
    def text_transform(data):
        data = HeTGB_undirected_transform(data)
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['co-purchasing']
        return data
    dataset = Products(root=root, transform=text_transform)
    return dataset


def hiv():
    from torch_geometric.datasets import MoleculeNet
    root = os.path.join(dataset_path, 'HIV')
    dataset = MoleculeNet(root=root, name='HIV')
    # the ability to inhibit HIV replication for over 40,000 compounds.
    dataset.data.label_names = ['inactive', 'active']
    dataset.data.label_descs = [
        "which refers to compounds that do not exhibit inhibitory activity against HIV replication.",
        "which refers to compounds that demonstrate inhibitory activity against HIV replication."
    ]
    dataset.data.raw_texts = ['molecule'] * dataset.data.num_nodes
    return dataset


def pcba():
    from torch_geometric.datasets import MoleculeNet
    root = os.path.join(dataset_path, 'PCBA')
    dataset = MoleculeNet(root=root, name='PCBA')
    return dataset


def cox2():
    from torch_geometric.datasets import TUDataset
    root = os.path.join(dataset_path, 'COX2')
    dataset = TUDataset(root=root, name='COX2')
    dataset.data.label_names = ['inactive', 'active']
    dataset.data.label_descs = [
        "which refers to COX-2 inhibitor molecules classified as inactive based on a pIC50 threshold of 6.5. These compounds show little or no inhibitory activity against the cyclooxygenase-2 (COX-2) enzyme.",
        "which refers to COX-2 inhibitor molecules classified as active based on a pIC50 threshold of 6.5. These compounds effectively inhibit the cyclooxygenase-2 (COX-2) enzyme and are potential candidates for anti-inflammatory therapies."
    ]
    return dataset


def bzr():
    from torch_geometric.datasets import TUDataset
    root = os.path.join(dataset_path, 'BZR')
    dataset = TUDataset(root=root, name='BZR')
    dataset.data.label_names = ['inactive', 'active']
    dataset.data.label_descs = [
        "which refers to BZR ligands classified as inactive based on a pIC50 threshold of 7.0. These molecules exhibit low binding affinity for the benzodiazepine receptor and are not considered effective.",
        "which refers to BZR ligands classified as active based on a pIC50 threshold of 7.0. These molecules exhibit high binding affinity for the benzodiazepine receptor and are considered effective for receptor modulation."
    ]
    return dataset


def proteins():
    from torch_geometric.datasets import TUDataset
    root = os.path.join(dataset_path, 'PROTEINS')
    dataset = TUDataset(root=root, name='PROTEINS')
    dataset.data.label_names = ['enzyme', 'non-enzyme']
    dataset.data.label_descs = [
        "which refers to proteins that catalyze biochemical reactions, often possessing structural features like an active site and a large surface pocket.",
        "which refers to all proteins that do not function as enzymes, representing a more structurally and functionally diverse class."
    ]
    return dataset


def enzymes():
    from torch_geometric.datasets import TUDataset
    root = os.path.join(dataset_path, 'ENZYMES')
    dataset = TUDataset(root=root, name='ENZYMES')
    return dataset


def ogbn_proteins():
    from ogb.nodeproppred import PygNodePropPredDataset
    root = os.path.join(dataset_path, 'ogbn-proteins')
    def text_transform(data):
        data.raw_texts = ['protein'] * data.num_nodes
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['association']
        return data
    with patch('builtins.input', return_value='y'):
        dataset = PygNodePropPredDataset(name='ogbn-proteins', root=root, transform=text_transform)
    return dataset


def ogbg_ppa():
    from ogb.graphproppred import PygGraphPropPredDataset
    root = os.path.join(dataset_path, 'ogbg-ppa')
    with patch('builtins.input', return_value='y'):
        dataset = PygGraphPropPredDataset(name='ogbg-ppa', root=root)
    return dataset


def wiki():
    from torch_geometric.datasets import Wikidata5M
    root = os.path.join(dataset_path, 'Wiki')
    def text_transform(data):
        data.raw_texts = ['entity'] * data.num_nodes
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['relation']
        return data
    dataset = Wikidata5M(root=root, transform=text_transform)
    return dataset


def fb15k237():
    root = os.path.join(dataset_path, 'FB15K-237')
    dataset = FB15k237(root=root)
    return dataset


def nell():
    from torch_geometric.datasets import NELL
    root = os.path.join(dataset_path, 'NELL')
    def text_transform(data):
        mapping = {old: new for new, old in enumerate(data.y.unique().tolist())}
        data.y = torch.tensor([mapping[x.item()] for x in data.y], dtype=torch.long)

        data.raw_texts = ['entity'] * data.num_nodes
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['relation']
        return data
    dataset = NELL(root=root, transform=text_transform)
    return dataset


def wn18rr():
    root = os.path.join(dataset_path, 'WN18RR')
    dataset = WN18RR(root=root)
    return dataset


def tfinance():
    root = os.path.join(dataset_path, 'T-Finance')
    def label_transform(data):
        data = undirected_transform(data)
        # The nodes are unique anonymized accounts with 10-dimension features related to registration days, logging activities and interaction frequency. 
        # The edges in the graph represent two accounts that have transaction records. 
        # Human experts annotate nodes as anomalies if they fall into categories like fraud, money laundering and online gambling.
        data.label_names = ['normal account', 'anomaly account']
        data.label_descs = [
            "which refers to accounts that behave normally according to platform records, with no indications of suspicious activity in registration, login, or transaction behaviors.",
            "which refers to accounts identified by human experts as exhibiting anomalous behaviors, including fraud, money laundering, or online gambling, based on account activity features and transaction interactions."
        ]
        data.raw_texts = ['account'] * data.num_nodes
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['transaction']
        return data
    dataset = TFinance(root=root, transform=label_transform)
    return dataset


def elliptic():
    from torch_geometric.datasets import EllipticBitcoinDataset
    root = os.path.join(dataset_path, 'Elliptic')
    def label_transform(data):
        data = undirected_transform(data)
        mask = (data.y == 2) # from [0,1,2] to [0,1,-1]
        data.y[mask] = -1
        # A given transaction is deemed licit (versus illicit) if the entity initiating the transaction (i.e., the entity controlling the private keys associated with the input addresses of a specific transaction) belongs to a licit (illicit) category
        # The remaining transactions are not labelled with regard to licit versus illicit, but have other features.
        data.label_names = ['licit', 'illicit', 'others']
        data.label_descs = [
            "which refers to transactions where the initiating entity controlling the private keys is categorized as licit, indicating legitimate financial behavior that complies with regulatory and legal frameworks.",
            "which refers to transactions where the initiating entity—the one controlling the private keys associated with the input addresses—belongs to an illicit category. These transactions are typically linked to illegal financial activities or suspicious operations.",
            "which refers to transactions that have not been labeled as either licit or illicit. Although they are not directly classified, they contain other relevant features and contribute structural and contextual information to the overall transaction graph."
        ]
        data.raw_texts = ['entity'] * data.num_nodes
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['transaction']
        return data
    dataset = EllipticBitcoinDataset(root=root, transform=label_transform)
    return dataset


def dgraph():
    from torch_geometric.datasets import DGraphFin
    root = os.path.join(dataset_path, 'DGraph')

    def zero_based(data):
        if hasattr(data, "edge_type") and data.edge_type.min().item() == 1:
            # from [1-11] to [0-10]
            data.edge_type.sub_(1)
        mask = (data.y == 2) | (data.y == 3) # background nodes
        data.y[mask] = -1
        # These nodes are labeled based on their borrowing behavior. 
        # We define users who exhibit at least one fraud activity, which means they do not repay the loans a long time after the due date and ignore the platform’s repeated reminders, as anomalies/fraudsters.
        # Background nodes who are registered users but have no borrowing behavior from the platform.
        data.label_names = ['normal user', 'fraudster', 'background nodes']
        data.label_descs = [
            "which refers to users who have not engaged in fraudulent borrowing activities on the platform.",
            "which refers to users who exhibit at least one fraud activity, which means they do not repay the loans a long time after the due date and ignore the platform’s repeated reminders.",
            "which refers to registered users who have no borrowing behavior from the platform."
        ]
        data.raw_texts = ['user'] * data.num_nodes
        return data
    
    dataset = DGraphFin(root=root, transform=zero_based)
    return dataset


def citeseer():
    from torch_geometric.datasets import Planetoid
    root = os.path.join(dataset_path, 'Citeseer')
    def label_transform(data):
        data.label_names = ['Agents', 'Machine Learning', 'Information Retrieval', 'Database',
                            'Human Computer Interaction', 'Artificial Intelligence']
        data.label_descs = [
            "Specifically, Agents are autonomous entities that perceive their environment through sensors and act upon it using actuators. They are designed to achieve specific goals or tasks.",
            "Specifically, Machine Learning research investigates how to create systems that can automatically improve their performance on tasks by identifying patterns and insights from vast amounts of data. Researchers in Machine Learning explore diverse techniques such as supervised learning, unsupervised learning, reinforcement learning, and deep learning to build systems that can predict outcomes, classify data, and make intelligent decisions.",
            "Specifically, Information Retrieval research focuses on the study of information retrieval systems, which are designed to help users find relevant information in large collections of data. Researchers in Information Retrieval explore techniques such as indexing, querying, and ranking to build systems that can efficiently retrieve information based on user queries.",
            "Specifically, Database research investigates how to design, build, and manage databases, which are organized collections of data that can be accessed, managed, and updated. Researchers in Database Systems explore techniques such as data modeling, query languages, and transaction processing to build systems that can store, retrieve, and manipulate data.",
            "Specifically, Human Computer Interaction research focuses on the study of human-computer interaction, which explores how people interact with computers and other digital technologies. Researchers in Human Computer Interaction investigate how to design user-friendly interfaces, improve usability, and enhance user experience to build systems that are intuitive, efficient, and effective.",
            "Specifically, Artificial Intelligence research investigates how to create intelligent systems that can perform tasks that typically require human intelligence, such as perception, reasoning, learning, and decision-making. Researchers in Artificial Intelligence explore diverse techniques such as knowledge representation, planning, and natural language processing to build systems that can solve complex problems, adapt to new environments, and interact with humans.",
        ]
        data.node_texts = ['paper'] * data.num_nodes
        data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)
        data.relation_texts = ['cites']
        return data
    dataset = Planetoid(root=root, name='Citeseer', transform=label_transform)
    return dataset


def facebook():
    from torch_geometric.datasets import FacebookPagePage
    root = os.path.join(dataset_path, 'Facebook')
    dataset = FacebookPagePage(root=root)
    return dataset


def lastfm():
    from torch_geometric.datasets import LastFMAsia
    root = os.path.join(dataset_path, 'LastFM')
    dataset = LastFMAsia(root=root)
    return dataset

def squirrel():
    from torch_geometric.datasets import WikipediaNetwork
    root = os.path.join(dataset_path, 'Squirrel')
    dataset = WikipediaNetwork(root=root, name='squirrel')
    return dataset