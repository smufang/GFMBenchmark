eval_template={
    "Pubmed": "it belongs to {c} research area",
    "Wikipedia": "{c}",
    "Actor": "this movie is acted by {c}",
    "T-Finance": "this account is {c}",
    "DGraph": "{c}",
    "BZR": "the molecule class is classified as {c}",
    "WN18RR": "relation type is {c}",
    'Cora': "this paper has a topic on {c}",
    'ACM': "this paper has a topic on {c}",
    'Reddit': "{c}",
    'Wisconsin': "this webpage belongs to {c}",
    'Elliptic': "this transaction is {c}",
    'FB15K-237': "relation type is {c}",
    'HIV': "compound activity is {c}",
    'COX2': "compound is classified as {c}",
    'PROTEINS': "this protein is {c}",
}


pretrain_categories = {
    'Pubmed': "specific research areas (e.g., 'Diabetes Mellitus Experimental', 'Diabetes Mellitus Type1', 'Diabetes Mellitus Type2')",
    'Cora': 'Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods',
    'ogbn-arxiv': 'arXiv CS sub-category',
    'ACM': 'Database, Wireless Communication, Data Mining',
    'DBLP': 'Database, Data Mining, Artificial Intelligence, Information Retrieval',
    'Reddit': 'Normal Users, Banned Users',
    'Texas': 'student, faculty, staff, course, project',
    'Wisconsin': 'student, faculty, staff, course, project',
    'Cornell': 'student, faculty, staff, course, project',
    'IMDB': 'action, comedy, drama, romance and thriller',
    'Photo': 'Amazon - Camera & Photo Category',
    'Computers': 'Amazon - Computers & Accessories Category',
    'Amazon': 'Amazon - Electronics categories',
    'Amazon-HeTGB': 'Rating: 1) 5.0, 2) 4.5, 3) 4.0, 4) 3.5, 5) lower than 3.5',
    'HIV': 'Inactive (CI) or Active (CA/CM)',
    'COX2': 'Active or Inactive',
    'PROTEINS': 'Enzyme or Non-enzyme',
    'ENZYMES': 'Oxidoreductases, Transferases, Hydrolases, Lyases, Isomerases, Ligases',# These are the common six categories, but they are not specified in the papers and datasets.
    'FB15K-237': 'Freebase category', # this is a placeholder, as FB15K-237 does not have specific categories defined in the dataset
    'NELL': ' 105 Concept categories such as person, organization, location, food, company.',
    'Elliptic': 'licit, illicit, or others',
}
pretrain_prompts = {
'Pubmed': '''
I am providing you with a GraphML file depicting a citation network in biomedicine.
Each node in the network represents a research paper, and each edge indicates the citation relationship between papers.
Please analyze the paper represented by node ‘n{seed}’ using the provided GraphML data in the following two ways:

1. Paper Summary and Context Analysis: 
- Extract and summarize the key findings or contributions of the paper denoted by ‘n{seed}’. Consider the details embedded within node ‘n{seed}’, including its title, abstract, and keywords (if available).
- Provide an overall summary of prevalent themes or concepts shared by the papers that cite or are cited by ‘n{seed}’ (its direct neighbors in the network). Identify common threads or research topics among these neighbors.

2. Research Area Classification:
- Based on the information summarized from ‘n{seed}’ and its neighboring nodes, determine one of the specific research areas: {categories} to which ‘n{seed}’ primarily contributes.
- Justify the classification by explaining which aspects of ‘n{seed}’ align with recognized themes, issues, or methodologies in the identified research area(s).

Please ensure your analyses are grounded in the data provided by the GraphML file within 500 tokens, focusing on node ‘n{seed}’ and its immediate citation neighborhood. The detailed GraphML citation network data is as follows:
''',
'Cora': '''
I am providing you with a GraphML file depicting a citation network in computer science.
Each node in the network represents a research paper, and each edge indicates the citation relationship between papers.
Please analyze the paper represented by node ‘n{seed}’ using the provided GraphML data in the following two ways:

1. Paper Summary and Context Analysis: 
- Extract and summarize the key findings or contributions of the paper denoted by ‘n{seed}’. Consider the details embedded within node ‘n{seed}’, including its title, abstract, and keywords (if available).
- Provide an overall summary of prevalent themes or concepts shared by the papers that cite or are cited by ‘n{seed}’ (its direct neighbors in the network). Identify common threads or research topics among these neighbors.

2. Research Area Classification:
- Based on the information summarized from ‘n{seed}’ and its neighboring nodes, determine one of the specific research areas: {categories} to which ‘n{seed}’ primarily contributes.
- Justify the classification by explaining which aspects of ‘n{seed}’ align with recognized themes, issues, or methodologies in the identified research area(s).

Please ensure your analyses are grounded in the data provided by the GraphML file within 500 tokens, focusing on node ‘n{seed}’ and its immediate citation neighborhood. The detailed GraphML citation network data is as follows:
''',
'ogbn-arxiv': '''
I am providing you with a GraphML file depicting a citation network in computer science.
Each node in the network represents a scholarly article, and each edge signifies a citation relationship between articles.
Please analyze the article represented by node ‘n{seed}’ using the provided GraphML data in the following two ways:
    
1. Paper Summary and Context Analysis:
- Extract and summarize the key findings or contributions of the paper denoted by ‘n{seed}’. Consider the details embedded within node ‘n{seed}’, including its title, abstract, and keywords (if available).
- Provide an overall summary of prevalent themes or concepts shared by the papers that cite or are cited by ‘n{seed}’ (its direct neighbors in the network). Identify common threads or research topics among these neighbors.

2. Research Area Classification:
- Based on the information summarized from ‘n{seed}’ and its neighboring nodes, determine one of the specific research areas: {categories} to which ‘n{seed}’ primarily contributes.
- Justify the classification by explaining which aspects of ‘n{seed}’ align with recognized themes, issues, or methodologies in the identified research area(s).

Please ensure your analyses are grounded in the data provided by the GraphML file within 500 tokens, focusing on node ‘n{seed}’ and its immediate citation neighborhood. The detailed GraphML citation network data is as follows:
''',
'ACM': '''
I am providing you with a GraphML file representing a heterogeneous academic graph in computer science.
Each node contains a text attribute that describes its node type (paper, author, subject, or term) and the types of edges it has to other nodes.
The graph includes relations such as citations (paper-paper), references (paper-paper), authorship (author-paper), subject membership (paper-subject), and term/keyword assignment (paper-term), along with their reverse edges.
Please analyze the node represented by ‘n{seed}’ using the provided GraphML data in the following two ways:

1. Node Summary and Context Analysis:
- Summarize the key structural properties of the node ‘n{seed}’ based on its text attribute, including its node type and the types of edges it has.
- For each type of edge mentioned in the text, provide an overview of prevalent patterns or characteristics shared by the neighbors connected via that edge. Identify common connections, research threads, or structural themes among these neighbors.

2. Research Area Classification (only for paper nodes):
- If ‘n{seed}’ is a paper node (as indicated in its text attribute), determine one of the specific research areas: {categories} it primarily belongs to based on its neighbors and edge patterns described in the text.
- Justify the classification by explaining which aspects of ‘n{seed}’ (its text, linked nodes, and neighbor edge patterns) align with recognized research areas.
- If ‘n{seed}’ is not a paper node, skip this step.

Please ensure your analyses are grounded in the data provided by the GraphML file within 500 tokens, focusing on node ‘n{seed}’ and its immediate neighborhood. The detailed GraphML heterogeneous graph data is as follows:
''',
'DBLP': '''
I am providing you with a GraphML file representing a heterogeneous academic graph in computer science.
Each node contains a text attribute that describes its node type (paper, author, term, or  venue) and the types of edges it has to other nodes.
The graph includes relations such as authorship (paper-author), publication (paper-venue), term assignment (paper-term), and citations (paper-paper), along with their reverse edges.
Please analyze the node represented by ‘n{seed}’ using the provided GraphML data in the following two ways:

1. Node Summary and Context Analysis:
- Summarize the key structural properties of the node ‘n{seed}’ based on its text attribute, including its node type and the types of edges it has.
- For each type of edge mentioned in the text, provide an overview of prevalent patterns or characteristics shared by the neighbors connected via that edge. Identify common connections, research threads, or structural themes among these neighbors.

2. Research Area Classification (only for paper nodes):
- If ‘n{seed}’ is a paper node (as indicated in its text attribute), determine one of the specific research areas: {categories} it primarily belongs to based on its neighbors and edge patterns described in the text.
- Justify the classification by explaining which aspects of ‘n{seed}’ (its text, linked nodes, and neighbor edge patterns) align with recognized research areas.
- If ‘n{seed}’ is not a paper node, skip this step.

Please ensure your analyses are grounded in the data provided by the GraphML file within 500 tokens, focusing on node ‘n{seed}’ and its immediate neighborhood. The detailed GraphML heterogeneous graph data is as follows:
''',
'Reddit': '''
I am providing you with a GraphML file representing a heterogeneous temporal interaction graph derived from social network.
Each node has a text attribute indicating whether it is a source (user, labeled as “src”) or a destination (post, labeled as “dst”). Directed edges represent interactions from users to posts, and reverse edges are also added to ensure bidirectional connectivity. The temporal order of edges reflects the posting and replying sequence.
Please analyze the node represented by ‘n{seed}’ in the following two aspects based on the provided GraphML data:

1. Node Summary and Interaction Context:
- Identify whether node ‘n{seed}’ is a user or a post according to its text attribute, and summarize its key characteristics such as activity level, interaction diversity, or textual features derived from LIWC-based post representations.
- Examine the temporal and structural patterns within its immediate neighborhood (via ‘src’ → ‘dst’ or reverse connections). Discuss how these patterns reflect behavioral tendencies, engagement style, or participation in subreddit communities.

2. Behavioral or State Classification (only for source nodes):
- If ‘n{seed}’ is a source node, assess its behavioral tendency or risk of state change ({categories}) based on its historical interaction sequence and neighboring post features.
- Provide reasoning grounded in patterns or textual indicators extracted from the GraphML data.
- If ‘n{seed}’ is a destination node, skip this step.

Please ensure your analysis remains within 500 tokens and focuses strictly on node ‘n{seed}’ and its local temporal neighborhood. The detailed GraphML temporal interaction data is as follows:
''',
'Texas': '''
I have a GraphML file representing a web-based social network constructed from university computer science department webpages.
Each node denotes a webpage, the node features correspond to the textual content extracted from that page, and edges denote hyperlinks between pages. Reverse edges have been added to ensure bidirectional connectivity.
I would like you to analyze the webpage represented by the node ‘n{seed}’ using the GraphML data in the following two ways:

1. Content Summary and Context Analysis:
- Extract and summarize the textual content associated with ‘n{seed}’. Identify the main topics, keywords, or focus areas of this webpage (e.g., course information, faculty introduction, research description).
- Analyze the webpages that link to or are linked from ‘n{seed}’. Summarize the general themes or purposes of these neighboring pages and describe how ‘n{seed}’ fits within its local hyperlink structure (e.g., within a research group, course cluster, or student directory).

2. Category Classification:
- Based on both the textual and link-based context, classify whether the webpage represented by ‘n{seed}’ most likely belongs to one of the categories: {categories}.
- Justify your classification by referencing textual cues (e.g., titles, mentions, content focus) and structural patterns (e.g., connections to other similar pages).

Your analysis should be directly based on the data provided in the GraphML file and should be limited to 500 tokens. Focus exclusively on node ‘n{seed}’ and its immediate neighborhood within the web-based hyperlink network. The detailed GraphML web network data is as follows:
''',
'Wisconsin': '''
I have a GraphML file representing a web-based social network constructed from university computer science department webpages.
Each node denotes a webpage, the node features correspond to the textual content extracted from that page, and edges denote hyperlinks between pages. Reverse edges have been added to ensure bidirectional connectivity.
I would like you to analyze the webpage represented by the node ‘n{seed}’ using the GraphML data in the following two ways:

1. Content Summary and Context Analysis:
- Extract and summarize the textual content associated with ‘n{seed}’. Identify the main topics, keywords, or focus areas of this webpage (e.g., course information, faculty introduction, research description).
- Analyze the webpages that link to or are linked from ‘n{seed}’. Summarize the general themes or purposes of these neighboring pages and describe how ‘n{seed}’ fits within its local hyperlink structure (e.g., within a research group, course cluster, or student directory).

2. Category Classification:
- Based on both the textual and link-based context, classify whether the webpage represented by ‘n{seed}’ most likely belongs to one of the categories: {categories}.
- Justify your classification by referencing textual cues (e.g., titles, mentions, content focus) and structural patterns (e.g., connections to other similar pages).

Your analysis should be directly based on the data provided in the GraphML file and should be limited to 500 tokens. Focus exclusively on node ‘n{seed}’ and its immediate neighborhood within the web-based hyperlink network. The detailed GraphML web network data is as follows:
''',
'Cornell': '''
I have a GraphML file representing a web-based social network constructed from university computer science department webpages.
Each node denotes a webpage, the node features correspond to the textual content extracted from that page, and edges denote hyperlinks between pages. Reverse edges have been added to ensure bidirectional connectivity.
I would like you to analyze the webpage represented by the node ‘n{seed}’ using the GraphML data in the following two ways:

1. Content Summary and Context Analysis:
- Extract and summarize the textual content associated with ‘n{seed}’. Identify the main topics, keywords, or focus areas of this webpage (e.g., course information, faculty introduction, research description).
- Analyze the webpages that link to or are linked from ‘n{seed}’. Summarize the general themes or purposes of these neighboring pages and describe how ‘n{seed}’ fits within its local hyperlink structure (e.g., within a research group, course cluster, or student directory).

2. Category Classification:
- Based on both the textual and link-based context, classify whether the webpage represented by ‘n{seed}’ most likely belongs to one of the categories: {categories}.
- Justify your classification by referencing textual cues (e.g., titles, mentions, content focus) and structural patterns (e.g., connections to other similar pages).

Your analysis should be directly based on the data provided in the GraphML file and should be limited to 500 tokens. Focus exclusively on node ‘n{seed}’ and its immediate neighborhood within the web-based hyperlink network. The detailed GraphML web network data is as follows:
''',
'IMDB': '''
I am providing you with a GraphML file representing a heterogeneous graph focuses on online movies and television programs.
Each node contains a text attribute that describes its node type (movie, actor, or director) and the types of edges it has to other nodes.
The graph includes relations such as acting (actor–movie), directing (director–movie), and their reverse edges to ensure bidirectional connectivity.
Please analyze the node represented by ‘n{seed}’ using the provided GraphML data in the following two ways:

1. Node Summary and Context Analysis:
- Summarize the key structural and semantic properties of the node ‘n{seed}’ based on its text attribute, including its node type and the types of edges it has.
- For each edge type mentioned in the text, provide an overview of prevalent patterns or characteristics shared by the neighbors connected via that edge. Identify common genres, collaboration networks, or role patterns among these neighboring nodes (e.g., actors appearing together, directors frequently working with the same actors, movies sharing similar casts).

2. Category Classification (only for movie nodes):
- If ‘n{seed}’ is a movie node (as indicated in its text attribute), determine the specific genres from {categories} it primarily belongs to based on its connected actors and directors.
- Justify the classification by explaining which aspects of ‘n{seed}’ (its textual content, neighboring nodes, and connection patterns) align with recognized genres or thematic categories.
- If ‘n{seed}’ is not a movie node, skip this step.

Please ensure your analyses are grounded in the data provided by the GraphML file within 500 tokens, focusing on node ‘n{seed}’ and its immediate heterogeneous neighborhood. The detailed GraphML heterogeneous movie network data is as follows:
''',
'Photo': '''
I have a GraphML file representing an Amazon product co-purchasing network.
In this network, nodes represent photo-related products sold on Amazon, and edges indicate that two products are frequently purchased together. Node features are bag-of-words encoded product reviews, and class labels correspond to the product category.
I would like you to analyze the product represented by the node ‘n{seed}’ using the GraphML data in the following two ways:

1. Product Summary and Context Analysis:
- Extract and summarize the details of the product denoted by ‘n{seed}’, including its title, description, and relevant review features.
- Provide an overall summary of the prevalent themes or trends among the products that are co-purchased with ‘n{seed}’. Identify common threads, complementary items, or typical usage patterns shared by these neighboring products.

2. Category Classification:
- Using the information gathered from ‘n{seed}’ and its neighboring nodes, classify ‘n{seed}’ into one of the product categories: {categories}.
- Justify the classification by explaining which aspects of ‘n{seed}’ align with recognized prevalent themes, trends, or threads in the identified product category.

Your analysis should be directly based on the data provided in the GraphML file and should be limited to 500 tokens. Focus exclusively on node ‘n{seed}’ and its immediate co-purchased neighborhood. The detailed GraphML co-purchased network data is as follows:
''',
'Computers': '''
I have a GraphML file representing an Amazon product co-purchasing network.
In this network, nodes represent computer-related products sold on Amazon, and edges indicate that two products are frequently purchased together. Node features are bag-of-words encoded product reviews, and class labels correspond to the product category.
I would like you to analyze the product represented by the node ‘n{seed}’ using the GraphML data in the following two ways:

1. Product Summary and Context Analysis:
- Extract and summarize the details of the product denoted by ‘n{seed}’, including its title, description, and relevant review features.
- Provide an overall summary of the prevalent themes or trends among the products that are co-purchased with ‘n{seed}’. Identify common threads, complementary items, or typical usage patterns shared by these neighboring products.

2. Category Classification:
- Using the information gathered from ‘n{seed}’ and its neighboring nodes, classify ‘n{seed}’ into one of the product categories: {categories}.
- Justify the classification by explaining which aspects of ‘n{seed}’ align with recognized prevalent themes, trends, or threads in the identified product category.

Your analysis should be directly based on the data provided in the GraphML file and should be limited to 500 tokens. Focus exclusively on node ‘n{seed}’ and its immediate co-purchased neighborhood. The detailed GraphML co-purchased network data is as follows:
''',
'Amazon': '''
I am providing you with a GraphML file representing a heterogeneous Amazon product graph.
Each node contains a text attribute describing a product, and each node is of type 'product'.
Edges in the graph denote two types of relationships between products: 'also_bought' indicates that the two products are frequently purchased together, and 'also_viewed' indicates that the two products are frequently viewed together. Reverse edges have been added to ensure bidirectional connectivity.
Please analyze the node represented by 'n{seed}' using the provided GraphML data in the following two ways:

1. Node Summary and Context Analysis:
- Summarize the key properties of node 'n{seed}' based on its textual content and its connections to other products. Include product attributes, title, description, or review-derived features if available.
- For each edge type ('also_bought' and 'also_viewed'), provide an overview of prevalent patterns among the neighboring products connected via that edge. Identify complementary items, co-viewing trends, or common characteristics shared by these neighbors.

2. Category Classification:
- Based on the information summarized from 'n{seed}' and its neighbors, classify the product into its specific category within {categories}.
- Justify the classification by explaining which aspects of 'n{seed}' (its textual content, connected nodes, and neighbor edge patterns) align with recognized trends or common characteristics in the identified category.

Please ensure your analysis remains within 500 tokens and focuses on node 'n{seed}' and its immediate heterogeneous neighborhood. The detailed GraphML heterogeneous Amazon Electronics product network data is as follows:
''',
'Amazon-HeTGB': '''
I am providing you with a GraphML file representing a product co-purchasing network derived from the Amazon dataset.
Each node represents a product (e.g., book, music CD, DVD, or VHS tape), and edges indicate that two products are frequently bought together.
Node features include product descriptions synthesized from up to five earliest individual user ratings per product, along with product metadata such as name and category. Reverse edges have been added to ensure bidirectional connectivity.
Please analyze the product represented by node 'n{seed}' using the provided GraphML data in the following two ways:  

1. Product Summary and Context Analysis:
- Extract and summarize the textual content and metadata of the product denoted by 'n{seed}', including its description, name, and category if available.
- Provide an overview of the neighboring products connected to 'n{seed}' via co-purchasing edges. Identify common patterns, themes, or complementary items among these neighbors.

2. Rating Classification:
- Based on the information from 'n{seed}' and its neighboring nodes, predict the product's average rating class {categories}.
- Justify the classification by explaining which aspects of 'n{seed}' (its textual content, metadata, and neighbor connections) align with typical trends or patterns in the predicted rating class.

Please ensure your analysis is directly based on the GraphML data and limited to 500 tokens, focusing exclusively on node 'n{seed}' and its immediate co-purchased neighborhood. The detailed GraphML Amazon product network data is as follows:
''',
'HIV': '''
I am providing you with a GraphML file representing a molecular graph dataset (HIV).
Each graph corresponds to a compound molecule, where nodes denote atoms and edges denote chemical bonds. Node features include atom types and other chemical attributes, while edge types indicate bond types (single, double, etc.).
Please analyze the entire graph represented by 'G' using the provided GraphML data in the following two ways:

1. Graph Summary and Context Analysis:
- Summarize the overall molecular structure of graph 'G', including number and types of atoms, types of bonds, and notable substructures or motifs.
- Provide an overview of connectivity patterns or chemical features relevant to molecular properties, highlighting features that may contribute to HIV inhibition.

2. Graph Property Classification:
- Using the information from graph 'G', predict whether the compound is {categories} against HIV replication.
- Justify the prediction by explaining which structural features, atom types, or bond patterns in 'G' support the predicted activity.

Please ensure your analysis is directly based on the GraphML data and limited to 500 tokens. Focus on the entire molecular graph 'G'. The detailed GraphML molecular graph data is as follows:
''',
'COX2': '''
I am providing you with a GraphML file representing a molecular graph dataset of cyclooxygenase-2 (COX-2) inhibitors.
Each graph corresponds to a compound molecule, where nodes denote atoms and edges denote chemical bonds. Node features include atom types and other chemical attributes, while edge types indicate bond types (single, double, etc.).
Please analyze the entire graph represented by 'G' using the provided GraphML data in the following two ways:  

1. Graph Summary and Context Analysis:
- Summarize the overall molecular structure of graph 'G', including the number and types of atoms, types of bonds, and notable substructures or motifs.
- Provide an overview of connectivity patterns or chemical features relevant to COX-2 inhibitory activity, highlighting structural elements that may influence activity.

2. Graph Property Classification:
- Using the information from graph 'G', predict whether the compound is {categories} against COX-2 based on the pIC50 threshold of 6.5.
- Justify the prediction by explaining which structural features, atom types, or bond patterns in 'G' support the predicted activity.

Please ensure your analysis is directly based on the GraphML data and limited to 500 tokens. Focus on the entire molecular graph 'G'. The detailed GraphML molecular graph data is as follows:
''',
'PROTEINS': '''
I am providing you with a GraphML file representing a protein structure dataset.
Each graph corresponds to a protein, where nodes represent amino acids or structural units, and edges represent sequential or spatial connections between residues. Node features include amino acid types, secondary-structure content, surface properties, and other structural descriptors.
Please analyze the entire graph represented by 'G' using the provided GraphML data in the following two ways:

1. Graph Summary and Context Analysis:
- Summarize the overall structure of graph 'G', including number and types of amino acids, secondary structure composition, surface features, and notable structural motifs.
- Provide an overview of connectivity patterns, key structural properties, and any features relevant to protein function, highlighting aspects that may influence enzymatic activity.

2. Graph Property Classification:
- Using the information from graph 'G', predict whether the protein is an {categories}.
- Justify the prediction by explaining which structural features, amino acid types, secondary-structure content, or spatial motifs in 'G' support the predicted function.

Please ensure your analysis is directly based on the GraphML data and limited to 500 tokens. Focus on the entire protein graph 'G'. The detailed GraphML protein structure graph data is as follows:
''',
'ENZYMES': '''
I am providing you with a GraphML file representing a protein function prediction dataset.
Each graph corresponds to a protein molecule, where nodes represent amino acids or structural units, and edges represent either chemical bonds or spatial proximities between residues. Node features encode sequential, structural, and chemical information derived from the protein sequence and 3D structure.
Please analyze the entire graph represented by 'G' using the provided GraphML data in the following two ways:

1. Graph Summary and Context Analysis:
- Summarize the overall structure of graph 'G', including amino acid composition, structural motifs, chemical connectivity patterns, and any surface or interaction-related features.
- Identify key structural or chemical patterns that might relate to functional differentiation among proteins, such as catalytic sites, binding regions, or motifs commonly associated with enzymatic activity.

2. Graph Property Classification:
- Based on the structural and chemical information extracted from graph 'G', classify the protein into one of the six enzyme functional classes: {categories}.
- Justify the classification by explaining which features or patterns in 'G' align with known characteristics of the identified enzyme class.

Please ensure your analysis is directly based on the GraphML data and limited to 500 tokens. Focus on the entire protein graph 'G'. The detailed GraphML protein function graph data is as follows:
''',
'FB15K-237': '''
I am providing you with a GraphML file representing a knowledge graph constructed from Freebase.
Each node represents an entity with textual attributes including its name and description, while each directed edge represents a specific semantic relation type between two entities.
Please analyze the node represented by ‘n{seed}’ using the provided GraphML data in the following two ways:

1. Entity Summary and Context Analysis:
- Summarize the key semantic content of the entity denoted by ‘n{seed}’, including its name, description, and its role within the Freebase knowledge graph.
- Analyze its neighboring nodes to identify common types of relations or entity categories that help contextualize its meaning and semantic position in the graph.

2. Entity Classification:
‑ Determine the most probable semantic class or category from {categories} that ‘n{seed}’ belongs to, based on its textual description and the relational patterns observed in its neighbourhood.
‑ Justify the classification by explaining which textual or relational cues from ‘n{seed}’ and its connected entities align with the known semantic types in Freebase.

Please ensure your analysis is grounded in the data provided by the GraphML file within 500 tokens, focusing on node ‘n{seed}’ and its immediate relational neighborhood. The detailed GraphML knowledge graph data is as follows:
''',
'NELL': '''
I am providing you with a GraphML file representing a knowledge graph constructed from the Never-Ending Language Learning (NELL) knowledge base and the ClueWeb09 corpus.
Each node in the graph corresponds an entity extracted from the NELL knowledge base. Edges represent the connections derived from knowledge base triplets (e1, r, e2), where we add two directed edges (e1 → r1) and (e2 → r2).
Please analyze the node represented by ‘n{seed}’ using the provided GraphML data in the following two ways:

1. Entity Summary and Context Analysis:
- Summarize its key semantic content, including text-derived features or relational properties captured from the NELL knowledge base.
- Analyze its neighboring nodes to identify common types of relations or connected entity categories that provide contextual meaning within the knowledge graph.

2. Entity Classification:
- Determine the most probable semantic class from {categories} that it belongs to based on its text features and graph neighborhood.
- Justify the classification by explaining which textual or relational patterns in ‘n{seed}’ and its neighbors align with known semantic types or conceptual clusters.

Please ensure your analysis is grounded in the data provided by the GraphML file within 500 tokens, focusing on node ‘n{seed}’ and its immediate neighborhood. The detailed GraphML heterogeneous knowledge graph data is as follows:
''',
'Elliptic': '''
I am providing you with a GraphML file representing a temporal transaction graph constructed from the Bitcoin network.
Each node represents a transaction, and each directed edge indicates a payment flow of Bitcoin (BTC) from one transaction to another.
Please analyze the node represented by 'n{seed}' using the provided GraphML data in the following two ways:

1. Transaction Summary and Context Analysis:
- Summarize the key attributes of the transaction node 'n{seed}', including its temporal behavior, structural connections, and feature characteristics.
- Examine the neighboring transactions linked through Bitcoin flows to identify typical interaction patterns or communities (e.g., clusters of exchanges, wallets, or suspicious chains of payments).

2. Transaction Legitimacy Classification:
- Based on the features and transaction flow patterns of 'n{seed}' and its neighboring nodes, determine whether 'n{seed}' is more likely to be a {categories} transaction.
- Justify your reasoning by referring to patterns such as transaction timing, connectivity motifs, or feature anomalies observed in the GraphML data.

Please ensure your analysis is grounded in the data provided by the GraphML file and limited to 500 tokens, focusing on node 'n{seed}' and its immediate temporal transaction neighborhood. The detailed GraphML temporal transaction graph data is as follows:
''',
}
