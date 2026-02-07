from GraphCLIP.data.load import load_data
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from GraphCLIP.utils.args import Arguments
from torch_geometric import seed_everything
from torch_geometric.utils import to_undirected, remove_self_loops


seed_everything(88) # 88

config = Arguments().parse_args()


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
text_ids = {
    'qwen2.5-0.5b': "../Qwen2.5-0.5B-Instruct",
    'qwen2.5-1.5b': "../Qwen2.5-1.5B-Instruct",
    'sbert': 'sentence-transformers/multi-qa-distilbert-cos-v1',
    'tiny': 'sentence-transformers/all-MiniLM-L6-v2', # use this one
    'e5': 'intfloat/e5-base-v2'
}
model_id = text_ids[config.lm_type]
model = AutoModel.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

dataset_name = config.dataset
data, text, num_classes, _ = load_data(dataset_name)
print(f"encoding {len(text)} nodes")
del data 
chunk_size=1024
# iters_cnt = len(text) // chunk_size
all_embs = []
for i in tqdm(range(0, len(text), chunk_size)):
    if i + chunk_size >= len(text):
        batch = text[i:]
    else:
        batch = text[i:i+chunk_size]
    batch_t = tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=512)
    batch_t = batch_t.to(device)
with torch.no_grad():
    text_output = model(**batch_t)

text_embs = mean_pooling(text_output.last_hidden_state, batch_t["attention_mask"])
# text_embs = text_output[0][:,0,:].squeeze(1) # use CLS token
all_embs.append(text_embs.cpu())
del text
all_embs = torch.cat(all_embs)
# torch.save(all_embs, f'{dataset_name}-{config.lm_type}_x.pt')
data.x=all_embs
edges,_= remove_self_loops(to_undirected(data.edge_index))
data.edge_index=edges
torch.save(data, f'./processed_data/{config.dataset}.pt')