import torch
import torch.nn as nn
import torch.nn.functional as F
from  TS_encoder import PatchTSTEncoder
from  transformers import AutoModelForCausalLM,AutoTokenizer
from ts_dataloader import ts_textual,collate_func
import os
import sys
import numpy as np
from torch.utils.data import Dataset,DataLoader
from modules.conv_module import ConvFeatureExtraction
from modules.transformer_enc import PatchTSTEncoder
from modules.ts_encoder import llm_projection

device ='cuda' if torch.cuda.is_available() else 'cpu'

##loading the base LLM model and tokenizer
model_name='microsoft/Phi-4-mini-reasoning'
model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True)
tokenizer =AutoTokenizer.from_pretrained(model_name,local_files_only=True)
model_dtype=next(model.parameters()).dtype

## to expand the tokenizer to add the special tokens <ts> <ts/>
special_token_dict={'pad_token':"<|pad|>","additional_special_tokens":['<ts>','<ts/>']}
tokenizer.add_special_tokens(special_token_dict)
model.resize_token_embeddings(len(tokenizer))

##dataset fetching
import json
_json_file = os.path.join(os.environ["SLURM_TMPDIR"],"align_256.jsonl")

###datapipeline
dataset=ts_textual(128,128,tokenizer,_json_file,device=device)
dataloader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer,device=device))
"""
dataset= ts_multimodal_text(128,128,_json_file,tokenizer,device=device,model_dtype=None)
dataloader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer,device=device))"""

class LLM_wrapper(nn.Module):
    def __init__(self,tokenizer,conv_layers,patch_len,llm_model,device=device):
        super().__init__()
        self.tokenizer=tokenizer
        self.llm_model=llm_model
        self.embed_size=llm_model.config.hidden_size
        """self.max_patches=max_patches
        self.max_channel=max_channel"""
        self.P=patch_len
        self.device=device
        self.conv_layers=conv_layers
        
        self.ts_conv_module=ConvFeatureExtraction(self.conv_layers,dropout=0.1)
        self.ts_transformer=PatchTSTEncoder(patch_len=self.P,n_layers=2,d_model=512,n_heads=4,
                                shared_embedding=True,d_ff=1024,norm='Layer',attn_dropout=0.,dropout=0.1,activation='gelu',store_attn=False,res_attention=False,pre_norm=True,pe='zeros',learn_pe=True,verbose=False)
        self.ts_encoder = llm_projection(self.ts_conv_module,64,self.ts_transformer,512,1024,3072)
        self.ts_encoder.to(self.device)
        
    def assemble_input_embeds(self,input_ids,ts_embeddings,ts_token_idx,text_token_idx,ts_pairs:torch.tensor):
        ###logic to assemble textual and ts_tokens 
        assemb_embed_tensor=[]
        channels=ts_pairs.shape[1]
        bs=ts_embeddings.shape[0]
        c_in=ts_embeddings.shape[1]
        assert c_in==channels
        num_ts_tokens=ts_embeddings.shape[2]
        
        ts_embeddings=ts_embeddings.view(bs*c_in,num_ts_tokens,-1)
        ts_emb_dim=ts_embeddings.shape[3]
        
        input_embeds=self.llm_model.get_input_embeddings()(input_ids) ##[bs,seq_len,d_emb]
        input_embeds.requires_grad_(requires_grad=True)
        text_emb_dim= input_embeds.shape[2]

        assert (ts_emb_dim==text_emb_dim)
        T_new=ts_token_idx.shape[1]+text_token_idx.shape[1]
        ts_container =torch.zeros((T_new,text_emb_dim),device=self.device) ### total_idx,total_idx
        text_container=torch.zeros((T_new,text_emb_dim),device=self.device)
        flat_ts_embeddings=ts_embeddings.view(-1,c_in*num_ts_tokens,ts_emb_dim)
        flat_ts_embeddings=flat_ts_embeddings.squeeze(0)
        flat_text_embeddings=input_embeds.squeeze(0)
        
        ##get the indices after the <ts>....<ts/> placeholder is offseted
        ts_indices=ts_token_idx.squeeze(0).view(-1,1)
        ts_indices=ts_indices.expand(-1,text_emb_dim)
        text_indices=text_token_idx.squeeze(0).view(-1,1)
        text_indices=text_indices.expand(-1,text_emb_dim)
        ##print(idx.shape)
        ###print(idx_expanded)
        ts_embeds_assemb= ts_container.scatter(dim=0,index=ts_indices,src=flat_ts_embeddings)
        text_embeds_assemb=text_container.scatter(dim=0,index=text_indices,src=flat_text_embeddings)
        final_tensor=ts_embeds_assemb+text_embeds_assemb
        assemb_embed_tensor.append(final_tensor)
        
        return torch.stack(assemb_embed_tensor)

    def forward(self,input_ids=None,ts_input=None,ts_pairs=None,ts_idx=None,text_idx=None,attention_mask=None,labels=None,):
        ##convert the ts_patches into ts_embeddings
        ts_tensor = ts_input.view(-1,self.max_patches,self.max_channel,self.P).to(self.device)  ## (bs,N,c_in,P)
        ts_embedding = self.ts_encoder(ts_tensor.to(self.device)) ## (bs,n_vars,num_patch,d_model)
        
        ##slicing
        ##ts_embedding_sliced =ts_embedding[ts_masks] ##flattened ts_embeddings
        input_embeddings= self.assemble_input_embeds(input_ids,ts_embedding,ts_idx,text_idx,ts_pairs)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        output= self.llm_model(inputs_embeds=input_embeddings,attention_mask=attention_mask,labels=labels)
        
        return output,input_embeddings
    
from tqdm import tqdm
conv_layers=[(128,5,1),(64,3,1)]
model_wrapper=LLM_wrapper(tokenizer,conv_layers,128,model,device=device)
model_wrapper.train()
model_wrapper.to(device)

##** freeze the LLM for stage-1 training
for p in model_wrapper.llm_model.parameters():
    p.requires_grad=False
for p in model_wrapper.llm_model.get_input_embeddings().parameters():
    p.requires_grad = True
for p in model_wrapper.ts_encoder.parameters():
    p.requires_grad = True
    
all_params = (list(model_wrapper.ts_encoder.parameters())+list(model_wrapper.llm_model.get_input_embeddings().parameters()))
optimizer = torch.optim.AdamW(all_params, lr=1e-5)
epoch_losses=[]
for epoch in range(1):  ##1 epochs
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    num_batches = 0
    running_loss=0
    epoch_loss=0
    ctr=0
    for batch in pbar:
        input_ids=batch['input_ids'].to(device) ## input and output
        labels_batch=batch['labels'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        ts_input=batch['time_series'].to(device) ### batch of patchified padded ts_inputs (bs,c_in,N,p)
        ts_pairs=batch['ts_pairs'].to(device)
        ts_indices=batch["ts_indices"].to(device)
        textual_indices=batch['textual_indices'].to(device)
        ###ts_mask = batch['ts_mask'].to(device)

        ##model_wrapper=LLM_wrapper(tokenizer,ts_input,model,device=device)
        outputs,_= model_wrapper(input_ids=input_ids,ts_input=ts_input,ts_pairs=ts_pairs,ts_idx=ts_indices,text_idx=textual_indices,attention_mask=attention_mask,labels=labels_batch,)
        loss=outputs.loss
        loss.backward()                     ##gradient calculation
        running_loss+=loss.item()
        num_batches+=1
        optimizer.step()
        ###gradient checking
        optimizer.zero_grad()
        pbar.set_postfix(loss=loss.item())
        epoch_loss=running_loss/num_batches
        epoch_losses.append(epoch_loss)
        ctr+=1

### save the plot
out_path = os.path.join(os.environ["SLURM_TMPDIR"], "training_loss_MTS.png")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.figure(figsize=(8, 5))
plt.plot(ctr, epoch_losses, marker='o')
plt.title("Training Loss Trend Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.savefig(out_path)