# This code is adapted from <https://github.com/terrierteam/pyterrier_t5/blob/master/pyterrier_t5/__init__.py>
import pyterrier as pt
import pandas as pd
from pyterrier.model import add_ranks
import torch
from torch.nn import functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer
from pyterrier.transformer import TransformerBase

class MonoT5(TransformerBase):
    def __init__(self,
                 model='castorini/monot5-base-msmarco',
                 batch_size=4,
                 text_field='text',
                 device=None,
                 verbose=True):
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model_name = model
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
        self.text_field = text_field
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]
        self.doc_maxlen = None
        self.doc_prompt = None
        self.n_positions = 512
        if hasattr(self.model.config, 'prompt'):
            if 'rel' in self.model.config.prompt:
                self.REL = self.tokenizer.encode(self.model.config.prompt['rel'])[0]
            if 'nrel' in self.model.config.prompt:
                self.NREL = self.tokenizer.encode(self.model.config.prompt['nrel'])[0]
            if 'maxlen' in self.model.config.prompt:
                self.doc_maxlen = self.model.config.prompt['maxlen']
            if 'document' in self.model.config.prompt:
                self.doc_prompt = self.model.config.prompt['document']
        if hasattr(self.model.config, 'n_positions'):
            self.n_positions = self.model.config.n_positions

    def __str__(self):
        return f"MonoT5({self.model_name})"

    def transform(self, run):
        scores = []
        queries = run['query']
        if self.doc_prompt is None:
            texts = run[self.text_field]
        else:
            texts = run.apply(lambda x: self.doc_prompt.format(**x)[:self.doc_maxlen], axis=1)
        it = range(0, len(queries), self.batch_size)
        prompts = self.tokenizer.batch_encode_plus([f'Relevant:' for _ in range(self.batch_size)], return_tensors='pt', padding='longest')
        max_vlen = self.n_positions - prompts['input_ids'].shape[1]
        if self.verbose:
            it = pt.tqdm(it, desc='monoT5', unit='batches')
        for start_idx in it:
            rng = slice(start_idx, start_idx+self.batch_size) # same as start_idx:start_idx+self.batch_size
            enc = self.tokenizer.batch_encode_plus([f'Query: {q} Document: {d}' for q, d in zip(queries[rng], texts[rng])], return_tensors='pt', padding='longest')
            for key, enc_value in list(enc.items()):
                enc_value = enc_value[:, :-1] # chop off end of sequence token-- this will be added with the prompt
                enc_value = enc_value[:, :max_vlen] # truncate any tokens that will not fit once the prompt is added
                enc[key] = torch.cat([enc_value, prompts[key][:enc_value.shape[0]]], dim=1) # add in the prompt to the end
            enc['decoder_input_ids'] = torch.full(
                (len(queries[rng]), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                result = self.model(**enc).logits
            result = result[:, 0, (self.REL, self.NREL)]
            scores += F.log_softmax(result, dim=1)[:, 0].cpu().detach().tolist()
        run = run.drop(columns=['score', 'rank'], errors='ignore').assign(score=scores)
        run = add_ranks(run)
        return run
