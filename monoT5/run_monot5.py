import os
import ir_datasets
import pyterrier as pt
pt.init()
from pyterrier.measures import *
from monot5 import MonoT5

dataset = pt.get_dataset('irds:istella22/test')

if not os.path.exists('runs/monot5.titleurltext.res.gz'):
  pipeline = pt.text.get_text(dataset, ['title', 'url', 'text']) >> MonoT5('macavaney/it5-base-istella-title_url_text')
  res_tut = pipeline(pt.io.read_results('runs/initial.res.gz', dataset=dataset))
  pt.io.write_results(res_tut, 'runs/monot5.titleurltext.res.gz')
else:
  res_tut = pt.io.read_results('runs/monot5.titleurltext.res.gz')

if not os.path.exists('runs/monot5.titleurl.res.gz'):
  pipeline = pt.text.get_text(dataset, ['title', 'url']) >> MonoT5('macavaney/it5-base-istella-title_url')
  res_tu = pipeline(pt.io.read_results('runs/initial.res.gz', dataset=dataset))

  pt.io.write_results(res_tu, 'runs/monot5.titleurl.res.gz')
else:
  res_tu = pt.io.read_results('runs/monot5.titleurl.res.gz')

print(pt.Experiment(
    [res_tu, res_tut],
    dataset.get_topics(),
    dataset.get_qrels(),
    [P@1, P@5, P@10, nDCG(dcg='exp-log2')@10, nDCG(dcg='exp-log2')@20, RR, AP],
    names=['title+url', 'title+url+text'],
    round=4
  ))
