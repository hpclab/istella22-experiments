# Experiments on the Istella22 Dataset

# LambdaMART

Three LambdaMART models are available in this repository. The experiments published in the Istella22 resource paper can be reproduced by following the Jupyter Notebook `evaluation`.

Additional features to build the MonoT5 SVM files can be found in the `lambdamart/data` subdirectory. The features should be pasted to the Istella22 official `test.svm` file to build the final test file for the MonoT5 and MonoT5 (Title + Url + Text) versions. Final test files can be produced using the `paste` command.

```bash
paste -d' ' test.svm monoT5.feature.svm > test.monoT5.svm
paste -d' ' test.svm monoT5.titleUrlText.svm > test.monoT5.titleUrlText.svm
```

Both models and features are gzipped. Unzip them before use.

# MonoT5

MonoT5 models are available on Huggingface as [macavaney/it5-base-istella-title_url_text](https://huggingface.co/macavaney/it5-base-istella-title_url_text) and
[macavaney/it5-base-istella-title_url](https://huggingface.co/macavaney/it5-base-istella-title_url).

You can use them using the `MonoT5` transformer included in this package. Example:

```python
import pandas as pd
import pyterrier as pt ; pt.init()
from monot5 import MonoT5

model = MonoT5('macavaney/it5-base-istella-title_url')
input = pd.DataFrame([{'qid': '1', 'query': 'test', 'title': 'test document', 'url': 'https://test.com/'}])
model(input)
# qid query          title                url     score  rank
#   1  test  test document  https://test.com/ -0.005107     0
```

You can run experiments for both models with:
```bash
python run_monot5.py
```
