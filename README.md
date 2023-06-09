# HawkesProcess
## HTNE
run HTNE.py
```
    hp = HP(data_name='dblp', model_name='htne')
    hp.train()
```
## HTNE_a
run HTNE.py
```
    hp = HP(data_name='dblp', model_name='htne_attn')
    hp.train()
```
## BI
run HTNE.py
```
    hp = HP(data_name='dblp', model_name='bi')
    hp.train()
```
## NIS
run NIS.py
```
    nis = NIS(data_name='dblp', optim='Adam')
    nis.train()
```