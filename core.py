from __future__ import annotations
import json, math, copy
from dataclasses import dataclass,asdict
import numpy as np,pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score,f1_score,classification_report,confusion_matrix,silhouette_score
from sklearn.model_selection import KFold,GroupKFold,StratifiedGroupKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler,normalize
from sklearn.svm import SVC
META_DEFAULT=['SampleID','Brand','Grade','City','State','Year']
@dataclass
class Step:name:str;params:dict;enabled:bool=True
DEFAULTS={'Range':{'minimum':400.0,'maximum':4000.0},'Exclude':{'ranges':''},'Baseline polynomial':{'order':2},'Savitzky-Golay':{'window':11,'poly':2},'Derivative':{'order':1,'window':11,'poly':2},'SNV':{},'Vector normalize':{},'Area normalize':{},'Mean center':{},'Autoscale':{},'Robust scale':{}}
def load_ftir(path):
 r=pd.read_csv(path,low_memory=False); marker=r.iloc[0,0] if len(r) else None
 if isinstance(marker,str) and 'wavenumber' in marker.lower():
  start=7;wn=pd.to_numeric(r.iloc[0,start:],errors='coerce').to_numpy(float);m=r.iloc[1:,1:start].copy().reset_index(drop=True);m.columns=META_DEFAULT[:m.shape[1]];x=r.iloc[1:,start:].apply(pd.to_numeric,errors='coerce').reset_index(drop=True)
 else:
  nums=[]
  for c in r.columns:
   try:nums.append((c,float(c)))
   except (TypeError,ValueError):pass
  if not nums:raise ValueError('No numeric wavenumber columns detected')
  cols=[c for c,_ in nums];wn=np.array([v for _,v in nums]);x=r[cols].apply(pd.to_numeric,errors='coerce');m=r.drop(columns=cols)
 good=np.isfinite(wn);order=np.argsort(wn[good]);return m.reset_index(drop=True),wn[good][order],x.loc[:,good].iloc[:,order].reset_index(drop=True)
def parse_ranges(text):
 out=[]
 for t in str(text).replace(';',',').split(','):
  if t.strip():a,b=map(float,t.strip().replace('–','-').split('-'));out.append((min(a,b),max(a,b)))
 return out
def snv(x):
 d=x.std(1,keepdims=True);d[d==0]=1;return (x-x.mean(1,keepdims=True))/d
def apply_recipe(X,wn,steps,state=None):
 x=np.asarray(X,float).copy();w=np.asarray(wn,float).copy();fitting=state is None;state={} if state is None else state
 for i,s in enumerate(steps):
  if not s.enabled:continue
  n,p=s.name,s.params
  if n=='Range':mask=(w>=float(p['minimum']))&(w<=float(p['maximum']));x=x[:,mask];w=w[mask]
  elif n=='Exclude':
   mask=np.ones(len(w),bool)
   for a,b in parse_ranges(p.get('ranges','')):mask&=~((w>=a)&(w<=b))
   x=x[:,mask];w=w[mask]
  elif n=='Baseline polynomial':
   g=np.linspace(-1,1,x.shape[1]);x=np.array([r-np.polyval(np.polyfit(g,r,int(p['order'])),g) for r in x])
  elif n in ('Savitzky-Golay','Derivative'):
   poly=int(p['poly']);win=max(int(p['window']),poly+2);win+=win%2==0;win=min(win,x.shape[1] if x.shape[1]%2 else x.shape[1]-1);x=savgol_filter(x,win,poly,deriv=int(p.get('order',0)) if n=='Derivative' else 0,axis=1)
  elif n=='SNV':x=snv(x)
  elif n=='Vector normalize':x=normalize(x)
  elif n=='Area normalize':a=np.trapz(abs(x),w,axis=1);a[a==0]=1;x=x/a[:,None]
  elif n in ('Mean center','Autoscale','Robust scale'):
   if fitting:sc=RobustScaler() if n=='Robust scale' else StandardScaler(with_std=n=='Autoscale');sc.fit(x);state[i]=sc
   x=state[i].transform(x)
 if x.shape[1]<2:raise ValueError('Recipe leaves fewer than 2 spectral variables')
 return x,w,state
class RecipeTransformer(BaseEstimator,TransformerMixin):
 def __init__(self,steps):self.steps=steps
 def fit(self,X,y=None):self.imp_=SimpleImputer(strategy='median').fit(X);_,_,self.state_=apply_recipe(self.imp_.transform(X),np.arange(X.shape[1]),self.steps);return self
 def transform(self,X):return apply_recipe(self.imp_.transform(X),np.arange(X.shape[1]),self.steps,self.state_)[0]
def exploratory_pca(X,wn,steps,components=20):
 z,w,_=apply_recipe(SimpleImputer(strategy='median').fit_transform(X),wn,steps);n=min(components,len(z)-1,z.shape[1]);p=PCA(n).fit(z);return {'processed':z,'wn':w,'scores':p.transform(z),'loadings':p.components_.T,'variance':p.explained_variance_ratio_,'pca':p}
def pca_cv(X,steps,max_components=30,folds=5,groups=None):
 X=np.asarray(X);nmax=min(max_components,len(X)-2,X.shape[1]);splitter=GroupKFold(min(folds,len(np.unique(groups)))) if groups is not None else KFold(min(folds,len(X)),shuffle=True,random_state=42);splits=list(splitter.split(X,groups=groups) if groups is not None else splitter.split(X));rows=[]
 for n in range(1,nmax+1):
  cal=[];val=[];press=tss=0
  for tr,te in splits:
   prep=RecipeTransformer(steps).fit(X[tr]);a=prep.transform(X[tr]);b=prep.transform(X[te]);k=min(n,len(tr)-1,a.shape[1]);p=PCA(k).fit(a);rc=a-p.inverse_transform(p.transform(a));rv=b-p.inverse_transform(p.transform(b));cal.append(np.mean(rc**2));val.append(np.mean(rv**2));press+=np.sum(rv**2);tss+=np.sum((b-a.mean(0))**2)
  rows.append({'components':n,'RMSEC_X':math.sqrt(np.mean(cal)),'RMSECV_X':math.sqrt(np.mean(val)),'PRESS_X':press,'Q2_X':1-press/tss if tss else np.nan})
 df=pd.DataFrame(rows);best=int(df.loc[df.RMSECV_X.idxmin(),'components']);return df,best
def embedding(X,method,components=2,**kw):
 if method=='PCA':return PCA(components).fit_transform(X)
 try:import umap
 except ImportError as e:raise RuntimeError('UMAP unavailable. Install umap-learn.') from e
 return umap.UMAP(n_components=components,n_neighbors=kw.get('n_neighbors',15),min_dist=kw.get('min_dist',.1),metric=kw.get('metric','euclidean'),random_state=kw.get('random_state',42)).fit_transform(X)
def cluster(z,method,k=3,eps=.8,min_samples=3):
 if method=='KMeans':lab=KMeans(k,n_init=30,random_state=42).fit_predict(z)
 elif method=='Agglomerative':lab=AgglomerativeClustering(k).fit_predict(z)
 else:lab=DBSCAN(eps=eps,min_samples=min_samples).fit_predict(z)
 ok=1<len(set(lab))<len(lab);return lab,silhouette_score(z,lab) if ok else np.nan
def kmeans_history(z,k,max_iter=30):
 rng=np.random.default_rng(42);cent=z[rng.choice(len(z),k,replace=False)].copy();frames=[]
 for i in range(max_iter):
  lab=((z[:,None,:]-cent[None,:,:])**2).sum(2).argmin(1);frames.append({'iteration':i,'labels':lab.copy(),'centers':cent.copy()});new=np.array([z[lab==j].mean(0) if np.any(lab==j) else cent[j] for j in range(k)])
  if np.allclose(new,cent):break
  cent=new
 return frames
def make_groups(meta,cols):return meta[cols].fillna('<missing>').astype(str).agg('|'.join,axis=1).to_numpy()
def eligibility(meta,label,cols,min_groups):
 y=meta[label].astype(str).to_numpy();g=make_groups(meta,cols);c=pd.DataFrame({'label':y,'group':g}).drop_duplicates().groupby('label').size();return pd.DataFrame({'class':c.index,'independent_groups':c.values,'eligible':c.values>=min_groups}),y,g
def candidate(rng,maxpcs,base):
 s=copy.deepcopy(base);fixed=[x for x in s if x.name in ('Range','Exclude')];t=[x for x in s if x.name not in ('Range','Exclude')];rng.shuffle(t);s=fixed+t
 for x in s:
  if x.name in ('Savitzky-Golay','Derivative'):x.params['window']=[7,11,15,21][int(rng.integers(4))]
  if x.name=='Derivative':x.params['order']=int(rng.integers(1,3))
 model=['SVM','Random Forest'][int(rng.integers(2))];p={'model':model,'use_pca':bool(rng.integers(2)),'pcs':int(rng.integers(2,maxpcs+1))}
 if model=='SVM':opts=['scale',.001,.01,.1];p.update(C=float(10**rng.uniform(-2,2)),kernel=['linear','rbf'][int(rng.integers(2))],gamma=opts[int(rng.integers(len(opts)))])
 else:p.update(trees=[200,400][int(rng.integers(2))],depth=[None,5,10,20][int(rng.integers(4))],leaf=int(rng.integers(1,5)))
 return s,p
def pipe(steps,p,max_pc_fit=None):
 chain=[('prep',RecipeTransformer(steps))]
 if p['model']=='SVM':chain.append(('scale',StandardScaler()))
 if p['use_pca']:chain.append(('pca',PCA(p['pcs'],svd_solver='randomized',random_state=42)))
 clf=SVC(C=p['C'],kernel=p['kernel'],gamma=p['gamma'],class_weight='balanced') if p['model']=='SVM' else RandomForestClassifier(n_estimators=p['trees'],max_depth=p['depth'],min_samples_leaf=p['leaf'],class_weight='balanced',n_jobs=1,random_state=42);chain.append(('clf',clf));return Pipeline(chain)
def search(X,y,g,base,trials,maxpcs,cv,seed):
 rng=np.random.default_rng(seed);hist=[];best=None
 for i in range(trials):
  s,p=candidate(rng,maxpcs,base)
  try:
   vals=cross_val_score(pipe(s,p),X,y,groups=g,cv=cv,scoring='balanced_accuracy',error_score='raise',n_jobs=1);score=float(vals.mean());std=float(vals.std());obj=score-.25*std-.002*(p['pcs'] if p['use_pca'] else 0);hist.append({'trial':i,'status':'complete','score':score,'std':std,'objective':obj,'recipe':json.dumps([asdict(x) for x in s]),**p})
   if best is None or obj>best[0]:best=(obj,s,p)
  except Exception as e:hist.append({'trial':i,'status':'failed','error':f'{type(e).__name__}: {e}','recipe':json.dumps([asdict(x) for x in s]),**p})
 if best is None:raise RuntimeError('All trials failed')
 return best[1],best[2],pd.DataFrame(hist)
def nested_optimize(X,y,g,base,trials=20,maxpcs=20,outer_folds=3,inner_folds=2):
 outer=StratifiedGroupKFold(outer_folds,shuffle=True,random_state=42);pred=np.empty(len(y),object);folds=[];history=[];best_rows=[]
 for f,(tr,te) in enumerate(outer.split(X,y,g),1):
  cv=StratifiedGroupKFold(inner_folds,shuffle=True,random_state=100+f);s,p,h=search(X[tr],y[tr],g[tr],base,trials,min(maxpcs,len(tr)-1),cv,200+f);m=pipe(s,p).fit(X[tr],y[tr]);q=m.predict(X[te]);pred[te]=q;h['outer_fold']=f;history.append(h);best_rows.append({'fold':f,'recipe':json.dumps([asdict(x) for x in s]),**p});folds.append({'fold':f,'balanced_accuracy':balanced_accuracy_score(y[te],q),'macro_f1':f1_score(y[te],q,average='macro',zero_division=0)})
 classes=np.unique(y);return {'pred':pred,'folds':pd.DataFrame(folds),'history':pd.concat(history,ignore_index=True),'best':pd.DataFrame(best_rows),'classes':classes,'cm':confusion_matrix(y,pred,labels=classes),'report':pd.DataFrame(classification_report(y,pred,output_dict=True,zero_division=0)).T,'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro',zero_division=0)}
