import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from core import Step, apply_recipe, pca_cv, make_groups, eligibility, search

def sample_data():
    rng=np.random.default_rng(1); wn=np.linspace(400,1800,80); meta=[]; X=[]
    for brand,shift in [('A',0.0),('B',0.7)]:
        for site in range(4):
            for rep in range(3):
                meta.append({'Brand':brand,'City':f'C{site}','State':'CA'})
                X.append(np.sin(wn/90)+shift+rng.normal(0,.03,len(wn)))
    return pd.DataFrame(meta),wn,np.asarray(X)

def test_preprocessing_order_changes_result():
    _,wn,X=sample_data()
    a=[Step('Derivative',{'order':1,'window':11,'poly':2}),Step('SNV',{})]
    b=list(reversed(a))
    assert not np.allclose(apply_recipe(X,wn,a)[0],apply_recipe(X,wn,b)[0])

def test_pca_cv_returns_finite_curve():
    m,_,X=sample_data(); groups=make_groups(m,['City','State'])
    result,best=pca_cv(X,[Step('SNV',{})],6,2,groups)
    assert len(result)==6 and 1<=best<=6 and np.isfinite(result.RMSECV_X).all()

def test_eligibility_counts_independent_groups():
    m,_,_=sample_data(); result,_,_=eligibility(m,'Brand',['City','State'],3)
    assert result.eligible.all()

def test_search_completes_and_gamma_type_is_valid():
    m,_,X=sample_data(); y=m.Brand.to_numpy(); groups=make_groups(m,['City','State'])
    cv=StratifiedGroupKFold(2,shuffle=True,random_state=1)
    _,params,history=search(X,y,groups,[Step('SNV',{})],4,5,cv,1)
    assert (history.status=='complete').any()
    assert params.get('gamma','scale')=='scale' or isinstance(params.get('gamma'),float)
