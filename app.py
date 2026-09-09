import sys,json,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np,pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import *
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from core import *
class Plot(Canvas):
 def __init__(self):self.fig=Figure(figsize=(8,5),tight_layout=True);super().__init__(self.fig)
class Main(QMainWindow):
 def __init__(self):super().__init__();self.setWindowTitle('FTIR Workbench v5');self.resize(1550,980);self.data=None;self.steps=[Step('Range',dict(DEFAULTS['Range'])),Step('Savitzky-Golay',dict(DEFAULTS['Savitzky-Golay'])),Step('SNV',{})];self.last=None;self.opt=None;self.build()
 def build(self):
  w=QWidget();self.setCentralWidget(w);v=QVBoxLayout(w);r=QHBoxLayout();self.path=QLineEdit();b=QPushButton('Load CSV');b.clicked.connect(self.load);r.addWidget(self.path);r.addWidget(b);v.addLayout(r);self.tabs=QTabWidget();v.addWidget(self.tabs,1)
  self.info=QPlainTextEdit();self.info.setReadOnly(True);self.tabs.addTab(self.info,'Data & QC')
  p=QWidget();pv=QVBoxLayout(p);self.lst=QListWidget();self.lst.setDragDropMode(QAbstractItemView.InternalMove);pv.addWidget(QLabel('Drag to reorder. Add any combination.'));pv.addWidget(self.lst);rr=QHBoxLayout();self.add=QComboBox();self.add.addItems(DEFAULTS);[rr.addWidget(x) for x in [self.add]];[(lambda bt,fn:(bt.clicked.connect(fn),rr.addWidget(bt)))(QPushButton(t),f) for t,f in [('Add',self.add_step),('Remove',self.remove_step),('Edit',self.edit_step),('Save recipe',self.save_recipe),('Load recipe',self.load_recipe)]];pv.addLayout(rr);self.tabs.addTab(p,'Preprocessing');self.refresh()
  ex=QWidget();ev=QVBoxLayout(ex);er=QHBoxLayout();self.method=QComboBox();self.method.addItems(['PCA','UMAP']);self.dim=QComboBox();self.dim.addItems(['2D','3D']);self.pc1=QSpinBox();self.pc2=QSpinBox();self.pc3=QSpinBox();[x.setRange(1,50) for x in [self.pc1,self.pc2,self.pc3]];self.pc1.setValue(1);self.pc2.setValue(2);self.pc3.setValue(3);self.npcs=QSpinBox();self.npcs.setRange(3,50);self.npcs.setValue(20);self.neigh=QSpinBox();self.neigh.setRange(2,200);self.neigh.setValue(15);self.mind=QDoubleSpinBox();self.mind.setRange(0,1);self.mind.setSingleStep(.05);self.mind.setValue(.1);run=QPushButton('Run projection');run.clicked.connect(self.project);cv=QPushButton('PCA CV');cv.clicked.connect(self.run_cv)
  for a in [self.method,self.dim,QLabel('X/PC'),self.pc1,QLabel('Y/PC'),self.pc2,QLabel('Z/PC'),self.pc3,QLabel('n_neighbors'),self.neigh,QLabel('min_dist'),self.mind,run,cv]:er.addWidget(a)
  ev.addLayout(er);self.et=QTabWidget();self.raw=Plot();self.proc=Plot();self.proj=Plot();self.loadings_plot=Plot();self.var=Plot();self.cv_plot=Plot();[self.et.addTab(a,b) for a,b in [(self.raw,'Raw'),(self.proc,'Processed'),(self.proj,'2D/3D projection'),(self.loadings_plot,'Loadings'),(self.var,'Variance'),(self.cv_plot,'PCA CV')]];ev.addWidget(self.et);cr=QHBoxLayout();self.cl=QComboBox();self.cl.addItems(['KMeans','Agglomerative','DBSCAN']);self.k=QSpinBox();self.k.setRange(2,20);self.k.setValue(3);cb=QPushButton('Cluster');cb.clicked.connect(self.do_cluster);self.iter=QSlider(Qt.Horizontal);self.iter.setRange(0,0);self.iter.valueChanged.connect(self.replay_cluster);[cr.addWidget(x) for x in [self.cl,self.k,cb,QLabel('K-means iteration'),self.iter]];ev.addLayout(cr);self.tabs.addTab(ex,'Exploratory Chemometrics')
  md=QWidget();mv=QVBoxLayout(md);g=QGridLayout();self.label=QLineEdit('Brand');self.groups=QLineEdit('City,State');self.minG=QSpinBox();self.minG.setRange(2,20);self.minG.setValue(3);self.outer=QSpinBox();self.outer.setRange(2,10);self.outer.setValue(3);self.inner=QSpinBox();self.inner.setRange(2,10);self.inner.setValue(2);self.trials=QSpinBox();self.trials.setRange(1,1000);self.trials.setValue(20);self.maxpc=QSpinBox();self.maxpc.setRange(2,100);self.maxpc.setValue(20)
  for i,(n,x) in enumerate([('Label',self.label),('Group columns',self.groups),('Min groups/class',self.minG),('Outer folds',self.outer),('Inner folds',self.inner),('Trials/fold',self.trials),('Max PCs',self.maxpc)]):g.addWidget(QLabel(n),i//4,(i%4)*2);g.addWidget(x,i//4,(i%4)*2+1)
  mv.addLayout(g);audit=QPushButton('Preview eligibility');audit.clicked.connect(self.preview);go=QPushButton('Run nested optimization');go.clicked.connect(self.optimize);mv.addWidget(audit);mv.addWidget(go);self.elig=QTableWidget();mv.addWidget(self.elig);self.tabs.addTab(md,'Predictive Modeling')
  rv=QWidget();rvl=QVBoxLayout(rv);self.reviewTabs=QTabWidget();self.hist=Plot();self.compare=Plot();self.conf=Plot();self.fold=Plot();self.trialplot=Plot();[self.reviewTabs.addTab(a,b) for a,b in [(self.hist,'Optimization history'),(self.compare,'Model comparison'),(self.conf,'Confusion matrix'),(self.fold,'Fold stability'),(self.trialplot,'Trial projection')]];rvl.addWidget(self.reviewTabs);sr=QHBoxLayout();self.trialSlider=QSlider(Qt.Horizontal);self.trialSlider.setRange(0,0);self.trialSlider.valueChanged.connect(self.replay_trial);self.trialText=QPlainTextEdit();self.trialText.setMaximumHeight(150);sr.addWidget(QLabel('Trial'));sr.addWidget(self.trialSlider);rvl.addLayout(sr);rvl.addWidget(self.trialText);self.tabs.addTab(rv,'Model Review')
 def sync(self):
  names=[self.lst.item(i).data(Qt.UserRole) for i in range(self.lst.count())];old=self.steps[:];self.steps=[old[i] for i in names]
 def refresh(self):
  self.lst.clear()
  for i,s in enumerate(self.steps):it=QListWidgetItem(f'{s.name} | {json.dumps(s.params)}');it.setData(Qt.UserRole,i);self.lst.addItem(it)
 def add_step(self):self.sync();n=self.add.currentText();self.steps.append(Step(n,dict(DEFAULTS[n])));self.refresh()
 def remove_step(self):self.sync();i=self.lst.currentRow();self.steps.pop(i) if i>=0 else None;self.refresh()
 def edit_step(self):
  self.sync();i=self.lst.currentRow()
  if i<0:return
  t,ok=QInputDialog.getMultiLineText(self,'Parameters','JSON',json.dumps(self.steps[i].params,indent=2))
  if ok:
   try:self.steps[i].params=json.loads(t);self.refresh()
   except Exception as e:QMessageBox.warning(self,'Invalid JSON',str(e))
 def save_recipe(self):self.sync();p,_=QFileDialog.getSaveFileName(self,'Save recipe','','JSON (*.json)');Path(p).write_text(json.dumps([asdict(s) for s in self.steps],indent=2)) if p else None
 def load_recipe(self):
  p,_=QFileDialog.getOpenFileName(self,'Load recipe','','JSON (*.json)')
  if p:self.steps=[Step(**x) for x in json.loads(Path(p).read_text())];self.refresh()
 def load(self):
  p=self.path.text().strip()
  if not p:p,_=QFileDialog.getOpenFileName(self,'CSV','','CSV (*.csv)')
  if not p:return
  try:self.data=load_ftir(p);self.path.setText(p);m,w,x=self.data;self.info.setPlainText(f'Samples {len(m)}\nWavenumbers {len(w)}\nRange {w.min():.2f}-{w.max():.2f}\nMetadata {list(m.columns)}\n\nMissing:\n{m.isna().sum()}')
  except Exception:self.err()
 def axes(self):return self.pc1.value()-1,self.pc2.value()-1,self.pc3.value()-1
 def project(self):
  if not self.data:return
  try:
   self.sync();m,w,x=self.data;r=exploratory_pca(x.to_numpy(),w,self.steps,self.npcs.value());self.last=r;z=r['scores'] if self.method.currentText()=='PCA' else embedding(r['processed'],'UMAP',3 if self.dim.currentText()=='3D' else 2,n_neighbors=self.neigh.value(),min_dist=self.mind.value());self.embed=z
   for q in [self.raw,self.proc,self.proj,self.loadings_plot,self.var]:q.fig.clear()
   a=self.raw.fig.add_subplot();a.plot(w,x.iloc[:80].T,alpha=.18,lw=.5);a.invert_xaxis();self.raw.draw();a=self.proc.fig.add_subplot();a.plot(r['wn'],r['processed'][:80].T,alpha=.18,lw=.5);a.invert_xaxis();self.proc.draw();self.draw_projection(z)
   a=self.loadings_plot.fig.add_subplot();a.plot(r['wn'],r['loadings'][:,:min(5,r['loadings'].shape[1])]);a.invert_xaxis();self.loadings_plot.draw();a=self.var.fig.add_subplot();a.bar(range(1,len(r['variance'])+1),100*r['variance']);self.var.draw()
  except Exception:self.err()
 def draw_projection(self,z,labels=None):
  self.proj.fig.clear();x,y,zz=self.axes();is3=self.dim.currentText()=='3D';a=self.proj.fig.add_subplot(projection='3d' if is3 else None);c=labels if labels is not None else None
  if self.method.currentText()=='UMAP':x,y,zz=0,1,2
  if is3:a.scatter(z[:,x],z[:,y],z[:,zz],c=c,cmap='tab20');a.set_zlabel(f'{self.method.currentText()} {zz+1}')
  else:a.scatter(z[:,x],z[:,y],c=c,cmap='tab20')
  a.set_xlabel(f'{self.method.currentText()} {x+1}');a.set_ylabel(f'{self.method.currentText()} {y+1}');self.proj.draw()
 def run_cv(self):
  try:self.sync();m,w,x=self.data;cols=[c.strip() for c in self.groups.text().split(',')];g=make_groups(m,cols) if all(c in m for c in cols) else None;d,b=pca_cv(x.to_numpy(),self.steps,self.maxpc.value(),self.outer.value(),g);self.cv_plot.fig.clear();a=self.cv_plot.fig.add_subplot();a.plot(d.components,d.RMSEC_X,label='RMSEC-X');a.plot(d.components,d.RMSECV_X,label='RMSECV-X');a.axvline(b,color='r',ls='--');a.legend();self.cv_plot.draw();self.et.setCurrentWidget(self.cv_plot)
  except Exception:self.err()
 def do_cluster(self):
  try:lab,s=cluster(self.embed[:,:min(10,self.embed.shape[1])],self.cl.currentText(),self.k.value());self.draw_projection(self.embed,lab);self.frames=kmeans_history(self.embed[:,:3],self.k.value()) if self.cl.currentText()=='KMeans' else [];self.iter.setRange(0,max(0,len(self.frames)-1))
  except Exception:self.err()
 def replay_cluster(self,i):
  if getattr(self,'frames',None):self.draw_projection(self.embed,self.frames[i]['labels'])
 def preview(self):
  try:m,_,_=self.data;cols=[c.strip() for c in self.groups.text().split(',')];d,_,_=eligibility(m.dropna(subset=[self.label.text()]).reset_index(drop=True),self.label.text(),cols,self.minG.value());self.table(self.elig,d)
  except Exception:self.err()
 def optimize(self):
  try:
   self.sync();m,w,x=self.data;v=m[self.label.text()].notna();mm=m.loc[v].reset_index(drop=True);X=x.loc[v].to_numpy();cols=[c.strip() for c in self.groups.text().split(',')];d,y,g=eligibility(mm,self.label.text(),cols,self.minG.value());ok=set(d.loc[d.eligible,'class']);keep=np.array([a in ok for a in y]);X,y,g=X[keep],y[keep],g[keep]
   if len(np.unique(y))<2:raise ValueError('Fewer than two eligible classes')
   per=pd.DataFrame({'y':y,'g':g}).drop_duplicates().groupby('y').size();outer=min(self.outer.value(),int(per.min()));res=nested_optimize(X,y,g,self.steps,self.trials.value(),self.maxpc.value(),outer,self.inner.value());self.opt=(res,X,y,g);self.render_review();self.tabs.setCurrentIndex(self.tabs.count()-1)
  except Exception:self.err()
 def render_review(self):
  r,X,y,g=self.opt;h=r['history'];complete=h[h.status=='complete'].copy();complete['running_best']=complete.objective.cummax();self.hist.fig.clear();a=self.hist.fig.add_subplot();a.plot(complete.trial,complete.score,'.',label='score');a.plot(complete.trial,complete.running_best,label='running best');a.legend();self.hist.draw();self.compare.fig.clear();a=self.compare.fig.add_subplot();complete.boxplot(column='score',by=['model','use_pca'],ax=a);self.compare.fig.suptitle('');self.compare.draw();self.conf.fig.clear();a=self.conf.fig.add_subplot();a.imshow(r['cm'],cmap='Blues');a.set_xticks(range(len(r['classes'])),r['classes'],rotation=45,ha='right');a.set_yticks(range(len(r['classes'])),r['classes']);self.conf.draw();self.fold.fig.clear();a=self.fold.fig.add_subplot();a.plot(r['folds'].fold,r['folds'].balanced_accuracy,'o-',label='balanced accuracy');a.plot(r['folds'].fold,r['folds'].macro_f1,'s-',label='macro F1');a.legend();self.fold.draw();self.trials_df=complete.reset_index(drop=True);self.trialSlider.setRange(0,max(0,len(self.trials_df)-1));self.replay_trial(0);out=Path(self.path.text()).with_name(Path(self.path.text()).stem+'_v5_results');out.mkdir(exist_ok=True);h.to_csv(out/'optimization_history.csv',index=False);r['folds'].to_csv(out/'outer_fold_metrics.csv',index=False);r['report'].to_csv(out/'classification_report.csv');pd.DataFrame(r['cm'],index=r['classes'],columns=r['classes']).to_csv(out/'confusion_matrix.csv')
 def replay_trial(self,i):
  if not hasattr(self,'trials_df') or self.trials_df.empty:return
  row=self.trials_df.iloc[i];self.trialText.setPlainText(row.to_string());self.trialplot.fig.clear();a=self.trialplot.fig.add_subplot(projection='3d');r=self.last if self.last else exploratory_pca(self.opt[1],np.arange(self.opt[1].shape[1]),self.steps,3);z=r['scores'];a.scatter(z[:,0],z[:,1],z[:,2]);a.set_title(f"Trial {int(row.trial)} {row.model} PCA={row.use_pca} score={row.score:.3f}");self.trialplot.draw()
 def table(self,w,d):w.setRowCount(len(d));w.setColumnCount(len(d.columns));w.setHorizontalHeaderLabels(list(d.columns));[[w.setItem(i,j,QTableWidgetItem(str(v))) for j,v in enumerate(row)] for i,row in d.reset_index(drop=True).iterrows()];w.resizeColumnsToContents()
 def err(self):t=traceback.format_exc();b=QMessageBox(self);b.setIcon(QMessageBox.Critical);b.setText(t.splitlines()[-1]);b.setDetailedText(t);b.exec()
if __name__=='__main__':app=QApplication(sys.argv);app.setStyle('Fusion');w=Main();w.show();sys.exit(app.exec())
