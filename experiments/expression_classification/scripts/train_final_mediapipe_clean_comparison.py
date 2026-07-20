"""One frozen final LBF-versus-RTMPose evaluation on the cleaned protocol."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import cv2, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[3]; LABELS=['angry','disgust','fear','happy','neutral','sad','surprise']
def save(path,obj): path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def cm_image(cm,path,title):
 s=75; im=np.full((80+s*7,110+s*7,3),255,np.uint8); mx=max(1,cm.max()); cv2.putText(im,title,(5,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
 for i,label in enumerate(LABELS):
  cv2.putText(im,label[:7],(110+i*s,55),cv2.FONT_HERSHEY_SIMPLEX,.34,(0,0,0),1); cv2.putText(im,label[:7],(4,92+i*s),cv2.FONT_HERSHEY_SIMPLEX,.34,(0,0,0),1)
  for j in range(7):
   v=int(cm[i,j]); shade=int(255-210*v/mx); cv2.rectangle(im,(110+j*s,62+i*s),(110+(j+1)*s,62+(i+1)*s),(shade,shade,255),-1); cv2.putText(im,str(v),(114+j*s,101+i*s),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),1)
 cv2.imwrite(str(path),im)
def evaluate(y,pred):
 prec,rec,f1,sup=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
 return {'accuracy':accuracy_score(y,pred),'macro_precision':float(np.mean(prec)),'macro_recall':float(np.mean(rec)),'macro_f1':f1_score(y,pred,average='macro'),'weighted_f1':f1_score(y,pred,average='weighted'),'per_class':[{"class_name":l,"precision":float(p),"recall":float(r),"f1":float(f),"support":int(n)} for l,p,r,f,n in zip(LABELS,prec,rec,f1,sup)],'confusion_matrix':confusion_matrix(y,pred,labels=LABELS).tolist()}
def fit_one(name,X,rows,out):
 d=out/name; model_dir=d/'model'; model_dir.mkdir(parents=True,exist_ok=True); splits=rows.split.to_numpy(); y=rows.class_name.to_numpy(); tr=splits=='train'; va=splits=='val'; te=splits=='test'
 scaler=StandardScaler(); started=time.perf_counter(); xtr=scaler.fit_transform(X[tr]); clf=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=2000,random_state=42); clf.fit(xtr,y[tr]); train_seconds=time.perf_counter()-started
 started=time.perf_counter(); vp=clf.predict(scaler.transform(X[va])); validation_seconds=time.perf_counter()-started
 started=time.perf_counter(); tp=clf.predict(scaler.transform(X[te])); test_seconds=time.perf_counter()-started
 validation=evaluate(y[va],vp); test=evaluate(y[te],tp); test['inference_seconds']=test_seconds; validation['inference_seconds']=validation_seconds
 meta={'method':name,'classifier':'LogisticRegression','configuration':{'C':1,'class_weight':'balanced','solver':'lbfgs','max_iter':2000,'random_state':42},'feature_dimensions':int(X.shape[1]),'cleaned_split_counts':rows.split.value_counts().sort_index().to_dict(),'training_seconds':train_seconds,'validation':validation,'test':test}
 save(out/f'final_clean_{name}_metrics.json',meta); save(d/'metrics.json',meta); pd.DataFrame(validation['per_class']).to_csv(out/f'final_clean_{name}_validation_per_class.csv',index=False); pd.DataFrame(test['per_class']).to_csv(out/f'final_clean_{name}_test_per_class.csv',index=False)
 cm=np.array(test['confusion_matrix']); cm_image(cm,out/f'final_clean_{name}_confusion_matrix.png',f'{name.upper()} cleaned test'); np.save(out/f'final_clean_{name}_confusion_matrix.npy',cm)
 test_rows=rows.loc[te,['official_row_index','sample_id','split','class_name','relative_path']].copy(); test_rows['true_label']=y[te]; test_rows['predicted_label']=tp; test_rows['correct']=test_rows.true_label.eq(test_rows.predicted_label); test_rows.to_csv(out/f'final_clean_{name}_test_predictions.csv',index=False)
 joblib.dump(scaler,model_dir/'scaler.joblib'); joblib.dump(clf,model_dir/'classifier.joblib'); return meta
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/expression_classification/outputs/classifier_comparison_mediapipe_clean');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 rows=pd.read_csv(ROOT/'experiments/expression_classification/manifests/mediapipe_filtered_leakage_safe_manifest.csv').sort_values('official_row_index').reset_index(drop=True); idx=np.load(a.output_dir/'lbf_mediapipe_clean_subset_indices.npy'); ridx=np.load(a.output_dir/'rtmpose_mediapipe_clean_subset_indices.npy')
 if len(rows)!=32960 or not np.array_equal(idx,ridx) or not np.array_equal(idx,rows.official_row_index.to_numpy()):raise ValueError('cleaned subset indices do not match final manifest')
 lbf=np.load(ROOT/'experiments/expression_classification/outputs/lbf_official_full/features_raw.npz')['features']; kp=np.load(ROOT/'experiments/expression_classification/outputs/rtmpose_official_full/keypoints_raw.npz')['keypoints']; cf=np.load(ROOT/'experiments/expression_classification/outputs/rtmpose_official_full/keypoint_confidence.npz')['confidence']
 center=kp-kp.mean(1,keepdims=True); scale=np.sqrt((center*center).sum(2)).mean(1)
 rtp=np.concatenate([(center/scale[:,None,None]).reshape(len(kp),212).astype(np.float32),cf.astype(np.float32)],axis=1)
 lbf_meta=fit_one('lbf',lbf[idx],rows,a.output_dir); rtp_meta=fit_one('rtmpose',rtp[idx],rows,a.output_dir)
 comparison=[]
 for meta,landmarks in [(lbf_meta,68),(rtp_meta,106)]: comparison.append({'method':meta['method'].upper() if meta['method']=='lbf' else 'RTMPose','landmarks':landmarks,'feature_dimensions':meta['feature_dimensions'],'validation_accuracy':meta['validation']['accuracy'],'validation_macro_f1':meta['validation']['macro_f1'],'test_accuracy':meta['test']['accuracy'],'test_macro_precision':meta['test']['macro_precision'],'test_macro_recall':meta['test']['macro_recall'],'test_macro_f1':meta['test']['macro_f1'],'test_weighted_f1':meta['test']['weighted_f1'],'training_seconds':meta['training_seconds'],'test_inference_seconds':meta['test']['inference_seconds']})
 pd.DataFrame(comparison).to_csv(a.output_dir/'final_clean_comparison.csv',index=False); delta={k:rtp_meta['test'][k]-lbf_meta['test'][k] for k in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1']}; save(a.output_dir/'runtime_summary.json',{'lbf_training_seconds':lbf_meta['training_seconds'],'rtmpose_training_seconds':rtp_meta['training_seconds'],'lbf_test_inference_seconds':lbf_meta['test']['inference_seconds'],'rtmpose_test_inference_seconds':rtp_meta['test']['inference_seconds'],'historical_extraction_mean_seconds_per_image':{'LBF':.023144332384880995,'RTMPose':.3955089669629776}})
 report=f'''# Final cleaned LBF vs RTMPose comparison\n\nOnly the final **MediaPipe-filtered leakage-safe protocol** was used for this training/evaluation. Duplicate/leakage cleaning was applied first; MediaPipe Face Detection then selected the reproducible face-detected subset. Earlier unfiltered results are development-stage evidence only.\n\n- Cleaned rows: {len(rows)} (train {int((rows.split=='train').sum())}, validation {int((rows.split=='val').sum())}, test {int((rows.split=='test').sum())})\n- Frozen classifier: StandardScaler fit on cleaned train only; LogisticRegression C=1, balanced, lbfgs, max_iter=2000, random_state=42.\n- LBF test: accuracy {lbf_meta['test']['accuracy']:.4f}; macro F1 {lbf_meta['test']['macro_f1']:.4f}.\n- RTMPose test: accuracy {rtp_meta['test']['accuracy']:.4f}; macro F1 {rtp_meta['test']['macro_f1']:.4f}.\n- RTMPose minus LBF: accuracy {delta['accuracy']:+.4f}; macro F1 {delta['macro_f1']:+.4f}.\n\nMediaPipe no-detection does not prove semantic invalidity: difficult faces can be filtered, while some semantic non-face content may pass face detection. Disgust has a low cleaned support ({int((rows.class_name=='disgust').sum())} total), so its class metrics require cautious interpretation.\n'''; (a.output_dir/'final_clean_comparison_report.md').write_text(report,encoding='utf-8'); print(json.dumps({'lbf_test':lbf_meta['test'],'rtmpose_test':rtp_meta['test'],'delta':delta},indent=2))
if __name__=='__main__':main()
