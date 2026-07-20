"""Build the final MediaPipe-filtered leakage-safe protocol without moving data."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
def dump(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf-8')
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/expression_classification/outputs/classifier_comparison_mediapipe_clean'); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
 man=ROOT/'experiments/expression_classification/manifests/official_protocol_manifest.csv'; safe=ROOT/'experiments/expression_classification/manifests/leakage_safe_manifest.csv'; det=ROOT/'experiments/expression_classification/outputs/mediapipe_face_validity_official_v1/raw_detections.csv'
 official=pd.read_csv(man).reset_index(names='official_row_index'); ls=pd.read_csv(safe); mp=pd.read_csv(det)
 if official.sample_id.duplicated().any() or ls.sample_id.duplicated().any() or mp.sample_id.duplicated().any(): raise ValueError('duplicate sample_id in input')
 aligned=ls.merge(official,on='sample_id',how='left',validate='one_to_one',suffixes=('_leakage_safe','_official')).merge(mp[['official_row_index','sample_id','face_detected']],on=['official_row_index','sample_id'],how='left',validate='one_to_one')
 if aligned.official_row_index.isna().any() or aligned.face_detected.isna().any(): raise ValueError('one-to-one official/MediaPipe alignment failure')
 final=aligned[aligned.face_detected].sort_values('official_row_index').copy(); excluded=aligned[~aligned.face_detected].sort_values('official_row_index').copy()
 if len(final)!=32960: raise ValueError(f'expected 32960 retained rows, found {len(final)}')
 outman=ROOT/'experiments/expression_classification/manifests/mediapipe_filtered_leakage_safe_manifest.csv'; outex=ROOT/'experiments/expression_classification/manifests/mediapipe_filtered_leakage_safe_exclusions.csv'; outcounts=ROOT/'experiments/expression_classification/manifests/mediapipe_filtered_leakage_safe_counts.csv'
 if not aligned.class_name_leakage_safe.equals(aligned.class_name_official) or not aligned.split_leakage_safe.equals(aligned.split_official): raise ValueError('sample ID label/split mismatch between leakage-safe and official inputs')
 path_hash_mismatches=int(((aligned.relative_path_leakage_safe!=aligned.relative_path_official)|(aligned.sha256_leakage_safe!=aligned.sha256_official)).sum())
 final_out=final.rename(columns={'class_name_leakage_safe':'class_name','split_leakage_safe':'split','relative_path_leakage_safe':'relative_path','sha256_leakage_safe':'sha256','relative_path_official':'official_relative_path','sha256_official':'official_sha256'})
 excluded_out=excluded.rename(columns={'class_name_leakage_safe':'class_name','split_leakage_safe':'split','relative_path_leakage_safe':'relative_path','sha256_leakage_safe':'sha256','relative_path_official':'official_relative_path','sha256_official':'official_sha256'})
 final_out[['official_row_index','sample_id','class_name','split','relative_path','sha256','official_relative_path','official_sha256']].to_csv(outman,index=False); excluded_out[['official_row_index','sample_id','class_name','split','relative_path','sha256','official_relative_path','official_sha256','face_detected']].assign(exclusion_reason='mediapipe_no_detection').to_csv(outex,index=False)
 counts=aligned.groupby(['split_leakage_safe','class_name_leakage_safe'],as_index=False).agg(leakage_safe_rows=('sample_id','size'),retained_rows=('face_detected','sum')).rename(columns={'split_leakage_safe':'split','class_name_leakage_safe':'class_name'}); counts['excluded_rows']=counts.leakage_safe_rows-counts.retained_rows; counts['retention_rate']=counts.retained_rows/counts.leakage_safe_rows; counts.to_csv(outcounts,index=False)
 # The official row positions index existing arrays; no features are duplicated.
 idx=final.official_row_index.to_numpy(dtype=np.int64); np.save(a.output_dir/'lbf_mediapipe_clean_subset_indices.npy',idx); np.save(a.output_dir/'rtmpose_mediapipe_clean_subset_indices.npy',idx)
 lbf=pd.read_csv(ROOT/'experiments/expression_classification/outputs/lbf_official_full/sample_index.csv'); rtp=pd.read_csv(ROOT/'experiments/expression_classification/outputs/rtmpose_official_full/sample_index.csv')
 target=final[['sample_id','class_name_official','split_official','relative_path_official','sha256_official']].rename(columns={'class_name_official':'class_name','split_official':'split','relative_path_official':'relative_path','sha256_official':'sha256'}).reset_index(drop=True)
 lbf_target=lbf.iloc[idx][target.columns].reset_index(drop=True); rtp_target=rtp.iloc[idx][target.columns].reset_index(drop=True)
 if not target.equals(lbf_target) or not target.equals(rtp_target) or not np.array_equal(idx,np.sort(idx)): raise ValueError('array/sample-index alignment validation failed')
 summary={'protocol':'MediaPipe-filtered leakage-safe dataset','source_leakage_safe_rows':len(aligned),'retained_rows':len(final),'excluded_rows':len(excluded),'expected_retained_rows':32960,'retained_all_mediapipe_detected':bool(final.face_detected.all()),'excluded_all_mediapipe_not_detected':bool((~excluded.face_detected).all()),'split_counts':final.split_leakage_safe.value_counts().sort_index().to_dict(),'class_counts':final.class_name_leakage_safe.value_counts().sort_index().to_dict(),'disgust_retention':counts[counts.class_name.eq('disgust')].to_dict('records'),'leakage_safe_policy_preserved':True,'leakage_safe_path_or_hash_differs_from_official_rows':path_hash_mismatches,'note':'No-detection is a reproducible face-detection criterion, not proof that an excluded source is semantically invalid. Leakage-safe path/hash fields are preserved; official path/hash fields are recorded for existing-array provenance.'}
 dump(ROOT/'experiments/expression_classification/manifests/mediapipe_filtered_leakage_safe_summary.json',summary)
 (ROOT/'experiments/expression_classification/manifests/mediapipe_filtered_leakage_safe_protocol.md').write_text('# Final cleaned protocol\n\nThe final cleaned protocol is the **MediaPipe-filtered leakage-safe dataset**: the existing leakage-safe manifest intersected with rows having a reliable MediaPipe Face Detection result. It retains a reproducible MediaPipe-detected face subset after duplicate/leakage cleaning. A no-detection is not claimed to prove semantic invalidity; difficult faces may be excluded and semantic non-face content may still be detected.\n',encoding='utf-8')
 dump(a.output_dir/'subset_alignment_report.json',{'status':'PASS','final_rows':len(final),'lbf_and_rtmpose_indices_identical':True,'official_row_order_preserved':True,'sample_id_label_split_path_match':True,'indices_sha256':hashlib.sha256(idx.tobytes()).hexdigest(),'no_feature_arrays_copied':True})
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
