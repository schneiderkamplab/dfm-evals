"""CPU-only discovery and exact producer-train/test exclusion for twenty languages."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'dfm-evals'))
from dfm_evals.tasks.dala_multilingual_heldout import ALL_LANGUAGES, LANGUAGES, pairs, sha256

PRODUCER=Path('/work/mimir/DaLA')


def normalized(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFC',text).casefold().split()).encode()).hexdigest()


def keys(row):
    return [('pair',row['pair_id']),('document',row['document_sha256']),
            ('text',normalized(row['original'])),('text',normalized(row['corrupted']))]


def descriptor(path):
    return dict(path=str(path.resolve()),sha256=sha256(path),bytes=path.stat().st_size)


def discover(language):
    if language in ('nb','nn','sv','is','fo','pl'):
        root=PRODUCER/'la_output/six_language_candidates_recovery_v1'/language
        manifest=root/'manifest.json'
        train,test=root/'train/pairs.jsonl',root/'test/pairs.jsonl'
        revision=None; repo=None
    else:
        package='dala-english-common-pile' if language=='en' else 'dala-dutch-dynaword' if language=='nl' else 'european-audited-20260928/dala-'+language.replace('_','-')+'-audited'
        root=PRODUCER/'export-upload'/package
        manifest=root/'metadata/manifest.json'
        train,test=root/'provenance/train.pairs.jsonl.gz',root/'provenance/test.pairs.jsonl.gz'
        if language in ('en','nl'):
            receipt=json.loads((PRODUCER/'wiki/artifacts'/('english-hf-upload.json' if language=='en' else 'dutch-hf-upload.json')).read_text())
        else:
            repo=json.loads(manifest.read_text())['repo_id']
            receipt=json.loads((root.parent/'upload-receipts.json').read_text())[repo]
        if receipt['manifest_sha256']!=sha256(manifest):raise ValueError('Published manifest drift: '+language)
        revision=receipt['revision'];repo=json.loads(manifest.read_text())['repo_id']
    for path in (manifest,train,test):
        if not path.is_file():raise FileNotFoundError(path)
    m=json.loads(manifest.read_text())
    generation=root/'metadata/generation-manifest.json' if (root/'metadata').is_dir() else manifest
    g=json.loads(generation.read_text())
    entries=m.get('files',m.get('artifacts',{}))
    for path in (train,test):
        item=entries[str(path.relative_to(root))]
        if sha256(path)!=item['sha256']:raise ValueError('Producer artifact drift: '+str(path))
    return root,manifest,train,test,g,repo,revision


def prepare_language(args):
    language,cap,prompts=args
    root,manifest,train,test,g,repo,revision=discover(language)
    test_rows={}; index=defaultdict(set); revisions=set()
    source_language=g['language']
    for row in pairs(test):
        if row.get('split')!='test' or row['language']!=source_language:raise ValueError('Wrong test split/language')
        key=row['pair_id']
        if key in test_rows:raise ValueError('Duplicate test pair ID')
        test_rows[key]=True
        for value in keys(row):index[value].add(key)
        revisions.add((row['source_dataset'],row['source_revision']))
    excluded=defaultdict(set);train_count=0
    for row in pairs(train):
        if row.get('split')!='train' or row['language']!=source_language:raise ValueError('Wrong train split/language')
        train_count+=1
        for value in keys(row):
            for key in index.get(value,()):excluded[key].add(value[0])
    eligible=sorted(set(test_rows)-set(excluded),key=lambda key:hashlib.sha256(f'4242:{language}:{key}'.encode()).hexdigest())
    selected=eligible[:cap//2] if cap else eligible
    if not selected or (cap and len(selected)*2!=cap):raise ValueError('Insufficient disjoint heldout: '+language)
    selected_prompts=prompts.get(language,g['prompts'])
    result=dict(status='ready',language=language,source_language=source_language,split='test',
        source_root=str(root),repo_id=repo,revision=revision,upstream_revisions=[dict(repo_id=r,revision=v) for r,v in sorted(revisions)],
        producer_manifest=descriptor(manifest),test_file=descriptor(test),train_file=descriptor(train),
        producer_quality=g.get('quality_status'),native_human_validation=False,
        train_pairs=train_count,test_pairs=len(test_rows),eligible_test_pairs=len(eligible),
        selected_pair_ids=selected,rows_per_task=2*len(selected),seed=4242,
        prompts=selected_prompts,producer_prompts=g['prompts'],
        prompt_origin='native_evaluation_translation_not_human_validated' if language in prompts else 'unchanged_producer_native_prompt',
        exclusion_proof=dict(scope='same producer full train versus full test; pair ID, document SHA256, NFC/casefold/whitespace text',
            remaining_train_matches=0,excluded_test_pairs=len(excluded),
            exclusions={key:sorted(value) for key,value in sorted(excluded.items())},
            inherited_corpus_coverage='not proven; no claim of full DFM11/DFM12 corpus decontamination'))
    print(language,train_count,len(test_rows),'excluded',len(excluded),'selected',len(selected),flush=True)
    return language,result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'config/dfm_dala_heldout_20260930.json')
    parser.add_argument('--prompts',type=Path,default=ROOT/'config/dfm_dala_native_prompts_20260930.json')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--samples',type=int,default=2000,help='Even cap per task; zero means all disjoint test pairs')
    args=parser.parse_args()
    if not 1<=args.workers<=4 or args.samples<0 or args.samples%2:raise ValueError('Require <=4 workers and even cap')
    if args.output.exists():raise FileExistsError(args.output)
    prompts=json.loads(args.prompts.read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        languages=dict(pool.map(prepare_language,[(l,args.samples,prompts) for l in ALL_LANGUAGES]))
    manifest=dict(schema='dfm-dala-heldout-v1',multilingual_languages=LANGUAGES,english_languages=['en'],
        samples_per_task=args.samples,selection='seed4242 hash-ranked eligible pair IDs; both controls retained',
        source_registry=descriptor(ROOT/'dfm12/european_synthetic_extension.yaml'),
        native_prompt_pin=descriptor(args.prompts),languages=languages)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(manifest,f,indent=2,ensure_ascii=False)


if __name__=='__main__':main()
