"""Pinned, pair-grouped multilingual DaLA test tasks; never load train as eval."""
import gzip
import hashlib
import json
from pathlib import Path
import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import generate
from dfm_evals.tasks.dala import dala_macro_f1, dala_mcc
from dfm_evals.tasks.gec_dala import gec_dala_scorer
from dfm_evals.tasks._sharding import shard_samples

LANGUAGES = ('nb','nn','sv','is','fo','nl','pl','de','fr','es','it','cs','pt_pt','fi','et','ca','el','ro','uk')
ALL_LANGUAGES = (*LANGUAGES, 'en')
DEFAULT_MANIFEST = str(Path(__file__).resolve().parents[3] / 'config/dfm_dala_heldout_20260930.json')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def pairs(path):
    path=Path(path)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def selected_pairs(manifest, language):
    if language not in ALL_LANGUAGES:
        raise ValueError('Unsupported language; no language fallback')
    document=json.loads(Path(manifest).read_text())
    if document.get('schema')!='dfm-dala-heldout-v1':raise ValueError('Unknown heldout manifest')
    source=document['languages'][language]
    if source['split']!='test' or source['status']!='ready':raise ValueError('Verified test source required')
    if source['exclusion_proof']['remaining_train_matches']!=0:raise ValueError('Train overlap unresolved')
    for item in (source['producer_manifest'],source['test_file']):
        if sha256(item['path'])!=item['sha256']:raise ValueError('Heldout source hash drift')
    ids=source['selected_pair_ids']
    if not ids or len(set(ids))!=len(ids):raise ValueError('Nonempty unique heldout selection required')
    wanted=set(ids); found={}; count=0
    for row in pairs(source['test_file']['path']):
        count+=1
        if row.get('split')!='test' or row.get('language')!=source['source_language']:
            raise ValueError('Heldout pair language/split drift')
        if row['pair_id'] in wanted:
            if row['pair_id'] in found:raise ValueError('Duplicate pair ID')
            if not all(isinstance(row.get(k),str) and row[k].strip() for k in ('original','corrupted')) or row['original']==row['corrupted']:
                raise ValueError('Invalid heldout corruption pair')
            found[row['pair_id']]=row
    if count!=source['test_pairs'] or set(found)!=wanted:raise ValueError('Heldout count/selection drift')
    return source,[found[key] for key in ids]


def samples(manifest,language,kind):
    source,rows=selected_pairs(manifest,language)
    prompt=source['prompts'][kind]
    for pair in rows:
        for variant,text in [('clean',pair['original']),('corrupted',pair['corrupted'])]:
            target=('correct' if variant=='clean' else 'incorrect') if kind=='acceptability' else pair['original']
            yield Sample(id=f"{language}:{kind}:{pair['pair_id']}:{variant}",input=prompt+'\n\n'+text,target=target,
                metadata=dict(language=language,split='test',pair_id=pair['pair_id'],variant=variant,
                    source_revision=pair['source_revision'],source_dataset=pair['source_dataset'],
                    document_id=pair['document_id'],document_sha256=pair['document_sha256'],
                    original=pair['original'],corrupted=pair['corrupted'],license=pair.get('license'),
                    quality_status=pair.get('quality_status'),source_manifest_sha256=source['producer_manifest']['sha256']))


def extract_label(text):
    # All native prompts explicitly ask for the same literal yes/no answer labels.
    label=text.strip().casefold()
    return {'yes':'correct','no':'incorrect'}.get(label)


@scorer(metrics=[dala_macro_f1(),dala_mcc()],name='linguistic-acceptability')
def multilingual_dala_scorer():
    async def score(state,target:Target):
        prediction=extract_label(state.output.completion)
        return Score(value=prediction or '',answer=prediction or '',
            explanation=f'predicted={prediction!r}, expected={target.text!r}',
            metadata=dict(prediction=prediction,target=target.text))
    return score


def make_task(language,kind,manifest,num_shards,shard_index,max_gen_toks):
    if max_gen_toks<1:raise ValueError('Positive output budget required')
    name=('dala_' if kind=='acceptability' else 'gec_dala_')+language
    # Shard pairs rather than individual clean/corrupt controls.
    if num_shards<1 or not 0<=shard_index<num_shards:raise ValueError('Invalid shard')
    rows=[s for i,s in enumerate(samples(manifest,language,kind)) if (i//2)%num_shards==shard_index]
    dataset=shard_samples(rows,name=name,location=f'{manifest}:{language}:test')
    return Task(dataset=dataset,solver=[generate(max_tokens=max_gen_toks,temperature=0)],
                scorer=multilingual_dala_scorer() if kind=='acceptability' else gec_dala_scorer())


@task(name='dala_heldout')
def dala_heldout(language:str,manifest:str=DEFAULT_MANIFEST,num_shards:int=1,shard_index:int=0,max_gen_toks:int=32):
    return make_task(language,'acceptability',manifest,num_shards,shard_index,max_gen_toks)


@task(name='gec_dala_heldout')
def gec_dala_heldout(language:str,manifest:str=DEFAULT_MANIFEST,num_shards:int=1,shard_index:int=0,max_gen_toks:int=512):
    return make_task(language,'correction',manifest,num_shards,shard_index,max_gen_toks)
