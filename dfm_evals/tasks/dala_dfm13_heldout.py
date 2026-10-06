"""Versioned accepted DaLA-v2 heldouts; historical DFM12 tasks stay unchanged."""
import hashlib
import json
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from dfm_evals.tasks.dala_multilingual_heldout import multilingual_dala_scorer
from dfm_evals.tasks.gec_dala import gec_dala_scorer
from dfm_evals.tasks._sharding import shard_samples

LANGUAGES=tuple('lt lv sq be bs bg hr hu lb sr sk sl fa'.split())
# parents[2] is dfm-evals; parents[3] is the repository root.
DEFAULT_MANIFEST=str(Path(__file__).resolve().parents[3]/'config/dfm13_dala_heldout_20261006.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def selected(manifest,language,allowed_languages=LANGUAGES):
    if language not in allowed_languages:raise ValueError('Unknown DFM13 language')
    data=json.loads(Path(manifest).read_text())
    if data.get('schema')!='dfm13-accepted-dala-heldout-v1':raise ValueError('Wrong heldout schema')
    e=data['languages'][language]
    if e['status']!='ready' or e['split']!='test' or e['view']!='representative':raise ValueError('Not verified test data')
    for pin in [e['selected'],*e['input_pins']]:
        if sha(pin['path'])!=pin['sha256']:raise ValueError('Heldout evidence changed')
    rows=json.loads(Path(e['selected']['path']).read_text())
    if len(rows)!=e['pairs'] or len({r['id'] for r in rows})!=len(rows) or len(rows)*2!=e['samples_per_task']:
        raise ValueError('Selection count/IDs changed')
    for r in rows:
        if r['original']==r['corrupted'] or r['provenance']['split']!='test':raise ValueError('Invalid pair')
    return e,rows


def make(language,kind,manifest,num_shards,shard_index,max_gen_toks,allowed_languages=LANGUAGES,namespace=None):
    if num_shards<1 or not 0<=shard_index<num_shards or max_gen_toks<1:raise ValueError('Invalid shard/budget')
    entry,rows=selected(manifest,language,allowed_languages);samples=[]
    for i,row in enumerate(rows):
        if i%num_shards!=shard_index:continue
        for variant,text in [('clean',row['original']),('corrupted',row['corrupted'])]:
            target=('correct' if variant=='clean' else 'incorrect') if kind=='acceptability' else row['original']
            samples.append(Sample(id=f"{language}:{kind}:{row['id']}:{variant}",
                input=entry['prompts'][kind]+'\n\n'+text,target=target,
                metadata=dict(language=language,split='test',view='representative',pair_id=row['id'],variant=variant,
                              provenance=row['provenance'],quality_basis=entry['quality'])))
    name=namespace or ('dala_' if kind=='acceptability' else 'gec_dala_')+language
    dataset=shard_samples(samples,name=name,location=manifest+':'+language+':test')
    return Task(dataset=dataset,solver=[generate(max_tokens=max_gen_toks,temperature=0)],
                scorer=multilingual_dala_scorer() if kind=='acceptability' else gec_dala_scorer())


@task(name='dala_dfm13_heldout')
def dala_dfm13_heldout(language:str,manifest:str=DEFAULT_MANIFEST,num_shards:int=1,shard_index:int=0,max_gen_toks:int=32):
    return make(language,'acceptability',manifest,num_shards,shard_index,max_gen_toks)


@task(name='gec_dala_dfm13_heldout')
def gec_dala_dfm13_heldout(language:str,manifest:str=DEFAULT_MANIFEST,num_shards:int=1,shard_index:int=0,max_gen_toks:int=512):
    return make(language,'correction',manifest,num_shards,shard_index,max_gen_toks)
