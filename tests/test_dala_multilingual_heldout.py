import importlib.util
import json
from pathlib import Path

import pytest

from dfm_evals.tasks import dala_multilingual_heldout as task


@pytest.fixture
def manifest(tmp_path):
    source=tmp_path/'producer.json'
    source.write_text('{}')
    test=tmp_path/'test.jsonl'
    rows=[dict(pair_id=str(i),language='nb',split='test',original=f'Correct {i}',corrupted=f'Incorrect {i}',
               document_id=f'doc{i}',document_sha256=str(i)*64,source_dataset='test/repo',source_revision='a'*40,license='CC0') for i in range(3)]
    test.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    entry=dict(status='ready',split='test',source_language='nb',
        exclusion_proof=dict(remaining_train_matches=0),
        producer_manifest=dict(path=str(source),sha256=task.sha256(source)),
        test_file=dict(path=str(test),sha256=task.sha256(test)),test_pairs=3,
        selected_pair_ids=['2','0'],prompts=dict(acceptability='Bokmål? yes/no',correction='Rett på bokmål.'))
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(schema='dfm-dala-heldout-v1',languages={'nb':entry})))
    return path


def test_balanced_controls_deterministic_ids_and_no_train_fallback(manifest):
    rows=list(task.samples(manifest,'nb','acceptability'))
    assert [r.id for r in rows]==['nb:acceptability:2:clean','nb:acceptability:2:corrupted',
                                 'nb:acceptability:0:clean','nb:acceptability:0:corrupted']
    assert [r.target for r in rows]==['correct','incorrect']*2
    assert rows[0].input=='Bokmål? yes/no\n\nCorrect 2'
    corrections=list(task.samples(manifest,'nb','correction'))
    assert [r.target for r in corrections]==['Correct 2','Correct 2','Correct 0','Correct 0']
    assert len({r.metadata['source_revision'] for r in rows})==1
    with pytest.raises(ValueError,match='fallback'):list(task.samples(manifest,'no','acceptability'))
    with pytest.raises(KeyError):list(task.samples(manifest,'en','acceptability'))


@pytest.mark.parametrize('mutation',['train','hash','missing_id','duplicates','overlap'])
def test_manifest_and_split_fail_closed(manifest,mutation):
    m=json.loads(manifest.read_text());s=m['languages']['nb']
    if mutation=='train':s['split']='train'
    if mutation=='hash':s['test_file']['sha256']='bad'
    if mutation=='missing_id':s['selected_pair_ids']=['absent']
    if mutation=='duplicates':s['selected_pair_ids']=['0','0']
    if mutation=='overlap':s['exclusion_proof']['remaining_train_matches']=1
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError):task.selected_pairs(manifest,'nb')


def test_pair_grouped_shards_and_generation_budgets(manifest):
    shards=[task.dala_heldout('nb',str(manifest),num_shards=2,shard_index=i) for i in range(2)]
    assert all(len(t.dataset)==2 for t in shards)
    assert [r.metadata['pair_id'] for r in shards[0].dataset]==['2','2']
    assert set(r.id for t in shards for r in t.dataset)==set(r.id for r in task.samples(manifest,'nb','acceptability'))
    assert len(task.gec_dala_heldout('nb',str(manifest)).dataset)==4


@pytest.mark.parametrize('text,expected',[('yes','correct'),(' NO ','incorrect'),('maybe',None),('yes no',None),('The answer is yes',None)])
def test_exact_labels(text,expected):
    assert task.extract_label(text)==expected


def test_metrics_match_retained_dala_and_gec():
    metrics=task.multilingual_dala_scorer().__registry_info__.metadata['metrics']
    assert len(metrics)==2
    assert task.gec_dala_scorer().__registry_info__.metadata['metrics'].keys()=={'exact_match'}


def test_file_selector_loads_without_shared_registry_edits(manifest):
    from inspect_ai._eval.loader import load_task_spec
    selector=str(Path(task.__file__).resolve())+'@dala_heldout'
    loaded=load_task_spec(selector,dict(language='nb',manifest=str(manifest)))
    assert len(loaded)==1 and len(loaded[0].dataset)==4


def test_scope_config_and_native_prompt_registry():
    import yaml
    root=Path(task.__file__).resolve().parents[3]
    registry=yaml.safe_load((root/'dfm12/european_synthetic_extension.yaml').read_text())['languages']
    assert set(task.LANGUAGES)==set(registry)|{'nb','nn','sv','is','fo','nl','pl'}
    config=yaml.safe_load((root/'config/dfm_evals_multilingual_dala_heldout.yaml').read_text())['sets']
    assert set(config)=={f'{prefix}_{lang}' for prefix in ('dala','gec_dala') for lang in task.ALL_LANGUAGES}
    prompts=json.loads((root/'config/dfm_dala_native_prompts_20260930.json').read_text())
    assert set(prompts)==set(registry)
    assert 'português europeu' in prompts['pt_pt']['correction']


def test_producer_exclusion_checks_normalize_and_protect_documents():
    path=Path(task.__file__).resolve().parents[2]/'scripts/prepare_dala_heldout_manifest.py'
    spec=importlib.util.spec_from_file_location('prepare_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a=dict(pair_id='a',document_sha256='doc',original='Één  Zin',corrupted='Other')
    b=dict(pair_id='b',document_sha256='doc',original='e\u0301e\u0301n zin',corrupted='Different')
    common=set(module.keys(a))&set(module.keys(b))
    assert ('document','doc') in common
    assert ('text',module.normalized('Één  Zin')) in common
