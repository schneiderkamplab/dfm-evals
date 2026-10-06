import hashlib
import json
from pathlib import Path
import pytest
import asyncio
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dfm_evals.tasks import dala_dfm13_heldout as module
from scripts.check_dfm13_multilingual_evals import check_merge
from scripts.dala_semantic import extract_semantic_label


def fixture(root):
    selected=root/'pairs.json'
    selected.write_text(json.dumps([dict(id='pair1',original='clean',corrupted='noisy',provenance={'split':'test'},compact_audit={})]))
    manifest=root/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='dfm13-accepted-dala-heldout-v1',languages={'lt':dict(
        status='ready',split='test',view='representative',pairs=1,samples_per_task=2,
        selected=dict(path=str(selected),sha256=hashlib.sha256(selected.read_bytes()).hexdigest()),input_pins=[],
        prompts={'acceptability':'Reply yes/no','correction':'Correct'},quality='fixture')})))
    return manifest


def test_no_training_fallback(tmp_path):
    p=fixture(tmp_path);d=json.loads(p.read_text());d['languages']['lt']['split']='train';p.write_text(json.dumps(d))
    with pytest.raises(ValueError):module.selected(p,'lt')


def test_no_mutated_source_or_unknown_language(tmp_path):
    p=fixture(tmp_path)
    with pytest.raises(ValueError):module.selected(p,'en')
    (tmp_path/'pairs.json').write_text('[]')
    with pytest.raises(ValueError):module.selected(p,'lt')


def test_pair_controls_share_shard(tmp_path):
    p=fixture(tmp_path)
    t=module.make('lt','acceptability',str(p),2,0,32)
    assert len(t.dataset)==2
    assert [s.target for s in t.dataset]==['correct','incorrect']
    with pytest.raises(ValueError,match='empty'):
        module.make('lt','acceptability',str(p),2,1,32)


def test_default_manifest_is_repository_config():
    assert Path(module.DEFAULT_MANIFEST)==Path(__file__).resolve().parents[2]/'config/dfm13_dala_heldout_20261006.json'
    assert Path(module.DEFAULT_MANIFEST).is_file()


@pytest.mark.parametrize('language',module.LANGUAGES)
def test_construct_score_merge_and_average_binding(tmp_path,language):
    p=fixture(tmp_path);data=json.loads(p.read_text())
    data['languages'][language]=data['languages'].pop('lt')
    p.write_text(json.dumps(data))
    t=module.make(language,'acceptability',str(p),1,0,32)
    result=asyncio.run(check_merge(language,list(t.dataset)))
    assert all(m['semantic_v1/macro_f1']==1 for m in result.values())
    assert extract_semantic_label('yes, but this sentence is wrong',language) is None
    assert extract_semantic_label('yes/no',language) is None
    root=Path(__file__).resolve().parents[2]
    populations=json.loads((root/'config/multilingual_headline_populations_dfm13_20261006.json').read_text())
    for pop in populations['populations']:
        assert pop['metrics'][language]['dfm_la']['key']==f'dfm_eval/dala_{language}/semantic_v1/macro_f1'


def test_merged_scores_feed_real_population(tmp_path):
    from scripts.headline_population_registry import build_population_row
    from scripts.log_multilingual_headline_averages import PopulationItem
    root=Path(__file__).resolve().parents[2]
    registry=json.loads((root/'config/multilingual_headline_populations_dfm13_20261006.json').read_text())
    pop=registry['populations'][0]
    values={b['key']:(50 if b['scale']=='percent' else .5)
            for bindings in pop['metrics'].values() for b in bindings.values()}
    p=fixture(tmp_path);data=json.loads(p.read_text());entry=data['languages']['lt']
    data['languages']={l:entry for l in module.LANGUAGES};p.write_text(json.dumps(data))
    for language in module.LANGUAGES:
        task=module.make(language,'acceptability',str(p),1,0,32)
        metrics=asyncio.run(check_merge(language,list(task.dataset)))['native']
        values.update({f'dfm_eval/dala_{language}/{k}':v for k,v in metrics.items()})
    (tmp_path/'merged_metrics.json').write_text(json.dumps(values))
    row,_=build_population_row(PopulationItem(1,1,[],[tmp_path],[tmp_path]),
                               dict(schema_version=1,populations=[pop]))
    expected=sum(.5+.5/len(m) for m in pop['metrics'].values())/13
    assert row['avg_population/dfm13_new_languages_v1/score']==pytest.approx(expected)
