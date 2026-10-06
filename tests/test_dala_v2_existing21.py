import asyncio
import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dfm_evals.tasks import dala_v2_heldout as module
from scripts.check_dfm13_multilingual_evals import check_merge
from scripts.check_dfm13_dala_v2_existing21 import check_gec


def manifest(tmp_path,language):
    selected=tmp_path/'pairs.json'
    selected.write_text(json.dumps([dict(id=str(i),original='clean',corrupted='noisy',
        provenance={'split':'test'}) for i in range(4)]))
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(schema='dfm13-accepted-dala-heldout-v1',languages={language:dict(
        status='ready',split='test',view='representative',pairs=4,samples_per_task=8,
        selected=dict(path=str(selected),sha256=hashlib.sha256(selected.read_bytes()).hexdigest()),
        input_pins=[],prompts=dict(acceptability='yes/no',correction='correct'),quality='fixture')})))
    return path


@pytest.mark.parametrize('language',module.LANGUAGES)
def test_versioned_construct_shard_score_merge(tmp_path,language):
    path=manifest(tmp_path,language)
    ids=[]
    for shard in range(4):
        task=module.dala_v2_heldout(language,str(path),4,shard)
        samples=list(task.dataset);ids.extend(s.id for s in samples)
        assert task.dataset.name=='dala_v2_'+language
        assert [s.target for s in samples]==['correct','incorrect']
        asyncio.run(check_merge(language,samples,'dala_v2_'+language))
    assert len(ids)==len(set(ids))==8
    gec=module.gec_dala_v2_heldout(language,str(path))
    assert gec.dataset.name=='gec_dala_v2_'+language
    asyncio.run(check_gec(list(gec.dataset),'gec_dala_v2_'+language))


def test_no_training_fallback_or_new_language_alias(tmp_path):
    path=manifest(tmp_path,'da');data=json.loads(path.read_text())
    data['languages']['da']['split']='train';path.write_text(json.dumps(data))
    with pytest.raises(ValueError):module.dala_v2_heldout('da',str(path))
    with pytest.raises(ValueError):module.dala_v2_heldout('lt',str(path))
