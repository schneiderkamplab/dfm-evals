"""Additional accepted-v2 tasks; historical task names/data remain unchanged."""
from pathlib import Path
from inspect_ai import task
from dfm_evals.tasks.dala_dfm13_heldout import make

LANGUAGES=tuple('da en nb nn sv is fo nl pl de fr es it cs pt_pt fi et ca el ro uk'.split())
DEFAULT_MANIFEST=str(Path(__file__).resolve().parents[3]/'config/dfm13_dala_v2_existing21_20261006.json')


@task(name='dala_v2_heldout')
def dala_v2_heldout(language:str,manifest:str=DEFAULT_MANIFEST,num_shards:int=1,shard_index:int=0,max_gen_toks:int=32):
    return make(language,'acceptability',manifest,num_shards,shard_index,max_gen_toks,
                LANGUAGES,'dala_v2_'+language)


@task(name='gec_dala_v2_heldout')
def gec_dala_v2_heldout(language:str,manifest:str=DEFAULT_MANIFEST,num_shards:int=1,shard_index:int=0,max_gen_toks:int=512):
    return make(language,'correction',manifest,num_shards,shard_index,max_gen_toks,
                LANGUAGES,'gec_dala_v2_'+language)
