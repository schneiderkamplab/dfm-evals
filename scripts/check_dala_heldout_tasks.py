"""Offline actual Inspect task loading and raw training-template context checks."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'dfm-evals'))
from dfm_evals.tasks import dala_multilingual_heldout as tasks


def main():
    import jinja2
    import yaml
    from tokenizers import Tokenizer
    from inspect_ai._eval.loader import load_task_spec
    manifest=ROOT/'config/dfm_dala_heldout_20260930.json'
    config=ROOT/'config/dfm_evals_multilingual_dala_heldout.yaml'
    tokenizer_path=ROOT/'data/dfm11_tokenizer/tokenizer.json'
    template_path=ROOT/'data/dfm11_tokenizer/chat_template.jinja'
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    template=jinja2.Environment().from_string(template_path.read_text())
    sets=yaml.safe_load(config.read_text())['sets'];report={}
    for name,definition in sets.items():
        item=definition['tasks'][0]
        arguments=dict(value.split('=',1) for value in item['args'] if value!='-T')
        budget=int(arguments['max_gen_toks']);arguments['max_gen_toks']=budget
        loaded=load_task_spec(item['name'],arguments)
        if len(loaded)!=1:raise ValueError('Task selector is ambiguous')
        task=loaded[0];max_prompt=max_target=0;ids=[]
        for row in task.dataset:
            rendered=template.render(messages=[dict(role='user',content=row.input)],tools=[],
                add_generation_prompt=True,enable_thinking=False,bos_token='<bos>',eos_token='<eos>')
            prompt_tokens=len(tokenizer.encode(rendered,add_special_tokens=False).ids)
            target_tokens=len(tokenizer.encode(row.target if isinstance(row.target,str) else row.target[0],add_special_tokens=False).ids)
            if prompt_tokens+budget>4096:raise ValueError('Prompt context overflow: '+row.id)
            if target_tokens+1>budget:raise ValueError('Reference exceeds output budget: '+row.id)
            max_prompt=max(max_prompt,prompt_tokens);max_target=max(max_target,target_tokens);ids.append(row.id)
        if len(ids)!=2000 or len(set(ids))!=len(ids):raise ValueError('Incomplete task sample inventory')
        report[name]=dict(samples=len(ids),max_prompt_tokens=max_prompt,max_reference_tokens=max_target,
                          max_output_tokens=budget,context_limit=4096,truncated=0)
        print(name,report[name],flush=True)
    output=ROOT/'config/dfm_dala_heldout_preflight_20260930.json'
    receipt=dict(status='passed',tasks=report,total_samples=sum(r['samples'] for r in report.values()),
        tokenizer_mode='raw training tokenizer; no Mistral fix',
        pins={str(p):tasks.sha256(p) for p in (manifest,config,tokenizer_path,template_path,
            Path(tasks.__file__),Path(__file__),ROOT/'config/dfm_dala_heldout_registry_20260930.json')})
    with output.open('x') as f:json.dump(receipt,f,indent=2)


if __name__=='__main__':main()
