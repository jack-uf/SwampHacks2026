"""Opt-in paid evaluation on frozen, nonsensitive synthetic scenarios; never runs in CI."""
import argparse
import asyncio
import hashlib
import inspect
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from backend import jev
from scripts.common import ROOT

CASES = ROOT/'tests/fixtures/jev-scenarios.json'
PROTOCOL = ROOT/'docs/jev-evaluation-protocol.json'
SUITES = dict(frozen=dict(cases=CASES, protocol=PROTOCOL, size=10, output=ROOT/'output/jev-evaluation.jsonl'),
              heldout=dict(cases=ROOT/'tests/fixtures/jev-heldout.json', protocol=ROOT/'docs/jev-heldout-protocol.json', size=16,
                           output=ROOT/'output/jev-heldout-evaluation.jsonl'))
INPUT_PRICE_PER_MILLION = .042  # Official models page retrieved 2026-10-16; estimate, not an invoice.


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def plan(suite='frozen'):
    cases=json.loads(SUITES[suite]['cases'].read_text())
    assert len({c['id'] for c in cases})==len(cases)==SUITES[suite]['size']
    for c in cases:
        assert set(c['categories'])<=set(jev.CATEGORIES) and len(c['categories'])==len(set(c['categories']))
        assert c['routing'] in jev.ROUTES and c['urgency'] in [0,1,2]
    request=jev.request_for('')
    configuration=dict(corpus_sha256=digest(cases),questions_sha256=digest(request['questions']),model=request['model'],
        scenarios=len(cases),data_kind='author-labeled synthetic scenarios; not independently labeled field observations',
        acceptance=.7,rejection=.3,typed_confidence_review_below=.7,
        estimated_input_tokens_per_call=5000,estimated_total_usd=len(cases)*5000*INPUT_PRICE_PER_MILLION/1e6,
        maximum_calls=len(cases),automatic_retries=0)
    if suite=='heldout':
        configuration.update(abstention_rule=jev.ABSTENTION_RULE,
            abstention_rule_sha256=hashlib.sha256(inspect.getsource(jev.abstention).encode()).hexdigest(),
            comparison='v1 and v2 abstention scored on the same responses; review queue logic unchanged',
            provenance='Cases and labels written after inspecting the frozen ten-case results and before any held-out inference.')
    return cases,configuration


def abstention_comparison(rows):
    """Score the original (v1) and hazard-decisive (v2) abstention rules on identical responses."""
    successful=[r for r in rows if r['result']['status']=='classified']
    rescored=[(r,jev.interpret(dict(model=r['result']['model'],answers=r['result']['answers'],usage=r['result']['usage']))) for r in successful]
    def score(rule,key):
        abstained=[(r,x) for r,x in rescored if x[key]]
        decisive=[(r,x) for r,x in rescored if not x[key]]
        exact=lambda pair:set(pair[0]['expected']['categories'])==set(pair[1]['categories'])
        hits=sum(map(exact,decisive))
        return dict(rule=rule,responses=len(rescored),abstained=len(abstained),
                    abstention_rate=len(abstained)/len(rescored) if rescored else None,decisive=len(decisive),decisive_exact=hits,
                    decisive_exact_rate=hits/len(decisive) if decisive else None,decisive_wrong=len(decisive)-hits,
                    abstained_exact=sum(map(exact,abstained)))
    return dict(v1=score('v1','abstained_v1'),v2=score(jev.ABSTENTION_RULE,'abstained'),
                review_queue_changed=sum(r['result']['review_queue']!=x['review_queue'] for r,x in rescored),
                per_case=[dict(id=r['id'],expected=sorted(r['expected']['categories']),suggested=x['categories'],
                               abstained_v1=x['abstained_v1'],abstained_v2=x['abstained'],reasons_v2=x['abstention']['reasons'],
                               review_queue=x['review_queue']) for r,x in rescored])


def metrics(rows):
    successful=[r for r in rows if r['result']['status']=='classified']
    counts={k:dict(tp=0,fp=0,fn=0) for k in jev.CATEGORIES}
    disagreements=[]
    for row in successful:
        expected=set(row['expected']['categories']);predicted=set(row['result']['categories'])
        for k in counts:
            counts[k]['tp']+=int(k in expected & predicted)
            counts[k]['fp']+=int(k in predicted-expected)
            counts[k]['fn']+=int(k in expected-predicted)
        if expected!=predicted or row['expected']['routing']!=row['result']['routing']['choice']:
            disagreements.append(dict(id=row['id'],expected=sorted(expected),suggested=sorted(predicted),
                expected_route=row['expected']['routing'],suggested_route=row['result']['routing']['choice']))
    def ratios(c):
        return dict(**c,precision=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else None,
                    recall=c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else None)
    totals={key:sum(v[key] for v in counts.values()) for key in ['tp','fp','fn']}
    per_class={k:ratios(v) for k,v in counts.items()}
    def mean(values):return sum(values)/len(values) if values else None
    return dict(attempted=len(rows),successful=len(successful),failed=len(rows)-len(successful),
        response_coverage=len(successful)/len(rows) if rows else None,
        micro=ratios(totals),per_class=per_class,
        macro_precision=mean([v['precision'] for v in per_class.values() if v['precision'] is not None]),
        macro_recall=mean([v['recall'] for v in per_class.values() if v['recall'] is not None]),
        abstention_rate=mean([int(r['result']['abstained']) for r in successful]),
        uncertain_rate=mean([int(bool(r['result']['uncertain'])) for r in successful]),
        clarification_queue_rate=mean([int(r['result']['review_queue']=='clarification') for r in successful]),
        routing_accuracy=mean([int(r['expected']['routing']==r['result']['routing']['choice']) for r in successful]),
        urgency_mean_absolute_error=mean([abs(r['expected']['urgency']-r['result']['urgency']['score']) for r in successful]),
        mean_latency_seconds=mean([r['latency_seconds'] for r in rows]),disagreements=disagreements,
        note='Classification metrics use successful responses only; failures are separate. Abstentions retain any accepted labels and count normally. Null means undefined, not zero. Macro averages exclude undefined class ratios. Urgency labels are subjective author expectations, not safety ground truth.')


async def evaluate(output,suite='frozen'):
    cases,configuration=plan(suite)
    protocol=json.loads(SUITES[suite]['protocol'].read_text())
    if configuration!=protocol:raise SystemExit('Corpus/questions/model differ from frozen protocol. Predeclare a new evaluation before spending.')
    if os.getenv('PLANIT_JEV_ENABLED')!='true' or not os.getenv('TYPESAFE_API_KEY'):
        raise SystemExit('Live evaluation requires PLANIT_JEV_ENABLED=true and server-side TYPESAFE_API_KEY; no inference attempted.')
    output=Path(output)
    if output.exists():raise SystemExit('Output already exists; choose a new path. Never silently overwrite paid evidence.')
    output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    with output.open('x') as journal:
        journal.write(json.dumps(dict(kind='protocol',created_at=datetime.now(timezone.utc).isoformat(),configuration=configuration))+'\n');journal.flush()
        for c in cases:
            start=time.monotonic()
            result=await jev.classify(c['text'],consent=True)
            row=dict(id=c['id'],text=c['text'],expected={k:c[k] for k in ['categories','routing','urgency']},
                     request_sha256=digest(jev.request_for(c['text'])),result=result,latency_seconds=round(time.monotonic()-start,4))
            rows.append(row);journal.write(json.dumps(dict(kind='response',**row))+'\n');journal.flush()
            # Stop on any external/contract failure; never spend ten calls on a broken setup.
            if result['status']!='classified':break
        usage={k:sum(r['result'].get('usage',{}).get(k,0) for r in rows) for k in ['input_tokens','output_tokens']}
        summary=dict(kind='summary',metrics=metrics(rows),abstention_comparison=abstention_comparison(rows),usage=usage,
                     estimated_usd=usage['input_tokens']*INPUT_PRICE_PER_MILLION/1e6,
                     billing_note='Estimate from returned input tokens and published price; failed-call usage may be unknown. Not an invoice.')
        journal.write(json.dumps(summary)+'\n')
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',help='Authorize one-shot paid text requests (one per case); default only prints the plan.')
    parser.add_argument('--suite',choices=sorted(SUITES),default='frozen')
    parser.add_argument('--output',help='Defaults to the suite evidence path; existing files are never overwritten.')
    parser.add_argument('--compare',metavar='JSONL',help='Offline: score v1 and v2 abstention on saved responses; no network.')
    args=parser.parse_args()
    load_dotenv(ROOT/'.env',override=False)
    if args.compare:
        rows=[json.loads(line) for line in Path(args.compare).read_text().splitlines()]
        print(json.dumps(abstention_comparison([r for r in rows if r['kind']=='response']),indent=2))
    elif args.live:asyncio.run(evaluate(args.output or SUITES[args.suite]['output'],args.suite))
    else:print(json.dumps(plan(args.suite)[1],indent=2))


if __name__=='__main__':main()
