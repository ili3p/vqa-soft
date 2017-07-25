import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import argparse
import sys
import numpy as np
import ujson
import operator 
from time import strftime, localtime

def log(msg):
    msg = str(msg)
    log_prefix ='['+strftime("%H:%M:%S", localtime())+']  process_answers.py:\033[1;32m '
    print(log_prefix + msg + '\033[0m')


parser = argparse.ArgumentParser(description='Process answers')
parser.add_argument('--input_train')
parser.add_argument('--input_val')
parser.add_argument('--output_qid2ans')
parser.add_argument('--output_qid2type')
parser.add_argument('--output_qid2anstype')
parser.add_argument('--output_ans_id2str')
parser.add_argument('--ans_type')
parser.add_argument('--answer_count', type=int)
parser.add_argument('--train_on_val', type=bool)
args = parser.parse_args()

counts = {}
mapping = {}
allans = {}
answers = []
qid2ans = {}
ans2id = {}
qid2type = {}
qid2anstype = {}

log('Train on val is '+str(args.train_on_val))
log('Reading '+args.input_train)
data = ujson.load(open(args.input_train))
log('Processing '+args.input_train)
for answer in data['annotations']:
    if args.ans_type != 'all' and answer['answer_type'] != args.ans_type:
        continue
    if answer['answer_type'] == 'yes/no':
        skip = False
        for a in answer['answers']:
            if not (a['answer'] == 'no' or a['answer'] == 'yes'):
                skip = True
                break
        if skip:
            continue
    ans_str = answer['multiple_choice_answer']
    qid = answer['question_id']
    qid2type[qid] = answer['question_type']
    qid2anstype[qid] = answer['answer_type']
    mapping[ans_str] = mapping[ans_str] if mapping.has_key(ans_str) else []
    mapping[ans_str].append(qid)
    cnt = counts[ans_str] if counts.has_key(ans_str) else 0
    counts[ans_str] = cnt + 1
    allans[qid] = allans[qid] if allans.has_key(qid) else []
    for a in answer['answers']:
        allans[qid].append(a['answer'])

if args.train_on_val:
    log('Processing ' + args.input_val)
    data = ujson.load(open(args.input_val))
    for answer in data['annotations']:
        if args.ans_type != 'all' and  answer['answer_type'] != args.ans_type:
            continue
        ans_str = answer['multiple_choice_answer']
        qid = answer['question_id']
        qid2type[qid] = answer['question_type']
        qid2anstype[qid] = answer['answer_type']
        mapping[ans_str] = mapping[ans_str] if mapping.has_key(ans_str) else []
        mapping[ans_str].append(qid)
        cnt = counts[ans_str] if counts.has_key(ans_str) else 0
        counts[ans_str] = cnt + 1
        allans[qid] = allans[qid] if allans.has_key(qid) else []
        for a in answer['answers']:
            allans[qid].append(a['answer'])

sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)


log('Preparing answer set.')
# get most frequent answers
for i in range(min(args.answer_count, len(sorted_counts))):
    ans_str = sorted_counts[i][0]
    answers.append(ans_str)
    ans2id[ans_str] = i + 1 # lua is 1-index 



answer_count = len(qid2type) # + 214354  # train + val, one per question not all 10

all_questions = 0
# log(answer_count)
# log(answers)
if not args.train_on_val:
    # process val answers
    log('Processing '+args.input_val+' as test set.')
    data = ujson.load(open(args.input_val))
    for answer in data['annotations']:
        if args.ans_type != 'all' and  answer['answer_type'] != args.ans_type:
            continue
        all_questions = all_questions + 1
        ans_str = answer['multiple_choice_answer']
        qid = answer['question_id']
        qid2type[qid] = answer['question_type']
        qid2anstype[qid] = answer['answer_type']
        if mapping.has_key(ans_str):
            mapping[ans_str].append(qid)
            allans[qid] = allans[qid] if allans.has_key(qid) else []
            for a in answer['answers']:
                allans[qid].append(a['answer'])

log('All questions: '+str(all_questions))
log('Preparing mappings...')
# make que_id to ans_id mapping
for ans_str in answers:
    for qid in mapping[ans_str]:
        qid2ans[qid] = [ans2id[ans_str]] # first element is the MC answer
        for a in allans[qid]:
            if ans2id.has_key(a): 
                qid2ans[qid].append(ans2id[a])
            else:
                qid2ans[qid].append(-1) # the answer is not in the most freq

id2ans = {}
for k in ans2id.keys():
    id2ans[ans2id[k]] = k


log('qid2ans size: '+str(len(qid2ans)))
log('qid2type size: '+str(len(qid2type)))
log('qid2anstype size: '+str(len(qid2anstype)))
log('id2ans size: '+str(len(id2ans)))

log('Saving data...')
ujson.dump(qid2ans, open(args.output_qid2ans,'w'))
ujson.dump(qid2type, open(args.output_qid2type,'w'))
ujson.dump(qid2anstype, open(args.output_qid2anstype,'w'))
ujson.dump(id2ans, open(args.output_ans_id2str,'w'))



# plot the percentange of questions vs number of answers

# all_answers = 443757 + 214354  # train + val, one per question not all 10
all_answers = 0
for i in xrange(len(sorted_counts)):
    all_answers = all_answers + sorted_counts[i][1]
c = 0

log('ALL answers: ' + str(all_answers))
p = []
for i in xrange(len(sorted_counts)):
    c = c + sorted_counts[i][1]
    p.append(100.*c/all_answers)

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.plot(range(min(22000,len(answers))),p[:min(22000, len(answers))],label='VQAv2')
ax.grid(True)
ax.set_ylabel("Percentage of questions covered")
ax.set_xlabel("Number of top K answers")
gridlines = ax.get_xgridlines() + ax.get_ygridlines()
for line in gridlines:
    line.set_linewidth(0.5)
    line.set_color('lightgray')
ax.set_xlim((0,min(22000, len(answers))))
# ax.xaxis.set_ticks(np.arange(0,min(22001, len(answers)), 1000))
# ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: str(x/1000)))
# ax.set_ylim((50,100))
# ax.yaxis.set_ticks(np.arange(50,101, 5))
ax.legend()
plt.tight_layout()
fig.savefig('coverage_'+args.ans_type.replace('/','-')+'.jpg')

h = []
for i in xrange(min(args.answer_count, len(sorted_counts))):
    c = sorted_counts[i][1]
    h.append(100.*c/answer_count)

# plot pie of classes
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.pie(h)
plt.tight_layout()
fig.savefig('classbalance_'+args.ans_type.replace('/','-')+'.jpg')
