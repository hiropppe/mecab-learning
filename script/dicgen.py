#!/usr/bin/env python
# -*- coding: utf-8 -*-

import MeCab
import sys, glob, codecs, os, re, shutil, subprocess, math, random, argparse, textwrap
from collections import defaultdict
from datetime import *
from tabulate import tabulate

def dic_gen(dic, model, corpus, output, lexeme):
    print '[GenDic] Setup seed dictionary'
    seed = make_seed(dic, lexeme)

    print '[GenDic] Build binary seed dictionary'
    mecab_dict_index(seed)
    
    print '[GenDic] Train CRF model'
    new_model = '.'.join([output, 'model'])
    mecab_cost_train(model, seed, corpus, new_model, c="1.0")
    
    print '[GenDic] Generate distribution dictionary'
    mecab_dict_gen(seed, new_model, output)
    
    print '[GenDic] Build binary dictionary'
    mecab_dict_index(output)

def make_seed(dic, lexeme):
    seed = os.path.join(tmp, os.path.basename(dic))
    os.makedirs(seed)

    files = glob.glob(os.path.join(dic, '*'))
    for f in [f for f in files if f.endswith('.csv') or f.endswith('.def') or os.path.basename(f) == 'dicrc']:    
        shutil.copy2(f, seed)

    if not lexeme is None:
        if os.path.isfile(lexeme):
            shutil.copy2(lexeme, seed)
        elif os.path.isdir(lexeme):
            for csv in glob.glob(os.path.join(lexeme, '*.csv')):
                shutil.copy2(csv, seed)
        else:
            pass

    return seed

def mecab_system_eval(result, answer, score):
    args = ['mecab-system-eval', '-l', '"0 1 2"', result, answer]
    result = call_mecab(args, 'mecab-system-eval', False)
    print result[1]
    with codecs.open(score, 'w', 'utf-8') as f:
        f.write(result[1])

def print_system_eval_average(eval_dirs):
    print 'Average Score:'
    level_results = defaultdict(lambda: defaultdict(lambda: []))
    for dir in eval_dirs:
        with codecs.open(dir + '/score', 'r', 'utf-8') as f:
            for lr in parse_mecab_system_eval(f):
                level_results[lr[0]]['p'].append(lr[1]) # precision
                level_results[lr[0]]['r'].append(lr[2]) # recall
                level_results[lr[0]]['f'].append(lr[3]) # F

    table = []
    for level in sorted(level_results.keys()):
        score_map = level_results[level]
        p = score_map['p']
        r = score_map['r']
        f = score_map['f']
        table.append(['LEVEL ' + level, sum(p)/len(p), sum(r)/len(r), sum(f)/len(f)])

    print tabulate(table, headers=['', 'precition', 'recall', 'F'])

def parse_mecab_system_eval(score_file):
    level_lst = []
    skip_header = False
    for line in score_file:
        if skip_header:
            cols = re.split(r'\s+', line)
            level = get_level(cols[1])
            precision =  get_score(cols[2])
            recall = get_score(cols[3])
            f = float(cols[4])
            level_lst.append((level, precision, recall, f))
        else:
            skip_header = True
    return level_lst

def get_level(level):
    return level[0:len(level)-1]

def get_score(score):
    return float(score[0:score.index('(')])

def mecab_parse(dic, text_file, result_file):
    mecab = MeCab.Tagger('-Ochasen -d ' + dic)
    with codecs.open(text_file, 'r', 'utf-8') as fi:
        with codecs.open(result_file, 'w', 'utf-8') as fo:
            for s in fi:
                encoded_text = s.encode('utf-8')
                node = mecab.parseToNode(encoded_text)
                node = node.next
                while node:
                    if node.feature.split(',')[0] == 'BOS/EOS':
                        fo.write('EOS\n')
                    else:
                        fo.write(node.surface + '\t' + node.feature + '\n')
                    node = node.next

def setup_cross_train_source():
    train_cross_dir = train_dir + '/cross'
    os.mkdir(train_cross_dir)

    n = 0
    files = glob.glob(corpus_corpus_dir + '/*.cabocha')
    docs = []
    for f in files:
        docs.append(os.path.basename(f))
        n += 1

    random.shuffle(docs)
    
    k = (int)(1 + math.log(n)/math.log(2))
    d = n/k
   
    eval_dirs = []
    for i in range(k):
        eval_dir = train_cross_dir + '/eval-' + `i`
        eval_dirs.append(eval_dir)
        os.mkdir(eval_dir)

        test_docs = docs[i*d:(i+1)*d]
        train_docs = list(set(docs) - set(test_docs))

        setup_each_cross_train(train_docs, test_docs, eval_dir)    
    
    return eval_dirs

def setup_each_cross_train(train_docs, test_docs, eval_dir):
    gen_corpus(train_docs, eval_dir + '/train')
    gen_corpus(test_docs, eval_dir + '/test')
    gen_test_corpus(eval_dir + '/test', eval_dir + '/test.plain')

def gen_test_corpus(src, dst):
    with codecs.open(src, 'r', 'utf-8') as fi:
        with codecs.open(dst, 'w', 'utf-8') as fo: 
            sent = []
            for lex in fi:
                if not lex.startswith('*') and re.match(r'^EOS\s*$', lex) is None:
                #if re.match(r'EOS\s*$', lex) == None:
                    sent.append(lex[0:lex.index('\t')])
                else:
                    if sent:
                        fo.write(''.join(sent) + '\n')
                    del sent[:]

def call_mecab(args, name, fail_on_err=True):
    print '[MeCab] {}'.format(args)
    p = subprocess.Popen(' '.join(args),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=mecab_env)
    stdout, stderr = p.communicate()
    
    if p.returncode != 0:
        if stderr:
            print stderr
        if fail_on_err:
            raise Exception('{} failed !'.format(args[0]))

    return (p.returncode, stdout, stderr)

def mecab_dict_index(dic, out=None):
    if out == None:
        out = dic
    
    args = ['mecab-dict-index',
        '-d', dic,
        '-o', out,
        '-t', 'utf-8',
        '-f', 'utf-8']
    call_mecab(args, 'mecab-dict-index')

def mecab_cost_assign(model, dic, input, output):
    args = ['mecab-dict-index',
        '-m', model,
        '-d', dic,
        '-a', input,
        '-u', output,
        '-t', 'utf-8',
        '-f', 'utf-8']
    call_mecab(args, 'mecab-dict-index')

def mecab_cost_train(model, dic, corpus, new_model, c="1.0"):
    args = ['mecab-cost-train', 
        '-M', model, 
        '-d', dic,
        '-c', c,
        corpus, new_model]
    call_mecab(args, 'mecab-cost-train')

def mecab_dict_gen(dic, model, output):
    os.makedirs(output)
    args = ['mecab-dict-gen',
        '-d', dic,
        '-o', output,
        '-m', model]
    call_mecab(args, 'mecab-dict-gen')

def gen_corpus(doc_names, dst):
    with codecs.open(dst, 'a', 'utf-8') as fo:
        for doc in doc_names:
            with codecs.open(corpus_corpus_dir + '/' + doc, 'r', 'utf-8') as fi:
                for line in fi:
                    if not line.startswith('* '):
                        fo.write(line)

def eval_dic(eval_corpus):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
            dicgen: MeCab辞書の再学習ヘルプツール
        
            例: IPA辞書(2.7.0-20070801)をコーパスを使用して再学習させる
                python script/dicgen.py mecab-ipadic-2.7.0-20070801 mecab-ipadic-2.7.0-20070801.model corpus myipadic -l lexeme
        ''')
        )
    parser.add_argument('dic', metavar='DIC', nargs='?', help='辞書')
    parser.add_argument('model', metavar='MODEL', nargs='?', help='CRFモデル')
    parser.add_argument('corpus', metavar='CORPUS', nargs='?', help='コーパス')
    parser.add_argument('output', metavar='OUTPUT_DIC', nargs='?', help='出力辞書')
    parser.add_argument('-l', '--lexeme', help='追加単語')
    parser.add_argument('-e', '--eval-corpus', help='評価用コーパス')
    parser.add_argument('-t', '--tmp', help='一時ディレクトリ')
    args = parser.parse_args()

    if args.dic is None or args.model is None or args.corpus is None or args.output is None:
        parser.print_usage()
        sys.exit(0)
    
    global mecab_env
    mecab_env = os.environ.copy()
    mecab_env['PATH'] = '/usr/local/libexec/mecab:' + mecab_env.get('PATH', '')
    
    global tmp
    tmp = args.tmp
    if tmp is None:
        tmp = os.path.join('.tmp', '_'.join(['train', datetime.now().strftime('%Y%m%d%H%M%S')]))
    os.makedirs(tmp)

    dic_gen(args.dic, args.model, args.corpus, args.output, args.lexeme)

    if not args.eval_corpus is None:
        eval_dic(args.eval_corpus)

