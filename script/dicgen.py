#!/usr/bin/env python
# -*- coding: utf-8 -*-

import MeCab
import sys, glob, codecs, os, re, shutil, subprocess, math, random, argparse, textwrap
from collections import defaultdict
from datetime import *
from tabulate import tabulate

dicname_bin = 'dic'
dicname_model = 'model'
dicname_train = 'train'

target_dic = 'ipa'
target_version = None
target_corpus = None

corpus_dir = None
corpus_lex_dir = None
corpus_corpus_dir = None

dic_dir = None
train_id = None
train_dir = None
train_dir_dic_org = None
train_dir_model_org = None
train_dir_corpus = None
train_dir_model = None
train_dir_dic = None

def train(dicdir, dicname, version, corpus):
    print '[GenDic] Train {} dictionary ({}) using latest {} corpus'.format(dicname, version, corpus)
    init_train_env(dicdir, dicname, version, corpus)
    gen_train_base_dict()
    run_cross_train_and_eval()
    release_dict()

def gen_train_base_dict():
    print '[GenDic] Generate train base dictionary ...'
    copy_original_dic_and_model(train_dir_dic_org, train_dir)
    copy_add_lexes(train_dir_dic_org)
    mecab_dict_index(train_dir_dic_org, train_dir_dic_org)

def run_cross_train_and_eval():
    print '[GenDic] Run cross train and evaluation ...'
    eval_dirs = setup_cross_train_source()
    for eval_dir in eval_dirs:
        train_and_eval(eval_dir)
    print_system_eval_average(eval_dirs)

def train_and_eval(eval_dir):
    print '[GenDic] Eval ' + eval_dir

    train_corpus = eval_dir + '/train'
    test_answer = eval_dir + '/test'
    test_plain = eval_dir + '/test.plain'
    test_result = eval_dir + '/test.result'
    test_score = eval_dir + '/score'
    new_model = eval_dir + '/' + target_version + '.model'
    new_dic = eval_dir + '/' + target_version
    
    mecab_cost_train(train_dir_model_org, train_dir_dic_org, train_corpus, new_model)
    mecab_dict_gen(train_dir_dic_org, new_model, new_dic)
    mecab_dict_gen_index(new_dic, new_dic)
    mecab_parse(new_dic, test_plain, test_result)
    mecab_system_eval(test_result, test_answer, test_score)

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

def release_dict():
    gen_release_dict()
    deploy_release_dict()

def gen_release_dict():
    print '[GenDic] Generate release dictionary ...'
    setup_release_train()
    
    mecab_cost_train(train_dir_model_org, train_dir_dic_org, train_dir_corpus, train_dir_model)
    print 'New Model {} generated !'.format(train_dir_model)
    
    mecab_dict_gen(train_dir_dic_org, train_dir_model, train_dir_dic)
    mecab_dict_gen_index(train_dir_dic, train_dir_dic)
    print 'New Dictionary {} generated !'.format(train_dir_dic)

def deploy_release_dict():
    print '[GenDic] Deploy new dictionary'
    release_path = target_corpus + '/dic/mecab-' + target_dic + 'dic-' + target_version + '-latest'
    if os.path.exists(release_path):
        shutil.rmtree(release_path)
    shutil.copytree(train_dir_dic, release_path)

def call_mecab(args, name, fail_on_err=True):
    print '[MeCab] {} ...'.format(name)
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
            raise Exception('{} failed !'.format(name))

    return (p.returncode, stdout, stderr)

def mecab_dict_index(dic, out):
    args = ['mecab-dict-index',
        '-d', dic,
        '-o', out,
        '-t', 'utf-8',
        '-f', 'utf-8']
    call_mecab(args, 'mecab-dict-index')

def mecab_cost_train(model, dict, corpus, new_model):
    args = ['mecab-cost-train', 
        '-M', model, 
        '-d', dict, 
        corpus, new_model]
    call_mecab(args, 'mecab-cost-train')

def mecab_dict_gen(dic, model, new_dic):
    os.mkdir(new_dic)
    shutil.copy(train_dir_dic_org + '/dicrc', new_dic)
    shutil.copy(train_dir_dic_org + '/pos-id.def', new_dic)
    args = ['mecab-dict-gen',
        '-d', dic,
        '-o', new_dic,
        '-m', model]
    call_mecab(args, 'mecab-dict-gen')

def mecab_dict_gen_index(dic, out):
    args = ['mecab-dict-index',
        '-d', dic,
        '-o', out,
        '-t', 'utf-8',
        '-f', 'utf-8']
    call_mecab(args, 'mecab-dic-index')

def setup_release_train():
    print ' Train directory ' + train_dir 
    
    global train_dir_corpus
    global train_dir_model
    global train_dir_dic
    
    train_dir_corpus = train_dir + '/train'
    train_dir_model = train_dir_model_org + '.train'
    train_dir_dic = train_dir_dic_org + '.gen'
   
    docs = [os.path.basename(f) for f in glob.glob(corpus_corpus_dir + '/*.cabocha')]
    gen_corpus(docs, train_dir_corpus)

def copy_original_dic_and_model(dst_dic, dst_model):
    original_dic = dic_dir + '/dic/' + target_version
    original_model = dic_dir + '/model/' + target_version + '.model'
    shutil.copytree(original_dic, dst_dic)
    shutil.copy(original_model, dst_model)

def copy_train_dic_and_model(dst_dic, dst_model):
    shutil.copytree(train_dir_dic_org, dst_dic)
    shutil.copy(train_dir_model_org, dst_model)

def init_train_env(dicdir, dicname, dicver, corpus):
    global mecab_env
    mecab_env = os.environ.copy()
    mecab_env['PATH'] = '/usr/local/libexec/mecab:' + mecab_env.get('PATH', '')
    
    global target_dic
    global target_version
    global target_corpus
    global dic_dir
    global corpus_dir
    global corpus_lex_dir
    global corpus_corpus_dir
    global train_id
    global train_dir
    global train_dir_dic_org
    global train_dir_model_org

    target_dic = dicname
    target_version = dicver
    target_corpus = corpus

    corpus_dir = corpus
    corpus_lex_dir = corpus_dir + '/mecab/dict'
    corpus_corpus_dir = corpus_dir + '/mecab/learn'

    dic_dir = dicdir + '/' + dicname + '/' + dicver
    train_id = datetime.now().strftime('%Y%m%d%H%M%S')
    train_dir = dic_dir + '/' + dicname_train + '/' + train_id
    train_dir_dic_org = train_dir + '/' + dicver
    train_dir_model_org = train_dir_dic_org + '.model'
    
    print ' Target dictionary ' + dic_dir
    print ' Target Corpus ' + corpus_dir
    print ' Train directory ' + train_dir

    os.mkdir(train_dir)

def copy_add_lexes(dst):
    files = glob.glob(corpus_lex_dir + '/*.csv')
    for f in files:
        shutil.copy(os.path.abspath(f), dst) 

def gen_corpus(doc_names, dst):
    with codecs.open(dst, 'a', 'utf-8') as fo:
        for doc in doc_names:
            with codecs.open(corpus_corpus_dir + '/' + doc, 'r', 'utf-8') as fi:
                for line in fi:
                    if not line.startswith('* '):
                        fo.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
            dicgen: MeCab辞書の再学習ヘルプツール
        
            例: IPA辞書(2.7.0-20070801)をコーパスを使用して再学習させる
                python script/dicgen.py dic/ ipa 2.7.0-20070801 corpus/sample_doc/
        ''')
        )
    parser.add_argument('dic_dir', metavar='DICTIONARY DIRECTORY',
                        nargs='?', help='辞書管理ルートディレクトリ')
    parser.add_argument('dic_name', metavar='DICTIONARY NAME',
                        nargs='?', help='ターゲット辞書名')
    parser.add_argument('dic_version', metavar='DICTIONARY VERSION',
                        nargs='?', help='ターゲット辞書バージョン')
    parser.add_argument('corpus_dir', metavar='CORPUS DIRECTORY',
                        nargs='?', help='ターゲットコーパスディレクトリ')
    args = parser.parse_args()

    if args.dic_dir is None or args.dic_name is None or args.dic_version is None or args.corpus_dir is None:
        parser.print_usage()
        sys.exit(0)

    train(args.dic_dir, args.dic_name, args.dic_version, args.corpus_dir)
