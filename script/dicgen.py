#!/usr/bin/env python
# -*- coding: utf-8 -*-

import MeCab
import sys, glob, codecs, os, re, shutil, subprocess, math, random, argparse, textwrap
from collections import defaultdict
from datetime import *
from tabulate import tabulate

def gen_dic(dic, model, corpus, output, lexeme):
    print '[DicGen] Setup seed dictionary'
    seed = make_seed(dic, lexeme)

    print '[DicGen] Build binary seed dictionary'
    mecab_dict_index(seed)
    
    print '[DicGen] Train CRF model'
    new_model = '.'.join([output, 'model'])
    mecab_cost_train(model, seed, corpus, new_model)
    
    print '[DicGen] Generate distribution dictionary'
    mecab_dict_gen(seed, new_model, output)
    
    print '[DicGen] Build binary dictionary'
    mecab_dict_index(output)

def eval_dic(test_corpus, dic):
    test_answer = os.path.join(tmp, 'test.answer')
    test_input = os.path.join(tmp, 'test.input')
    test_output = os.path.join(tmp, 'test.output')
    
    with codecs.open(test_answer, 'w', 'utf-8') as outfile:
        for f in glob.glob(os.path.join(test_corpus, '*')):
            with codecs.open(f, 'r', 'utf-8') as infile:
                outfile.write(infile.read())
    
    gen_plain_corpus(test_answer, test_input)
    
    mecab_parse(dic, test_input, test_output)

    mecab_system_eval(test_output, test_answer, os.path.join(tmp, 'score'))

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

def gen_plain_corpus(src, dst):
    with codecs.open(src, 'r', 'utf-8') as fi:
        with codecs.open(dst, 'w', 'utf-8') as fo: 
            sent = []
            for lex in fi:
                if not lex.startswith('*') and re.match(r'^EOS\s*$', lex) is None:
                    sent.append(lex[0:lex.index('\t')])
                else:
                    if sent:
                        fo.write(''.join(sent) + '\n')
                    del sent[:]

def mecab_parse(dic, text_file, result_file):
    mecab = MeCab.Tagger('-Ochasen -d {}'.format(dic))
    with codecs.open(result_file, 'w', 'utf-8') as outfile:
        with codecs.open(text_file, 'r', 'utf-8') as infile:
            for sent in infile:
                encoded_text = sent.encode('utf-8')
                node = mecab.parseToNode(encoded_text)
                node = node.next
                while node:
                    if node.feature.split(',')[0] == 'BOS/EOS':
                        outfile.write('EOS\n')
                    else:
                        outfile.write(node.surface.decode('utf-8') + '\t' + node.feature.decode('utf-8') + '\n')
                    node = node.next

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

def mecab_system_eval(result, answer, score):
    args = ['mecab-system-eval', '-l', '"0 1 2"', result, answer]
    result = call_mecab(args, 'mecab-system-eval', False)
    with codecs.open(score, 'w', 'utf-8') as f:
        f.write(result[1])

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

    print stdout

    return (p.returncode, stdout, stderr)

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
    parser.add_argument('-e', '--test-corpus', help='評価用コーパス')
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

    gen_dic(args.dic, args.model, args.corpus, args.output, args.lexeme)

    if not args.test_corpus is None:
        eval_dic(args.test_corpus, args.output)

