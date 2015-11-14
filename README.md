# mecab-learning
MeCab界隈のツール

## digen.py
MeCab辞書の再構築ツール
- 単語の追加
- CRFパラメータの再学習

### usage
```
usage: dicgen.py [-h] [-l LEXEME] [-e TEST_CORPUS] [-t TMP]
                 [DIC] [MODEL] [CORPUS] [OUTPUT_DIC]

dicgen: MeCab辞書の再学習ヘルプツール

例: IPA辞書(2.7.0-20070801)をコーパスを使用して再学習させる
    python script/dicgen.py mecab-ipadic-2.7.0-20070801 mecab-ipadic-2.7.0-20070801.model corpus myipadic -l lexeme

positional arguments:
  DIC                   辞書
  MODEL                 CRFモデル
  CORPUS                コーパス
  OUTPUT_DIC            出力辞書

optional arguments:
  -h, --help            show this help message and exit
  -l LEXEME, --lexeme LEXEME
                        追加単語
  -e TEST_CORPUS, --test-corpus TEST_CORPUS
                        評価用コーパス
  -t TMP, --tmp TMP     一時ディレクトリ
```
