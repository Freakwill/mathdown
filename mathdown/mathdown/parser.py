#!/usr/bin/env python

import pyparsing as pp
import pyparsing.unicode as ppu

from .action import *
from .constant import *


pp.ParserElement.setDefaultWhitespaceChars(" ")
pp.ParserElement.enable_packrat()

NL = pp.LineEnd()
single_newline = NL + ~pp.FollowedBy(NL)
strict_single_newline = NL + ~pp.FollowedBy(NL | pp.one_of(BEGIN_SYMBLES + KEYWORDS_STAR) | (pp.common.integer+pp.Literal('. ').leave_whitespace()))

one_or_more_newlines = NL[1, ...].suppress()
zero_or_more_newlines = NL[0, ...]
two_newlines = NL + NL

language = True
if language is None:
    marks = MARKS
    chars = pp.alphanums + marks
else:
    marks = MARKS + MARKS_CN
    chars = pp.alphanums + marks + ppu.Chinese.alphas

chars_ext = chars + '-<>+'

word = pp.Word(chars, chars_ext)
word_ext = pp.Word(chars_ext)
words = pp.OneOrMore(word)
digit = pp.Word(pp.nums)

div = pp.Forward()
equation = pp.Combine((pp.QuotedString('$$', multiline=True)|pp.QuotedString('\\[', end_quote_char='\\]', multiline=True)).set_parse_action(lambda t: t[0])('body').set_parse_action(Equation()) + pp.Optional(single_newline + '其中' + div))
inline = pp.QuotedString('$', unquote_results=False, multiline=True)
bold = pp.QuotedString('**', multiline=False)('body').set_parse_action(Textbf())
italy = pp.QuotedString('*', multiline=False)('body').set_parse_action(Textit())
quote = pp.QuotedString('`', multiline=False).set_parse_action(lambda t: f"\\verb|{t[0]}|")
center = pp.QuotedString('<center>', end_quote_char='</center>', multiline=True)('body').set_parse_action(LatexEnv('center'))
code = pp.QuotedString('```', multiline=True)('body').set_parse_action(LatexEnv('lstlisting'))

figure = pp.Suppress('!') + pp.QuotedString('[', end_quote_char=']')('captain') + pp.QuotedString('(', end_quote_char=')')('path')
figure.set_parse_action(Figure())

span = inline | quote | bold | italy | word
span_ext = inline | quote | bold | italy | word_ext
row = pp.Combine(span + pp.ZeroOrMore(span_ext), adjacent=False)
title = pp.Combine(pp.OneOrMore(pp.Combine((inline  | word) + pp.Optional(word_ext), adjacent=False)), adjacent=False)
div <<= pp.OneOrMore(equation | row | strict_single_newline).set_parse_action(span_join)
ulist = pp.DelimitedList((pp.Suppress('- ') + div).set_parse_action(lambda t: '\\item '+t[0]), one_or_more_newlines).set_parse_action(LatexList())
olist = pp.DelimitedList((pp.Suppress(pp.common.integer + pp.Literal('. ').leave_whitespace()) + div).set_parse_action(lambda t: '\\item '+t[0]), one_or_more_newlines).set_parse_action(LatexList(env='enumerate'))

def table_row(p):
    return pp.Suppress('|') + pp.DelimitedList(p, pp.Suppress('|')) + pp.Suppress('|')

table_item = pp.OneOrMore(span).set_parse_action(span_join) | '-'
table_head = table_row(table_item)
table_line = table_row(pp.OneOrMore('-'))
table_body = pp.DelimitedList(pp.Group(table_row(table_item)), single_newline.suppress())
table = table_head('head') + single_newline.suppress() + table_line.suppress() + single_newline.suppress() + table_body('body')
table.set_parse_action(tabular)
table_with_captain = center('captain').set_parse_action() + table().set_parse_action()
table_with_captain.set_parse_action(table)

paragraph = code | table_with_captain | center | figure | table | pp.DelimitedList(ulist | olist | div, single_newline)

remark_key = pp.Suppress('*注*')
example_key = pp.Suppress('*例*')
proof_key = pp.Suppress('*证明*')
exercise_key = pp.Suppress('*练习*')
ref_key = pp.Suppress('*参考文献*')
def_key = pp.Suppress('**定义') + pp.Optional(pp.Suppress(pp.nums)) + pp.Optional( (pp.QuotedString('(', end_quote_char=')') |pp.QuotedString('[', end_quote_char=']') | pp.QuotedString('（', end_quote_char='）') | title).set_parse_action(lambda t: t[0])('name')) + pp.Suppress('**')
fact_key = pp.Suppress('**事实') + pp.Optional((pp.QuotedString('（', end_quote_char='）') | title)('name')) + pp.Suppress('**')
algo_key = pp.Suppress('**算法') + pp.Optional(title('name')) + pp.Suppress('**')
thm_key = pp.Suppress('**定理') + pp.Optional((pp.QuotedString('[', end_quote_char=']') | pp.QuotedString('（', end_quote_char='）') | title).set_parse_action(lambda t: t[0])('name')) + pp.Suppress('**')
den_key = pp.Suppress('**约定')  + pp.Optional(pp.Suppress(pp.nums)) + pp.Optional(title('name')) + pp.Suppress('**')

remark = remark_key + paragraph('body')
remark.set_parse_action(LatexEnv('remark'))
remarks = pp.DelimitedList(remark, one_or_more_newlines).set_parse_action('\n\n'.join)

example = example_key + paragraph('body')
example.set_parse_action(LatexEnv('example'))

fact = fact_key  + zero_or_more_newlines + paragraph('body')
fact.set_parse_action(LatexEnv('fact'))

definition = def_key + zero_or_more_newlines + paragraph('body')
definition.set_parse_action(LatexEnv('definition'))

algorithm = algo_key + zero_or_more_newlines + paragraph('body')
algorithm.set_parse_action(LatexEnv('algorithm'))

theorem = thm_key + zero_or_more_newlines + paragraph('body')
theorem.set_parse_action(LatexEnv('theorem'))

proof = proof_key + zero_or_more_newlines + paragraph('body')
proof.set_parse_action(LatexEnv('proof'))

denotation = den_key + zero_or_more_newlines + paragraph('body')
denotation.set_parse_action(LatexEnv('denotation'))

exlist = pp.DelimitedList((pp.Suppress(pp.common.integer + pp.Literal('. ').leave_whitespace()) + div), one_or_more_newlines)
exercise = exercise_key + one_or_more_newlines + exlist('exercises')
exercise.set_parse_action(list_exercise)

references = ref_key
references.set_parse_action(lambda t: r"\bibliography{bib-file}")

chapter_title = pp.Literal('#')('sharp') + title('body') + pp.FollowedBy(NL)
chapter_title.set_parse_action(LatexCommand('chapter'))

section_title = pp.Literal('##')('sharp') + title('body') + pp.FollowedBy(NL)
section_title.set_parse_action(LatexCommand('section'))
subsection_title = '###' + title('body') + pp.FollowedBy(NL)
subsection_title.set_parse_action(LatexCommand('subsection'))
subsubsection_title = '####' + title('body') + pp.FollowedBy(NL)
subsubsection_title.set_parse_action(LatexCommand('subsubsection'))

comment = pp.QuotedString('<!--', end_quote_char='-->', multiline=True)('body').set_parse_action(latex_comment)

block = (definition | fact | pp.Combine(theorem + pp.Optional(one_or_more_newlines + proof), join_string='\n') | algorithm | remarks | example)

text = pp.DelimitedList(comment | block | paragraph, one_or_more_newlines).set_parse_action('\n\n'.join)

subsubsection = subsubsection_title + one_or_more_newlines + text
subsection = subsection_title + pp.Optional(one_or_more_newlines + text) + pp.Optional(one_or_more_newlines  + pp.DelimitedList(subsubsection, one_or_more_newlines))
subsection.set_parse_action('\n\n'.join)
section = section_title + pp.Optional(one_or_more_newlines + text) + pp.Optional(one_or_more_newlines + pp.DelimitedList(subsection, one_or_more_newlines))
section.set_parse_action('\n\n'.join)

hrule = pp.Literal('---').set_parse_action(lambda t: r'\vspace{2\baselineskip}\hrule')
chapter = (zero_or_more_newlines + chapter_title + pp.Optional(one_or_more_newlines + pp.Suppress('[TOC]')) + pp.Optional(one_or_more_newlines + text) + one_or_more_newlines + pp.DelimitedList(section, one_or_more_newlines)
    + pp.Optional(one_or_more_newlines + hrule + one_or_more_newlines + exercise)+ pp.Optional(one_or_more_newlines + references))
chapter.set_parse_action('\n\n'.join)
