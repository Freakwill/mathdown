#!/usr/bin/env python

from .constant import *


class KeywordError(Exception):

    def __init__(self, keyword=''):
        self.keyword = keyword

    def __str__(self):
        return f'`{self.keyword}` is a keyword, could not be used here!'


def indent(x):
    if isinstance(x, str):
        x = x.strip().split('\n')
    return '\n'.join(['  '+i.strip() for i in x])


def latex_comment(x):
    if isinstance(x[0], str):
        x = x[0].strip().split('\n')
    return '\n'.join(['%  '+i.strip() for i in x])


def span_join(x):
    if len(x) <= 1:
        return x
    else:
        res = x[0]
        for xi, xj in zip(x[1:], x[:-1]):
            if xj != '\n':
                if xj[-1] not in MARKS_CN + '$`*' and xi[0] not in '$`*\\':
                    xi = ' ' + xi
            res += xi
        return res


def add_item(x):
    if isinstance(x, str):
        x = x.strip().split('\n')
    return '\\item ' + x[0].strip() + '\n'.join(['  '+i.strip() for i in x[1:]])


class LatexAction:

    def __str__(self):
        return self.to_latex()


class LatexEnv(LatexAction):

    def __init__(self, env='align', name=None):
        self.env = env
        self.name = name
        
    def __call__(self, t):
        name = self.name or t.get('name', None)
        content = t['body']
        if name is None:
            return f"""\\begin{{{self.env}}}
{indent(content)}
\\end{{{self.env}}}"""
        else:
            return f"""\\begin{{{self.env}}}[{name}]
{indent(content)}
\\end{{{self.env}}}"""


class Equation(LatexEnv):

    def __call__(self, t):
        content = t['body']
        if '\n' in indent(content):
            self.env = 'align'
        else:
            self.env = 'equation'
        reutrn super().__call__(t)


class LatexCommand(LatexAction):

    def __init__(self, name):
        self.name = name

    def __call__(self, t):
        return f"\\{self.name}{{{t['body']}}}"


class Textbf(LatexCommand):

    def __init__(self):
        super().__init__(name='textbf')


class Textit(LatexCommand):

    def __init__(self):
        super().__init__(name='textit')


class Section(LatexCommand):

    def __init__(self):
        super().__init__('section')

    def __call__(self, t):
        n_sub = len(t['sharp']) - 2
        return f"\\{'sub' * n_sub +self.name}{{{t['body']}}}\n"


class Chapter(LatexCommand):

    def __init__(self):
        super().__init__('chapter')

    def __call__(self, t):
        return f"\\{self.name}{{{t['body']}}}\n"


class LatexList(LatexAction):

    def __init__(self, env='itemize', form=None):
        self.env = env
        self.form = form
        
    def __call__(self, t):
        form = self.form or t.get('form', None)
        content = t
        if form is None:
            return f"""\\begin{{{self.env}}}
{indent(content)}
\\end{{{self.env}}}"""
        else:
            return f"""\\begin{{{self.env}}}[{form}]
{indent(content)}
\\end{{{self.env}}}"""


def table_to_latex(t):
    import pandas as pd
    col = len(t['head'])
    assert all(l == col for l in map(len, t['body'])), Exception("t['body'] should has the same column with t['head']!")
    return pd.DataFrame(columns=t['head'], data=t['body']).to_latex(index=False)


def list_exercise(t):
    return '\\ex\n\n' + '\n\n'.join(f"""\\begin{{exercise}}
{ti}
\\end{{exercise}}""" for ti in t['exercises'])
