#!/usr/bin/env python

from .constant import *


class KeywordError(Exception):

    def __init__(self, keyword=''):
        self.keyword = keyword

    def __str__(self):
        return f'`{self.keyword}` is a keyword, could not be used here!'


def indent(x, leading=' '):
    if isinstance(x, str):
        x = x.strip().split('\n')
    return '\n'.join([leading + ' '+i.strip() for i in x])


def insert_and(xs):
    x = xs[0]
    next_ = False
    for k in range(len(x)):
        for c in RELATIONS:
            if x[k:].startswith(c):
                xs[0] = x[:k] + ' & ' + x[k:]
                next_ = True
                break
        if next_: break
    for j, x in enumerate(xs[1:], 1):
        for c in RELATIONS:
            if x.startswith(c):
                xs[j] = ' & ' + x
        else:
            next_ = False
            for k in range(len(x)):
                for c in RELATIONS:
                    if x[k:].startswith(c):
                        xs[j] = x[:k] + ' & ' + x[k:]
                        next_ = True
                        break
                if next_: break
    return xs


def latex_comment(x):
    return indent(x, leading='%')


def span_join(x):
    if len(x) <= 1:
        return x
    else:
        res = x[0]
        for xi, xj in zip(x[1:], x[:-1]):
            if xj[-1] != '\n':
                if xj[-1] not in MARKS_CN + '$`*' and xi[0] not in '$`*\\':
                    res += ' '
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

    def __init__(self, env=None, default_option=None):
        self.env = env or self.__class__.__name__.lower()
        self.default_option = default_option
        
    def __call__(self, t):
        self.set_content(t)
        if not isinstance(self.content, str):
            self.content = '\n'.join(self.content)
        if self.option is None:
            return f"""\\begin{{{self.env}}}
{indent(self.content)}
\\end{{{self.env}}}"""
        else:
            return f"""\\begin{{{self.env}}}[{self.option}]
{indent(self.content)}
\\end{{{self.env}}}"""

    def set_content(self, t):
        self.option = t.get('name', None) or self.default_option
        self.content = indent(t['body'])


class Equation(LatexEnv):

    def set_content(self, t):
        super().set_content(t)
        if '\n' in self.content:
            self.env = 'align'
            self.content = insert_and(self.content.split('\n'))
        else:
            self.env = 'equation'


class LatexCommand(LatexAction):

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__.lower()

    def __call__(self, t):
        return f"\\{self.name}{{{t['body']}}}"


class Textbf(LatexCommand):
    pass


class Textit(LatexCommand):
    pass

class Texttt(LatexCommand):
    pass


class Section(LatexCommand):

    def __call__(self, t):
        n_sub = len(t['sharp']) - 2
        return f"\\{'sub' * n_sub +self.name}{{{t['body']}}}\n"


class Chapter(LatexCommand):

    def __call__(self, t):
        return super().__call__(t) + "\n"


class LatexList(LatexEnv):

    def __init__(self, env=None, form=None):
        self.env = env
        self.option = form
        
    def set_content(self, t):
        self.option = self.option or t.get('form', None)
        self.content = indent(t, '\\item')


def table_to_latex(t):
    import pandas as pd
    col = len(t['head'])
    assert all(l == col for l in map(len, t['body'])), Exception("t['body'] should has the same column with t['head']!")
    caption = t.get('caption', None)
    return pd.DataFrame(columns=t['head'], data=t['body']).to_latex(index=False, caption=caption)

def list_exercise(t):
    return '\\ex\n\n' + '\n\n'.join(f"""\\begin{{exercise}}
{ti}
\\end{{exercise}}""" for ti in t['exercises'])


class Figure(LatexEnv):

    def set_content(self, t):
        self.option = "h"
        self.content = f"""\\centering
\\includegraphics[width=0.8\\textwidth]{{{t['path']}}}
\\caption{{{t['caption']}}}
\\label{{fig:figure-label}}"""


class Algorithm(LatexEnv):

    def set_content(self, t):
        super().set_content(t)
        body = indent(t['body'], "\\State")
        require = t.get('input', None)
        if require:
            self.content = f"""\\caption{{{t.get('caption', 'No caption')}}}
    \\begin{{algorithmic}}
      \\Require {t['input']}
      \\Ensure {t['return']}
      {body}
    \\end{{algorithmic}}"""
        else:
            self.content = f"""\\caption{{{t.get('caption', 'No caption')}}}
    \\begin{{algorithmic}}
      {body}
    \\end{{algorithmic}}"""
