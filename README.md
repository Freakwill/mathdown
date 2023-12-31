# mathdown
`mathdown` is a markdown parser of mathematics notes, based on `pyparsing`.

## Introduction
I have made many notes on various theories in mathematics using Markdown. I need a unified format to record them for future reference. Alternatively, they can be compiled into LaTeX to generate PDFs.

## Syntax
Following is the syntax of so-called math notes.

### Paragraph

paragraphs are areas of texts, seperated by `\n\n`

```
This is a paragraph;
This is the second line of the paragraph.

This is another paragraph

The solution to the equation $f(x)=0$ is
$$
x=a+1
$$
where $a$ is ...

I hope that mathdown can help you to learn math!
```

### Block
A block is a paragraph with a keyword.


#### Theorem-type
```
**Definition**
(X, *) is a group, if ...
where ...
```

```
**Theorem**
If a, b is two points of a Hilbert space, then we have
$$
|<a,b>| \leq \|a\|\|b\|
$$
Specially, in Euclidean space, ....

*Proof* Let $b = a + \lambda t,\lambda\in\R$. where $a\perp t$ and $\|t\|=1$
...
```

Or

#### Remark-type
```
*Remark* Here X is a finite set.
```

```
*Example* the set of rational numbers Q is a group.
```

