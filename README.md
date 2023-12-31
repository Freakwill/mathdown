# mathdown
parse markdown of math note based on pyparsing

## Paragraph

paragraphs are areas of texts, seperated by `\n\n`

```
This is a paragraph

This is another paragraph
```

## Block
A block is a paragraph with a keyword.


### Theorem-type
```
**Definition**
(X, *) is a group, if ...
```

```
**Theorem**
If a, b is two points of a Hilbert space, then we have
$$
|<a,b>| \leq \|a\|\|b\|
$$
```

Or

### Remark-type
```
*Remark* Here X is a finite set.
```

```
*Example* the set of rational numbers Q is a group.
```

