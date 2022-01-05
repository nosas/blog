# Blog

January 4, 2022<br><br>

## Repo Purpose
---

I have a passion for learning and sharing my findings but I've struggled to find a suitable framework for quickly writing blog posts. WordPress is the goto for blogging, but there's too much bloated overhead for creating a simple blog post.

Last week, I stumbled upon [Fabien Sanglard's website](https://fabiensanglard.net/) on HackerNews and it matched the exact vibe I was after. As I was drafting an email inquiring about his website's framework, I clicked a post named: [0x10 Rules](https://fabiensanglard.net/ilike/index.html).

It turns out websites are as easy as converting one writing format into HTML and prettying it up with simple CSS. Fabien developed his framework with PHP and I can do just the same with Python.

This repo is my rendition of a simple web framework which converts markdown files to static HTML pages. Thanks Fabien.


## Framework
---

The HTML pages are generated from markdown using Python. A master script called "generate_html.py" searches all sub-directories for files name "article.md" and executes "article.py" to convert the `.md` files to `index.html`.

