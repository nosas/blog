# Convert .md file to static .html
# markdown2 article.md -x code-friendly,fenced-code-blocks,footnotes,header-ids,tables > index.html
from datetime import date

import markdown2

EXTRAS = ["code-friendly", "fenced-code-blocks", "footnotes", "header-ids", "tables"]
HEADER = """
<html>

<base href="https://sasonreza.com/">

<head>
  <link rel="stylesheet" type="text/css" href="../css/default_dark.css">
  <link rel="stylesheet" type="text/css" href="../css/syntax_dark.css">
</head>

<body>
  <center>
      <div style="display: inline-block; vertical-align:middle;">
          <a href="/" style="text-decoration: none;">SASON REZA<br>
          </a>
          <hr>
          <div style="text-align: center;display: inline-block; width: 100%;">
              <a class="title" href="/books">BOOKS</a> &nbsp;<a class="title" href="/about">ABOUT</a> &nbsp;<a
                  class="title" href="/contact">CONTACT</a>
          </div>
      </div>
  </center>

  <br>
  <p style="margin-bottom: 2ch;text-align: right;font-style: italic;">{date}</p>

"""  # noqa
FOOTER = """
</body>
</html>
"""


def convert_md_to_html(article: str, depth: int = 1):
    article_dir = article.strip("article.md")

    with open(article) as a:
        article_contents = "".join(line for line in a.readlines())

    article_html = markdown2.markdown(text=article_contents, extras=EXTRAS)
    header = HEADER.format(date=date.today().strftime("%B %d, %Y"))
    header = header.replace("../", "../" * depth)

    with open(f"{article_dir}/index.html", "w+") as f:
        f.write(header)
        f.write(article_html)
        f.write(FOOTER)
    print(f"\t[+] Finished converting {article}, don't forget to add the title!")
