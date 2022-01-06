# Convert .md file to static .html
# markdown2 article.md --use-file-vars=../scripts/markdown2_extras > index.html
# markdown2 article.md -x code-friendly,fenced-code-blocks,footnotes,header-ids,tables > index.html
import markdown2


EXTRAS = ["code-friendly", "footnotes", "header-ids", "tables"]

# Public APIs, to be used in `generate_html.py`
