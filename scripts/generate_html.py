"""Search all sub-directories/sub-folders for files named "article.md".

Invoke `article.py` on found `article.md` files to convert the .md files to HTML pages.
"""
from article import convert_md_to_html
from os import walk as os_walk
from os.path import join as os_path_join
from os.path import abspath
import sys

ARTICLE_FILENAME = "article.md"
WEBSITE_FILENAME = "index.html"


def search_folders() -> list:
    article_paths = []

    # Nested loops are undesirable, but the iteration space is small enough to ignore performance.
    for dirpath, dirnames, filenames in os_walk(abspath("../blog/")):
        # Iterate through all files in /blog/ directory to find article.md files
        if ARTICLE_FILENAME in filenames:
            article = os_path_join(dirpath, ARTICLE_FILENAME)

            # Ignore article.md files that have already been converted to HTML files.
            if WEBSITE_FILENAME in filenames:
                # Ideally, we'd like to compare hashes to see if the article's contents differ
                # from the HTML file, and update the HTML file accordingly.
                # However, the above solution would overwrite manual edits made to the HTML.
                # We'll ignore articles with existing HTML files in the meantime.
                print(
                    f"[-] Ignoring article in dir {dirpath}, found {WEBSITE_FILENAME}")
            else:
                print(f"[+] Found article in dir {dirpath}: {article}")
                article_paths.append(abspath(path=article))

    return article_paths


if __name__ == "__main__":
    # Pass in a relative/absolute path to article.md as a command-line argument
    # ! NOTE: This will overwrite the existing index.html file, if present
    # * Example: python3 scripts/generate_html.py second_post/article.md
    # *          python3 scripts/generate_html.py ./second_post/article.md
    # *          python3 ../scripts/generate_html.py article.md
    if len(sys.argv) == 2:
        article = abspath(sys.argv[1])
        article_split = article.split('/')
        # Make sure filename is article.md and is located within the blog directory before
        # converting the file to index.html
        if article_split[-1] == ARTICLE_FILENAME and article_split[-3] == 'blog':
            convert_md_to_html(article=article)
        else:
            print(f"ERROR: Make sure CLI argument is a file named {ARTICLE_FILENAME} located in a "
                  "subdirectory within the blog directory")
    else:
        articles = search_folders()
        for article in articles:
            convert_md_to_html(article=article)
