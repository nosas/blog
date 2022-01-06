"""Search all sub-directories/sub-folders for files named "article.md".

Invoke `article.py` on found `article.md` files to convert the .md files to HTML pages.
"""
# Search all sub-directories for files name "article.md"
from article import convert_md_to_html
from os import walk as os_walk
from os.path import join as os_path_join
from os.path import abspath


def search_folders() -> list:
    article_paths = []

    # Nested loops are undesirable, but the iteration space is small enough to ignore performance.
    for dirpath, dirnames, filenames in os_walk(abspath("../blog/")):
        # Iterate through all files in /blog/ directory to find article.md files
        for file in filenames:
            if file == "article.md":
                article = os_path_join(dirpath, file)

                # Ignore article.md files that have already been converted to HTML files.
                if "index.html" in filenames:
                    # Ideally, we'd like to compare hashes to see if the article's contents differ
                    # from the HTML file, and update the HTML file accordingly.
                    # However, the above solution would overwrite manual edits made to the HTML.
                    # We'll ignore articles with existing HTML files in the meantime.
                    print(f"[-] Ignoring article in dir {dirpath}, found index.html")
                else:
                    print(f"[+] Found article in dir {dirpath}: {article}")
                    article_paths.append(abspath(path=article))

    return article_paths


if __name__ == "__main__":
    articles = search_folders()
    for article in articles:
        convert_md_to_html(article=article)
