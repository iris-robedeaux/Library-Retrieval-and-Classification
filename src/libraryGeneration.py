"""Haunted Library Catalog: build a category/ISBN library tree, then mess it up."""

import copy
import random
from collections import Counter, defaultdict

import pandas as pd

KEEP_COLS = ["isbn13", "title", "authors", "categories",
             "published_year", "average_rating", "num_pages"]

FIRST_DIGIT = 3   # skip the 978/979 prefix
LAST_DIGIT = 12   # index 12 is the check digit, so only digits 0..11 are split on


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def loadBooks(path="data/books.csv"):
    """Read the CSV, keep the useful columns, and make ISBNs clean 13-char strings."""
    df = pd.read_csv(path, dtype={"isbn13": str})
    df = df[KEEP_COLS].dropna(subset=["isbn13"]).copy()
    df["isbn13"] = (df["isbn13"].astype(str)
                    .str.replace(r"\.0$", "", regex=True)
                    .str.zfill(13))
    return df.drop_duplicates("isbn13").reset_index(drop=True)


def sampleBooks(books, sampSize=1000, seed=None):
    """Random sample of books."""
    return books.sample(sampSize, random_state=seed)


# --------------------------------------------------------------------------
# Guessing missing categories from the ISBN
# --------------------------------------------------------------------------
# An ISBN does not encode subject, but its leading digits identify the language
# group and the publisher, and publishers tend to stick to a few subjects. So we
# guess a missing category from labeled books sharing the longest ISBN prefix.
def _prefixIndex(reference, minLen, maxLen):
    index = {}
    for isbn, cat in zip(reference["isbn13"], reference["categories"]):
        if pd.isna(cat):
            continue
        for L in range(minLen, maxLen + 1):
            index.setdefault(isbn[:L], Counter())[cat] += 1
    return index


def _guessCategory(isbn, index, minLen, maxLen):
    for L in range(maxLen, minLen - 1, -1):
        counts = index.get(isbn[:L])
        if counts:
            return counts.most_common(1)[0][0]
    return None


# --------------------------------------------------------------------------
# The library structure
# --------------------------------------------------------------------------
class Section:
    """One node in the library tree: the root, a category, or an ISBN-prefix subsection.

    Only leaf sections (shelves) store books directly; every other section
    reports the books of everything beneath it via allBooks().
    `node` is a spare slot for attaching a spatial Node (x, y) later, when the
    shelves get placed in rooms for BFS/DFS/A*/Greedy.
    """

    def __init__(self, name, parent=None, depth=0):
        self.name = name
        self.parent = parent
        self.depth = depth
        self.children = {}
        self.books = []      # ISBN strings, leaves only
        self.node = None

    def isLeaf(self):
        return not self.children

    def allBooks(self):
        out = list(self.books)
        for child in self.children.values():
            out.extend(child.allBooks())
        return out

    def leaves(self):
        if self.isLeaf():
            return [self]
        out = []
        for child in self.children.values():
            out.extend(child.leaves())
        return out

    def path(self):
        parts, sec = [], self
        while sec is not None:
            parts.append(sec.name)
            sec = sec.parent
        return " > ".join(reversed(parts))

    def __repr__(self):
        return f"Section({self.name!r}, {len(self.allBooks())} books, {len(self.children)} children)"


class Library:
    """Wraps the tree plus the book info (indexed by ISBN)."""

    def __init__(self, root, info):
        self.root = root
        self.info = info

    def leaves(self):
        return self.root.leaves()

    def locate(self, isbn):
        """All shelves holding this ISBN (more than one if it was duplicated)."""
        return [leaf for leaf in self.leaves() if isbn in leaf.books]

    def find(self, where):
        """Return a Section from a Section, a full path ("Fiction > 9780*"), or a unique name."""
        if isinstance(where, Section):
            return where
        if where is None:
            return self.root
        allSecs, stack = [], [self.root]
        while stack:
            sec = stack.pop()
            allSecs.append(sec)
            stack.extend(sec.children.values())
        for sec in allSecs:
            if sec.path() in (where, "Library > " + where):
                return sec
        byName = [sec for sec in allSecs if sec.name == where]
        if len(byName) == 1:
            return byName[0]
        if not byName:
            raise KeyError(f"No section matches {where!r}")
        raise KeyError(f"{where!r} is ambiguous; use a longer path. Matches: "
                       + ", ".join(sec.path() for sec in byName[:5]))

    def listBooks(self, where=None):
        """DataFrame of every shelved copy at or below a section (default: whole library).

        `where` can be a Section, a path like "Fiction > 9780*", or a unique section name.
        The `misplaced` column is True when a copy sits on a shelf other than its home
        shelf, so it is all False in a clean library and flags strays after messLibraryUp.
        """
        sec = self.find(where)
        rows = []
        for leaf in sec.leaves():
            shelf = leaf.path()
            for isbn in sorted(leaf.books):
                rec = self.info.loc[isbn]
                rows.append({"shelf": shelf, "isbn13": isbn, "title": rec["title"],
                             "authors": rec["authors"], "category": rec["category"],
                             "misplaced": shelf != rec["home"]})
        return pd.DataFrame(rows, columns=["shelf", "isbn13", "title", "authors",
                                           "category", "misplaced"])

    def toDataFrame(self):
        """One row per shelved copy in the whole library (same as listBooks())."""
        return self.listBooks()

    def copy(self):
        return copy.deepcopy(self)

    def show(self, maxDepth=2):
        def walk(sec, indent):
            if sec.depth > maxDepth:
                return
            print("  " * indent + f"{sec.name}  ({len(sec.allBooks())})")
            for child in sec.children.values():
                walk(child, indent + 1)
        walk(self.root, 0)


def _build(section, isbns, digit, shelfCapacity):
    """Split a section on the next informative ISBN digit until shelves are small enough."""
    if len(isbns) <= shelfCapacity:
        section.books = list(isbns)
        return
    # Skip digit positions where every book agrees (avoids empty one-child chains).
    while digit < LAST_DIGIT and len({i[digit] for i in isbns}) == 1:
        digit += 1
    if digit >= LAST_DIGIT:
        section.books = list(isbns)   # can't split further (e.g. duplicate-ish ISBNs)
        return
    groups = defaultdict(list)
    for isbn in isbns:
        groups[isbn[digit]].append(isbn)
    for d, grp in sorted(groups.items()):
        name = grp[0][:digit + 1] + "*"
        child = Section(name, parent=section, depth=section.depth + 1)
        section.children[name] = child
        _build(child, grp, digit + 1, shelfCapacity)


def classifyBooks(books, shelfCapacity=25, reference=None,
                  guessMissing=True, minPrefix=6, maxPrefix=10,
                  minCategorySize=10, fallbackName="General Stacks"):
    """Build the perfectly classified library.

    Root -> one section per category -> ISBN-prefix subsections, split until every
    shelf holds at most `shelfCapacity` books. Books with a missing category get a
    guess from ISBN prefixes in `reference` (default: `books` itself), else go to
    'Uncategorized'.

    minCategorySize: categories with fewer books than this (in this collection) get
    no section of their own. Their books are shelved together in a fallback section
    (`fallbackName`) organized purely by ISBN prefix, like a catalog-number stacks
    area. Set to 1 to give every category its own section. (Each book's true
    category is still kept in `Library.info`.)
    """
    books = books.copy()
    books["isbn13"] = books["isbn13"].astype(str)
    books["category"] = books["categories"]

    if guessMissing:
        ref = books if reference is None else reference
        index = _prefixIndex(ref, minPrefix, maxPrefix)
        missing = books["category"].isna()
        books.loc[missing, "category"] = [
            _guessCategory(i, index, minPrefix, maxPrefix)
            for i in books.loc[missing, "isbn13"]
        ]
    books["category"] = books["category"].fillna("Uncategorized")

    root = Section("Library")
    sizes = books["category"].value_counts()
    rare = set(sizes[sizes < minCategorySize].index)

    for cat, grp in books[~books["category"].isin(rare)].groupby("category"):
        section = Section(cat, parent=root, depth=1)
        root.children[cat] = section
        _build(section, sorted(grp["isbn13"]), FIRST_DIGIT, shelfCapacity)

    if rare:
        fallback = Section(fallbackName, parent=root, depth=1)
        root.children[fallbackName] = fallback
        rareIsbns = sorted(books.loc[books["category"].isin(rare), "isbn13"])
        _build(fallback, rareIsbns, FIRST_DIGIT, shelfCapacity)

    info = books.set_index("isbn13")
    info["home"] = pd.Series({isbn: leaf.path()
                              for leaf in root.leaves() for isbn in leaf.books})
    return Library(root, info)


# --------------------------------------------------------------------------
# Haunting
# --------------------------------------------------------------------------
def messLibraryUp(library, spookyLevel=15, movePct=50, seed=None, logMess=False):
    """Return a messed-up copy of the library.

    spookyLevel: percent of books affected (each at most once).
    movePct: percent of affected books that are MOVED to a random other shelf;
             the rest are DUPLICATED onto one (the original stays put).
    logMess: if True, return (library, log DataFrame) with the ground truth.
    """
    rng = random.Random(seed)
    lib = library.copy()
    leaves = lib.leaves()

    pairs = [(isbn, leaf) for leaf in leaves for isbn in leaf.books]
    nSpooked = round(len(pairs) * spookyLevel / 100)
    chosen = rng.sample(pairs, nSpooked)

    log = []
    for isbn, src in chosen:
        targets = [leaf for leaf in leaves if leaf is not src]
        if not targets:
            continue
        dest = rng.choice(targets)
        action = "move" if rng.random() * 100 < movePct else "duplicate"
        if action == "move":
            src.books.remove(isbn)
        dest.books.append(isbn)
        log.append({"isbn13": isbn,
                    "title": lib.info.loc[isbn, "title"],
                    "action": action,
                    "trueShelf": src.path(),
                    "newShelf": dest.path()})

    if logMess:
        return lib, pd.DataFrame(log)
    return lib


# --------------------------------------------------------------------------
if __name__ == "__main__":
    allBooks = loadBooks("data/books.csv")
    sample = sampleBooks(allBooks, 1000, seed=42)
    library = classifyBooks(sample, shelfCapacity=25, reference=allBooks,
                            minCategorySize=10)
    library.show(maxDepth=2)

    messy, log = messLibraryUp(library, spookyLevel=15, movePct=50, seed=42, logMess=True)
    print(log.head())