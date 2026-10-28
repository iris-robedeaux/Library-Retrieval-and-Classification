"""Visualization for the Haunted Library Catalog (needs matplotlib).

Works on a Library object from library_catalog.classifyBooks / messLibraryUp:

    from library_catalog import *
    from library_viz import drawLibrary
"""

from collections import Counter


def drawLibrary(library, maxDepth=2, log=None, figsize=(16, 7), title=None, ax=None):
    """Draw the library tree (needs matplotlib).

    Each circle is a section; its area is proportional to the number of books at
    or below it. Sections at `maxDepth` are shown collapsed (their shelves are
    summed up), so the picture stays readable. Colors follow top-level category.
    If you pass the `log` from messLibraryUp, sections that received strays are
    outlined in red and labeled with how many books landed there.
    Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    strays = Counter()
    if log is not None and len(log):
        def _count(sec):
            prefix = sec.path()
            n = sum(1 for dest in log["newShelf"]
                    if dest == prefix or dest.startswith(prefix + " > "))
            strays[prefix] = n
            for child in sec.children.values():
                _count(child)
        _count(library.root)

    # Layout: x = left-to-right order of terminal nodes, y = depth.
    pos, nBooks = {}, {}
    counter = [0]

    def layout(sec):
        key = id(sec)
        nBooks[key] = len(sec.allBooks())
        if sec.isLeaf() or sec.depth >= maxDepth:
            pos[key] = (counter[0], -sec.depth)
            counter[0] += 1
        else:
            xs = [layout(c) for c in sec.children.values()]
            pos[key] = (sum(xs) / len(xs), -sec.depth)
        return pos[key][0]

    layout(library.root)

    cats = list(library.root.children)
    cmap = plt.get_cmap("tab20")
    colorOf = {c: cmap(i % 20) for i, c in enumerate(cats)}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    maxBooks = max(nBooks.values())
    scale = 1800 / maxBooks

    def draw(sec, top):
        key = id(sec)
        x, y = pos[key]
        top = sec.name if sec.depth == 1 else top
        terminal = sec.isLeaf() or sec.depth >= maxDepth
        if not terminal:
            for child in sec.children.values():
                cx, cy = pos[id(child)]
                ax.plot([x, cx], [y, cy], color="0.7", lw=1, zorder=1)
                draw(child, top)
        color = "0.35" if sec.depth == 0 else colorOf[top]
        n = strays.get(sec.path(), 0)
        ax.scatter([x], [y], s=max(nBooks[key] * scale, 25), color=color, zorder=2,
                   edgecolor="red" if n else "white", linewidth=1.5 if n else 1)
        label = f"{sec.name}\n{nBooks[key]}" + (f"  (+{n})" if n else "")
        if sec.depth == 0:
            ax.annotate(label, (x, y), xytext=(0, 14), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold")
        elif terminal:
            ax.annotate(label, (x, y), xytext=(0, -16), textcoords="offset points",
                        ha="center", va="top", fontsize=7, rotation=60,
                        color="red" if n else "black")
        else:
            ax.annotate(label, (x, y), xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=7, color="red" if n else "black")

    draw(library.root, None)
    ax.set_title(title or f"Library structure (shown to depth {maxDepth})")
    ax.margins(0.08, 0.35)
    ax.axis("off")
    return ax