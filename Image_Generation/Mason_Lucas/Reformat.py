from nbformat import read, write

with open("ImageGeneration.ipynb", "r", encoding="utf-8") as f:
    nb = read(f, as_version=4)

nb.metadata.pop("widgets", None)

with open("ImageGeneration.ipynb", "w", encoding="utf-8") as f:
    write(nb, f)
