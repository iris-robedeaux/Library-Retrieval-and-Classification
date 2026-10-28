#load libraries
import pandas as pd
from src.libraryGeneration import *
from src.visualizationMethods import *

#load books in
allBooks = loadBooks("data/books.csv")

#sample from books
sample = sampleBooks(allBooks, 1000)

# create perfect library structure
library = classifyBooks(sample, shelfCapacity=25, reference=allBooks)

#mess up the library
messy, log = messLibraryUp(library, spookyLevel=15, movePct=50, seed=42, logMess=True)

drawLibrary(messy, maxDepth=2, log=log)

messy.listBooks()
list = messy.listBooks().query("misplaced")
