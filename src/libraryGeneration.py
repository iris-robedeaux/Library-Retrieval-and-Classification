#load in libraries
import pandas as pd

#read in the data
bookDt = pd.read_csv("data/books.csv")

#only retain best columns
bookDt = bookDt[["isbn13", "title", "authors", "categories", "published_year", "average_rating", "num_pages"]]

#take a random sample of 1k books
def sampleBooks(sampSize = 1000):
    return bookDt.sample(sampSize)

#classify perfectly based on isbn
