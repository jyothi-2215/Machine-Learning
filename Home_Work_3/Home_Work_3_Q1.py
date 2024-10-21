import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class datapreprocessing:
    def __init__(self,data):
        self.data = data
        self.normalized_data = None
        self.standardized_data = None
        self.iqr_data=None

    def Normalized(self):
        self.normalized_data = (self.data - self.data.min())/(self.data.max()-self.data.min())

    def Standardized(self):
        self.standardized_data = (self.data - self.data.mean())/(self.data.std())

    def IQR(self):
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3-Q1
        self.iqr_data = (self.data - Q1)/IQR

    def Show_original(self):
        plt.figure(figsize=(6,4))
        for column in self.data.columns:
            plt.plot(self.data[column],label=column)
        plt.title("Original AAPL data set")
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def Show_normalized(self):
        if self.normalized_data is None:
            self.Normalized()
            plt.figure(figsize=(6, 4))
            for column in self.normalized_data.columns:
                plt.plot(self.normalized_data[column], label=column)
            plt.title("Normalized AAPL data set")
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

    def Show_standardized(self):
        if self.standardized_data is None:
            self.Standardized()
            plt.figure(figsize=(6, 4))
            for column in self.standardized_data.columns:
                plt.plot(self.standardized_data[column], label=column)
            plt.title("Standardized AAPL data set")
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

    def Show_IQR(self):
        if self.iqr_data is None:
            self.IQR()
            plt.figure(figsize=(6, 4))
            for column in self.iqr_data.columns:
                plt.plot(self.iqr_data[column], label=column)
            plt.title("IQR transformation AAPL data set")
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()



