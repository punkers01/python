import csv
import urllib
import numpy
import matplotlib.pyplot as plt

import datetime


def get_data(url):
    response = urllib.urlopen(url).readlines()
    cr = csv.DictReader(response)
    stock_train = []
    # with open('/Users/Fabio/Documents/aapl.csv') as csvfile:
    # cr = csv.DictReader(csvfile)

    for row in cr:
        stock_train.append(
            (float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), float(row["Volume"])))

        labels_train = [None] * len(stock_train)
        labels_train[0] = 0
        for idx, _ in enumerate(stock_train):
            if idx > 0:
                if stock_train[idx] >= stock_train[idx - 1]:
                    labels_train[idx] = 1
                else:
                    labels_train[idx] = 0

                    # for row in labels_train:
                    #   print (row)

    return stock_train, labels_train


def main():
    start = datetime.date(int("2010"), int("01"), int("01"))
    end = datetime.date(int("2015"), int("12"), int("31"))
    url_string = "http://www.google.com/finance/historical?q={0}".format("AAPL")
    url_string += "&startdate={0}&enddate={1}&output=csv".format(
        start.strftime('%b %d, %Y'), end.strftime('%b %d, %Y'))

    stock_train, labels_train = get_data(url_string)

    train_data = numpy.array(stock_train)
    train_label = numpy.array(labels_train)

    start = datetime.date(int("2015"), int("01"), int("01"))
    end = datetime.date(int("2015"), int("12"), int("31"))
    url_string = "http://www.google.com/finance/historical?q={0}".format("AAPL")
    url_string += "&startdate={0}&enddate={1}&output=csv".format(
        start.strftime('%b %d, %Y'), end.strftime('%b %d, %Y'))

    stock_test, labels_test = get_data(url_string)

    from sklearn import svm

    clf = svm.SVC(kernel="poly")
    clf.fit(train_data, train_label)

    from sklearn.naive_bayes import GaussianNB

    # clf = GaussianNB()
    # clf.fit(train_data, train_label)
    # data 23oct pred 26 oct
    # print(clf.predict([118.77, 119.23, 116.33, 119.08, 59139552]))

    test_data = numpy.array(stock_test)
    test_label = numpy.array(labels_test)

    accuracy = []
    count = 0
    while count < 201:
        accuracy.append(clf.score(test_data[count:count + 10], test_label[count:count + 10]))
        count += 10

    print("done")

    plt.plot(numpy.arange(0, 210, 10), accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('days')
    plt.axis([0, 200, 0, 1])
    plt.show()


# 'http://www.google.com/finance/historical?q=AAPL&startdate=Jan 01, 2011&enddate=Dec 31, 2013&output=csv'


main()
