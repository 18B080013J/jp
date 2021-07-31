from django.shortcuts import render,HttpResponse
from .models import Product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import io, base64
import urllib
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
# Create your views here.

def home(request):
    return render(request, 'ML/base.html')

def data_prev(request):
    list = Product.objects.all().values()
    df = pd.DataFrame(list)
    context = {'det': df.to_html()}
    return render(request, 'ML/data_prev.html', context)

def select_features_algo(request):
    return render(request, 'ML/select_features.html')

def cluster(request):
    Amount = request.POST.get('Amount','0')
    NIFTY_PE=request.POST.get('NIFTY_PE','0')
    Type_of_Purchase = request.POST.get('Type_of_Purchase','off')
    algo = request.POST['algo']
    d2 = {"Amount": Amount, "NIFTY_PE": NIFTY_PE, "Type_of_Purchase": Type_of_Purchase}
    listdj = []
    for x, y in d2.items():
        if y == 'on':
            listdj.append(x)
    if algo=='1':
        list = Product.objects.all().values()
        df = pd.DataFrame(list)
        df2 = df[listdj]
        mms = MinMaxScaler()
        mms.fit(df2)
        X1 = mms.transform(df2)
        # pca = PCA(n_components=2)
        # pca.fit(X1)
        # X1=pca.transform(X1)
        model = KMeans(n_clusters=4)
        label = model.fit_predict(X1)
        df2['labels'] = label
        df2['Branch_Code'] = df['Branch_Code']
        u_labels = np.unique(label)
        for i in u_labels:
            plt.scatter(X1[label == i, 0], X1[label == i, 1], label=i)
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], color="black")
        plt.xlabel('Amount')
        plt.ylabel('NIFTY_PE')
        plt.title('cluster')
        plt.legend()
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        context = {'data': uri,
                   'det': df2.to_html()}
        return render(request, 'ML/cluster.html', context)
    elif algo=='2':
        list = Product.objects.all().values()
        df = pd.DataFrame(list)
        df2 = df[listdj]
        mms = MinMaxScaler()
        mms.fit(df2)
        X1 = mms.transform(df2)


        dbscan = DBSCAN(eps=0.1, min_samples=2)

        label = dbscan.fit_predict(X1)
        u_labels = np.unique(label)
        df2['labels'] = label
        df2['Branch_Code'] = df['Branch_Code']

        for i in u_labels:
            plt.scatter(X1[label == i, 0], X1[label == i, 1], label=i)

        plt.xlabel('Amount')
        plt.ylabel('NIFTY_PE')
        plt.title('cluster')
        plt.legend()
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        context = {'data': uri,
                   'det': df2.to_html()}
        return render(request, 'ML/cluster.html', context)
    elif algo == '3':
        list = Product.objects.all().values()
        df = pd.DataFrame(list)
        df2 = df[listdj]
        mms = MinMaxScaler()
        mms.fit(df2)
        X1 = mms.transform(df2)

        dbscan = AgglomerativeClustering(n_clusters=4)

        label = dbscan.fit_predict(X1)
        u_labels = np.unique(label)
        df2['labels'] = label
        df2['Branch_Code'] = df['Branch_Code']

        for i in u_labels:
            plt.scatter(X1[label == i, 0], X1[label == i, 1], label=i)

        plt.xlabel('Amount')
        plt.ylabel('NIFTY_PE')
        plt.title('cluster')
        plt.legend()
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        context = {'data': uri,
                   'det': df2.to_html()}
        return render(request, 'ML/cluster.html', context)
    # df3=df2[['Amount','NIFTY_PE']]
    #     #
    #     # u_labels = np.unique(label)
    #     # dic = {}
    #     # for i in u_labels:
    #     #     dic[i] = {}
    #     #
    #     #     filtered_label = X1[label == i]
    #     #     sum0 = 0
    #     #     sum1 = 0
    #     #     count = 0
    #     #     for j in filtered_label:
    #     #         sum0 = sum0 + j[0]
    #     #         sum1 = sum1 + j[1]
    #     #         count = count + 1
    #     #     avg0 = sum0 / count
    #     #     avg1 = sum1 / count
    #     #     dic[i]['item0'] = avg0
    #     #     dic[i]['item1'] = avg1
    #     # dic['data']= uri
    #     # dic['det']= df2.to_html()
    #     # dic['jp'] = dic.pop(0)
    #     # dic['pj'] = dic.pop(1)


def djmean(request):
    command= request.POST.get('calc', 'off')
    print(command)
    list = Product.objects.all()
    x=0
    count=0
    if command == 'Mean':
        for i in list.iterator():
            y = i.Amount
            x = x + y
            count = count + 1
        return HttpResponse(x / count)

def pie_chart(request):
    return render(request, 'ML/index.html')
    # labels = []
    # data = []
    #
    # queryset = Product.objects.order_by('-Amount')[:5]
    # for x in queryset:
    #     labels.append(x.Branch_Code)
    #     data.append(x.Amount)








