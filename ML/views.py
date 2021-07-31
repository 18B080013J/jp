from django.shortcuts import render,HttpResponse
from .models import Product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import io, base64
import urllib
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
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
    c=0
    for x, y in d2.items():
        if y == 'on':
            listdj.append(x)
            c=c+1
    if algo=='1':
        list = Product.objects.all().values()
        df = pd.DataFrame(list)
        df2 = df[listdj]
        mms = MinMaxScaler()
        mms.fit(df2)
        X1 = mms.transform(df2)
        model = KMeans(n_clusters=3)
        label = model.fit_predict(X1)
        colums =df2.columns
        df2['labels'] = label
        df2['Branch_Code'] = df['Branch_Code']
        u_labels = np.unique(label)



        dflist = []
        i = 0
        while i < 3:
            dflist.append(df2[df2['labels'] == i])
            jp = dflist[i]

            y = jp.mean()
            z = y.to_dict()

            jp = jp.append(z, ignore_index=True)
            dflist[i] = jp
            i = i + 1



        x = 0
        k = 0
        while x < c:
            j = x + 1
            while j < c:
                plt.subplot(1, 3, k + 1)
                k=k+1
                for i in u_labels:
                    plt.scatter(X1[label == i, x], X1[label == i, j], label=i)
                plt.xlabel(colums[x])
                plt.ylabel(colums[j])
                j = j + 1
            x = x + 1
        plt.legend(bbox_to_anchor =(0.75, 1.15), ncol = 2)
        plt.suptitle('cluster')
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.5,
                            hspace=0.5)

        fig = plt.gcf()
        buf = io.BytesIO()
        plt.close(fig)

        fig.savefig(buf, format='png',facecolor = 'tomato',dpi = 1000)
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)

        context = {'data': uri,
                   'dflist0': dflist[0].to_html,
                   'dflist1': dflist[1].to_html,
                   'dflist2': dflist[2].to_html}
        return render(request, 'ML/cluster.html', context)
    elif algo=='2':
        list = Product.objects.all().values()
        df = pd.DataFrame(list)
        df2 = df[listdj]
        colums = df2.columns
        mms = MinMaxScaler()
        mms.fit(df2)
        X1 = mms.transform(df2)


        dbscan = DBSCAN(eps=0.1, min_samples=2)

        label = dbscan.fit_predict(X1)
        u_labels = np.unique(label)
        df2['labels'] = label
        df2['Branch_Code'] = df['Branch_Code']
        p = len(u_labels)

        dflist = []
        i = -1
        while i < p-1:
            dflist.append(df2[df2['labels'] == i])
            jp = dflist[i]

            y = jp.mean()
            z = y.to_dict()

            jp = jp.append(z, ignore_index=True)
            dflist[i] = jp
            i = i + 1

        l=len(dflist)
        v=0
        tlist=[]
        while v<l:
            tlist.append(v)
            v=v+1
        d = dict.fromkeys(tlist)
        for i in tlist:
            d[i] = dflist[i]

        x = 0
        k = 0
        while x < c:
            j = x + 1
            while j < c:
                plt.subplot(1, 3, k + 1)
                k = k + 1
                for i in u_labels:
                    plt.scatter(X1[label == i, x], X1[label == i, j], label=i)
                plt.xlabel(colums[x])
                plt.ylabel(colums[j])
                j = j + 1
            x = x + 1
        plt.legend(bbox_to_anchor=(0.75, 1.15), ncol=5)
        plt.suptitle('cluster')
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.5,
                            hspace=0.5)
        fig = plt.gcf()
        buf = io.BytesIO()
        plt.close(fig)
        fig.savefig(buf, format='png',facecolor = 'tomato',dpi = 1000)
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)

        context = {'data': uri,
                   'det': df2.to_html,
                   'dflist0': dflist[0].to_html,
                   'dflist1': dflist[1].to_html,
                   'dflist2': dflist[2].to_html}
        # for a,b in d.items():
        #     context[a]= b + ".to_html"



        return render(request, 'ML/cluster.html', context)


    elif algo == '3':
        list = Product.objects.all().values()
        df = pd.DataFrame(list)
        df2 = df[listdj]
        colums = df2.columns
        mms = MinMaxScaler()
        mms.fit(df2)
        X1 = mms.transform(df2)

        dbscan = AgglomerativeClustering(n_clusters=3)

        label = dbscan.fit_predict(X1)
        u_labels = np.unique(label)
        df2['labels'] = label
        df2['Branch_Code'] = df['Branch_Code']

        dflist = []
        i = 0
        while i < 3:
            dflist.append(df2[df2['labels'] == i])
            jp = dflist[i]

            y = jp.mean()
            z = y.to_dict()

            jp = jp.append(z, ignore_index=True)
            dflist[i] = jp
            i = i + 1

        x = 0
        k = 0
        while x < c:
            j = x + 1
            while j < c:
                plt.subplot(1, 3, k + 1)
                k = k + 1
                for i in u_labels:
                    plt.scatter(X1[label == i, x], X1[label == i, j], label=i)
                plt.xlabel(colums[x])
                plt.ylabel(colums[j])
                j = j + 1
            x = x + 1
        plt.legend(bbox_to_anchor=(0.75, 1.15), ncol=2)
        plt.suptitle('cluster')
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.5,
                            hspace=0.5)
        fig = plt.gcf()
        buf = io.BytesIO()
        plt.close(fig)
        fig.savefig(buf, format='png',facecolor = 'tomato',dpi = 1000)
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)

        context = {'data': uri,
                   'dflist0': dflist[0].to_html,
                   'dflist1': dflist[1].to_html,
                   'dflist2': dflist[2].to_html}
        return render(request, 'ML/cluster.html', context)



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
def pred_feat_sel(request):
    return render(request, 'ML/pred_feat_sel.html')

def predict(request):
    algo = request.POST.get('algo')
    list = Product.objects.all().values()
    df = pd.DataFrame(list)
    djlist2=[]
    j=float(request.POST.get('NIFTY_PE','0'))
    f = float(request.POST.get('Type_of_Purchase', '0'))
    # print(j)
    # print(f)
    djlist2.append(j)
    djlist2.append(f)
    # print(djlist)
    k=np.array(djlist2).reshape((1,-1))
    # print(k)

    # d2 = {"Amount": Amount, "NIFTY_PE": NIFTY_PE, "Type_of_Purchase": Type_of_Purchase}
    # listdj = []
    # c = 0
    # for x, y in d2.items():
    #     if y == 'on':
    #         listdj.append(x)
    #         c = c + 1
    X_R1=df[['NIFTY_PE','Type_of_Purchase']]
    X_R1=X_R1.to_numpy().reshape((281, 2))
    # print(X_R1)
    y_R1=df['Amount']
    y_R1 = y_R1.to_numpy()
    # print(y_R1)
    X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state=0)


    if algo == '1':
        knnreg = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
        final = knnreg.predict(k)
        context = {'data': final,
                   }

        return render(request, 'ML/predict.html', context)
    elif algo == '2':
        linreg = LinearRegression().fit(X_train, y_train)
        c=linreg.intercept_
        m=linreg.coef_
        final=m[0]*j + m[1]*f + c
        context = {'data': final,
                   }

        return render(request, 'ML/predict.html', context)
    elif algo == '3':
        return HttpResponse('work under progress')






def contact(request):
    return render(request, 'ML/index.html')








