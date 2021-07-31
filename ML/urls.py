from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
    path('', views.home, name='home'),
path('a', views.data_prev, name='dataprev'),
path('b', views.select_features_algo, name='select_features'),
path('d', views.cluster, name='cluster'),
path('e', views.contact, name='contact'),
path('c', views.pred_feat_sel, name='pred_feat_sel'),
path('f', views.predict, name='predict'),
path('mean', views.djmean, name='djmean'),

]

