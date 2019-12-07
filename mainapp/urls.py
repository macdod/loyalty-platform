from django.conf.urls import url,include
from . import views

urlpatterns = [
	url(r'^$', views.index , name = 'index'),
	url(r'^plot/$', views.plot , name = 'plot'),
	url(r'^plotpage/$', views.plotpage , name = 'plotpage'),
	url(r'^plotpage2/$', views.plotpage2 , name = 'plotpage2'),
	url(r'^plotnums/$', views.plotnums , name = 'plotnums'),
]