from django.shortcuts import render
from django.http import HttpResponse
from . import amdocs_loyalty
def index(Request):
	return render(Request, "mainapp/index.html")


def plot(Request):
	if Request.method == 'POST':

		val = Request.POST['val']
		amdocs_loyalty.ideal(val)

		return HttpResponse("Success")

def plotnums(Request):
	if Request.method == 'POST':

		val = Request.POST['str']
		amdocs_loyalty.LOYALTY(val)

		return HttpResponse("Success")

def plotpage(Request):
	return render(Request, "mainapp/plots/plot1.html")

def plotpage2(Request):
	return render(Request, "mainapp/plots/plot2.html")