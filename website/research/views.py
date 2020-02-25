from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse

def research(request):
	return HttpResponse("科学研究")
