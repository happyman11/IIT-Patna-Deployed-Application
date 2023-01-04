from django.shortcuts import render
from .models import Document


def  index(request):
    
    return render(request, 'index.html')
# Create your views here.
def  test(request):

    if request.method=='POST':
        file_name=str(request.FILES['file'])
        if file_name.split('.')[1] == 'xlsx':

            Document_obj=Document.objects.create(docfile=request.FILES['file'])
            Document_obj.save()

        else:
            
            print("file Type")

    
        return render(request, 'test.html')
    
    else:
        # print("not Post METhor")
    
        return render(request, 'test.html')