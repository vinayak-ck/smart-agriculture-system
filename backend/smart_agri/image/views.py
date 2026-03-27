from django.http import JsonResponse

def test_image(request):
    return JsonResponse({"message": "Image API working"})